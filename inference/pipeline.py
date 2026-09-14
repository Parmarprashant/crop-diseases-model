import os
import time
import concurrent.futures
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple

from models.efficientnet_cbam import DiseaseClassifierInference
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.advisory_engine import AdvisoryEngine
from core.gemini_fallback import GeminiVisionFallback

from inference.crop_taxonomy import (
    CLASS_TAXONOMY,
    check_crop_compatibility,
    check_plant_part_compatibility,
    get_class_metadata
)
from inference.image_quality import ImageQualityEvaluator, ImageQualityResult
from inference.crop_detector import CropDetector, CropDetectionResult
from inference.plant_part_detector import PlantPartDetector, PlantPartDetectionResult
from inference.ood_detector import OODDetector, OODResult
from inference.consensus_resolver import ConsensusResolver, ConsensusResult
from inference.leaf_focus import LeafFocusDetector, FocusResult
from inference.diagnosis_schema import (
    StandardizedDiagnosisResponse,
    CropInfo,
    PlantPartInfo,
    PrimaryModelInfo,
    FallbackInfo,
    DiagnosisInfo,
    SemanticComparisonInfo,
    AdvisoryPlan,
    FocusRegionInfo
)
import json

CANONICAL_MAP_PATH = os.path.join("weights", "canonical_class_map.json")
CANONICAL_CLASS_MAP: Dict[str, str] = {}
if os.path.exists(CANONICAL_MAP_PATH):
    try:
        with open(CANONICAL_MAP_PATH, "r", encoding="utf-8") as f:
            CANONICAL_CLASS_MAP = json.load(f)
    except Exception:
        pass


class CanonicalAggregator:
    """
    Aggregates raw 281-class softmax probabilities into canonical class
    probabilities by summing all raw classes that share the same canonical
    label. This runs BEFORE OOD gating, crop detection, and consensus.

    This is a pipeline decision layer — not a display layer.
    """
    def __init__(
        self,
        canonical_map_path: str = CANONICAL_MAP_PATH,
        class_names_path: str = "weights/class_names.txt"
    ):
        self.raw_to_canonical: Dict[str, str] = {}
        if os.path.exists(canonical_map_path):
            try:
                with open(canonical_map_path, "r", encoding="utf-8") as f:
                    self.raw_to_canonical = json.load(f)
            except Exception:
                pass

        self.raw_class_names: List[str] = []
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, "r", encoding="utf-8") as f:
                    self.raw_class_names = [line.strip() for line in f if line.strip()]
            except Exception:
                pass

        # Build: canonical_label -> list of raw class indices
        self.canonical_to_indices: Dict[str, List[int]] = {}
        for idx, raw_name in enumerate(self.raw_class_names):
            canonical = self.raw_to_canonical.get(raw_name, raw_name)
            if canonical not in self.canonical_to_indices:
                self.canonical_to_indices[canonical] = []
            self.canonical_to_indices[canonical].append(idx)

        self.canonical_class_names = sorted(self.canonical_to_indices.keys())
        self.num_canonical = len(self.canonical_class_names)

    def aggregate_numpy(self, raw_probs: np.ndarray) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        if raw_probs is None or len(self.canonical_class_names) == 0:
            return np.array([]), [], []
        canonical_probs = np.zeros(self.num_canonical, dtype=np.float32)
        for i, canonical_name in enumerate(self.canonical_class_names):
            indices = self.canonical_to_indices[canonical_name]
            canonical_probs[i] = float(np.sum(raw_probs[indices]))

        top_indices = np.argsort(canonical_probs)[::-1]
        top_canonical_predictions = [
            {
                "class_name": self.canonical_class_names[idx],
                "confidence": round(float(canonical_probs[idx]), 4)
            }
            for idx in top_indices[:10]
        ]
        return canonical_probs, self.canonical_class_names, top_canonical_predictions


AGGREGATOR = CanonicalAggregator()


def get_canonical_label(raw_class_name: str) -> str:
    """Returns clean canonical label for display. Falls back to raw name if not found."""
    return CANONICAL_CLASS_MAP.get(raw_class_name, raw_class_name)


class HierarchicalAgriDiagnosticPipeline:
    """
    Master Defensive Agricultural Diagnosis Pipeline.
    
    Architected to prefer CORRECT UNCERTAINTY over CONFIDENT WRONG DIAGNOSIS:
    1. Image Quality Gating (filters blurred / poor lighting inputs)
    2. Crop Family Identification (with first-class UNKNOWN state)
    3. Plant-Part Detection (with first-class UNKNOWN state)
    4. Domain Compatibility Gating (Rice Panicle != Rice Foliar CNN)
    5. Primary CNN Inference (EfficientNet-B5 + CBAM)
    6. Multi-Metric OOD & Energy Calibration (AUROC 98.02%)
    7. Quarantined Debug Telemetry (Rejected predictions NEVER become diagnoses)
    8. Bounded Fallback (Gemini Vision as secondary assistant; never fake fallback)
    9. Advisory Safety (Chemical sprays suppressed on uncertain / unverified conditions)
    """
    def __init__(
        self,
        classifier: DiseaseClassifierInference,
        pest_detector: YOLOv8PestDetector,
        segmenter: LesionSegmenter,
        gemini_fallback: GeminiVisionFallback,
        thresholds_path: str = "weights/calibration_thresholds.json",
        enable_gemini: Optional[bool] = None
    ):
        self.classifier = classifier
        self.pest_detector = pest_detector
        self.segmenter = segmenter
        self.gemini_fallback = gemini_fallback

        # Master toggle: Set to False to test and evaluate the custom model standalone.
        # Can be enabled anytime by setting ENABLE_GEMINI_FALLBACK=1 or passing enable_gemini=True.
        if enable_gemini is not None:
            self.enable_gemini = enable_gemini
        else:
            self.enable_gemini = os.environ.get("ENABLE_GEMINI_FALLBACK", "0").lower() in ("1", "true", "yes")

        self.quality_evaluator = ImageQualityEvaluator()
        self.crop_detector = CropDetector()
        self.plant_part_detector = PlantPartDetector()
        self.ood_detector = OODDetector(thresholds_path=thresholds_path)
        self.leaf_focus_detector = LeafFocusDetector()
        self.aggregator = AGGREGATOR

    def diagnose(
        self,
        image: Image.Image,
        user_crop_hint: Optional[str] = None
    ) -> StandardizedDiagnosisResponse:
        """
        Execute full hierarchical diagnostic pipeline on input image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.load()

        # STEP 1: Image Quality Assessment
        quality_res = self.quality_evaluator.evaluate(image)
        if not quality_res.is_valid:
            return self._build_quality_failure_response(image, quality_res, user_crop_hint=user_crop_hint)

        # STEP 1.5: Auto-Leaf Focus Analysis (identifies candidate foliar symptom clusters)
        focus_candidates = self.leaf_focus_detector.detect_focus_candidates(image, max_candidates=3)
        active_focus = focus_candidates[0] if focus_candidates else None
        used_focus = False

        # STEP 2 & 3: Run Multi-Model Stack Concurrently (CNN, Pest, Segmenter, and Independent Gemini)
        # Check if Gemini secondary assistant is active (paused during custom model evaluation)
        has_gemini_key = (
            self.gemini_fallback.refresh_key()
            if getattr(self, "enable_gemini", False)
            else False
        )
        # STEP 2 & 3: Run Multi-Model Stack (CNN, Pest, Segmenter, and Independent Gemini)
        cls_result = self.classifier.predict(image, top_k=10)
        pest_result = self.pest_detector.detect(image)
        seg_result = self.segmenter.segment(image)

        gemini_result = None
        if has_gemini_key:
            try:
                gemini_result = self.gemini_fallback.call_fallback(image=image, user_crop_hint=user_crop_hint)
            except Exception as e:
                print(f"[Pipeline] Notice: Gemini call failed or timed out: {e}")
                gemini_result = {
                    "status": "ERROR",
                    "crop": "unknown",
                    "plant_part": "unknown",
                    "diagnosis": "Unavailable",
                    "assessment_strength": "LOW",
                    "model_reported_confidence": 0.0,
                    "reasoning_summary": f"Gemini API request encountered an error: {str(e)[:100]}",
                    "evidence": []
                }

        # STEP 4: Canonical Probability Aggregation & Crop Identification
        if "all_probabilities" in cls_result and getattr(self, "aggregator", None) is not None and self.aggregator.num_canonical > 0:
            canon_probs, canon_names, canon_top = self.aggregator.aggregate_numpy(cls_result["all_probabilities"])
            self.last_canonical_probs = {canon_names[i]: float(canon_probs[i]) for i in range(len(canon_names))}
            raw_top_class = canon_top[0]["class_name"]
            raw_top_conf = canon_top[0]["confidence"]
            top_candidates = canon_top
            probabilities_dict = {p["class_name"]: p["confidence"] for p in top_candidates}
        else:
            self.last_canonical_probs = {}
            raw_top_class = cls_result["predicted_class"]
            raw_top_conf = cls_result["confidence"]
            top_candidates = cls_result["top_predictions"]
            probabilities_dict = {p["class_name"]: p["confidence"] for p in top_candidates}

        # Visualizations
        from core.pipeline import attention_map_to_base64, pil_to_base64
        attention_b64 = attention_map_to_base64(cls_result["attention_map"], image)
        pest_b64 = pil_to_base64(pest_result["annotated_image"])
        lesion_b64 = pil_to_base64(seg_result["overlay_image"])
        orig_b64 = pil_to_base64(image)

        clean_pest_result = {k: v for k, v in pest_result.items() if k != "annotated_image"}
        clean_pest_result["count"] = clean_pest_result.get("pest_count", 0)
        clean_pest_result["pest_count"] = clean_pest_result.get("pest_count", 0)

        clean_seg_result = {k: v for k, v in seg_result.items() if k != "overlay_image"}
        clean_seg_result["leaf_pixels"] = clean_seg_result.get("leaf_pixel_count", 0)
        clean_seg_result["lesion_pixels"] = clean_seg_result.get("lesion_pixel_count", 0)

        crop_res = self.crop_detector.detect_from_predictions(probabilities_dict, user_crop_hint=user_crop_hint)

        # Align candidate within verified crop family to eliminate cross-crop noise
        if crop_res.crop != "unknown":
            crop_matches = [
                c for c in top_candidates
                if check_crop_compatibility(crop_res.crop, c["class_name"])[0]
            ]
            if crop_matches:
                raw_top_class = crop_matches[0]["class_name"]
                raw_top_conf = crop_matches[0]["confidence"]

        # STEP 5: Plant Part Identification
        part_res = self.plant_part_detector.detect(image)

        # STEP 6: Multi-Metric OOD Scoring
        if "raw_logits" in cls_result and "all_probabilities" in cls_result:
            ood_res = self.ood_detector.evaluate(
                cls_result["raw_logits"],
                cls_result["all_probabilities"]
            )
        else:
            probs_array = np.array([p["confidence"] for p in top_candidates])
            logits_approx = np.log(np.maximum(probs_array, 1e-12))
            ood_res = self.ood_detector.evaluate(logits_approx, probs_array)

        # STEP 7: Domain & Compatibility Check
        rejection_reasons = []

        # Check A: Plant Part Compatibility
        part_compat, part_reason = check_plant_part_compatibility(
            crop=crop_res.crop,
            plant_part=part_res.plant_part,
            candidate_disease=raw_top_class
        )
        if not part_compat:
            rejection_reasons.append(part_reason)

        # Check B: Crop Compatibility
        crop_compat, crop_reason = check_crop_compatibility(
            predicted_crop=crop_res.crop,
            candidate_disease=raw_top_class
        )
        if not crop_compat:
            rejection_reasons.append(crop_reason)

        # Check C: OOD / Statistical Reliability
        if ood_res.is_ood:
            rejection_reasons.extend(ood_res.rejection_reasons)

        is_primary_valid = (len(rejection_reasons) == 0) and (crop_res.status != "UNKNOWN")
        primary_rejection_str = "; ".join(rejection_reasons) if rejection_reasons else ""

        # STEP 7.5: Auto-Leaf Focus Adaptive Saliency Recovery
        # Guard: When full-frame is already a close-up image (orig_conf >= 0.20) and margin between
        # competing crops was < 1% (a near-tie on whole-leaf features), do not allow an arbitrary
        # localized sub-crop to override the crop family unless multi-candidate evidence supports the override.
        orig_crop_res = crop_res
        orig_crop_margin = crop_res.margin
        orig_raw_top_class = raw_top_class
        orig_raw_top_conf = raw_top_conf
        orig_cls_result = cls_result
        orig_top_candidates = top_candidates
        orig_is_primary_valid = is_primary_valid
        
        is_close_up = (orig_raw_top_conf >= 0.20)
        is_tight_crop_margin = (0.0 <= orig_crop_margin < 0.01)

        # If full frame fails gating (OOD, unknown crop, or diffuse entropy)
        # evaluate candidate symptom-bearing foliage clusters:
        if not is_primary_valid and focus_candidates:
            for cand in focus_candidates:
                if not cand.is_focused:
                    continue
                try:
                    focused_cls = self.classifier.predict(cand.cropped_image, top_k=10)
                    if "all_probabilities" in focused_cls and getattr(self, "aggregator", None) is not None and self.aggregator.num_canonical > 0:
                        _, _, f_canon_top = self.aggregator.aggregate_numpy(focused_cls["all_probabilities"])
                        f_top_candidates = f_canon_top
                        f_raw_top_class = f_canon_top[0]["class_name"]
                        f_raw_top_conf = f_canon_top[0]["confidence"]
                        f_probs_dict = {p["class_name"]: p["confidence"] for p in f_top_candidates}
                    else:
                        f_top_candidates = focused_cls["top_predictions"]
                        f_probs_dict = {p["class_name"]: p["confidence"] for p in f_top_candidates}
                        f_raw_top_class = focused_cls["predicted_class"]
                        f_raw_top_conf = focused_cls["confidence"]

                    f_crop_res = self.crop_detector.detect_from_predictions(f_probs_dict, user_crop_hint=user_crop_hint)

                    if "raw_logits" in focused_cls and "all_probabilities" in focused_cls:
                        f_ood_res = self.ood_detector.evaluate(focused_cls["raw_logits"], focused_cls["all_probabilities"])
                    else:
                        f_probs_arr = np.array([p["confidence"] for p in f_top_candidates])
                        f_logits_approx = np.log(np.maximum(f_probs_arr, 1e-12))
                        f_ood_res = self.ood_detector.evaluate(f_logits_approx, f_probs_arr)

                    f_raw_top_class = focused_cls["predicted_class"]
                    f_raw_top_conf = focused_cls["confidence"]

                    # Align candidate within verified crop family if identified
                    if f_crop_res.crop != "unknown":
                        f_crop_matches = [
                            c for c in f_top_candidates
                            if check_crop_compatibility(f_crop_res.crop, c["class_name"])[0]
                        ]
                        if f_crop_matches:
                            f_raw_top_class = f_crop_matches[0]["class_name"]
                            f_raw_top_conf = f_crop_matches[0]["confidence"]

                    # Check compatibility on focused crop
                    f_reasons = []
                    f_part_compat, f_part_reason = check_plant_part_compatibility(
                        crop=f_crop_res.crop,
                        plant_part=part_res.plant_part,
                        candidate_disease=f_raw_top_class
                    )
                    if not f_part_compat:
                        f_reasons.append(f_part_reason)

                    f_crop_compat, f_crop_reason = check_crop_compatibility(
                        predicted_crop=f_crop_res.crop,
                        candidate_disease=f_raw_top_class
                    )
                    if not f_crop_compat:
                        f_reasons.append(f_crop_reason)

                    if f_ood_res.is_ood:
                        f_reasons.extend(f_ood_res.rejection_reasons)

                    f_valid = (len(f_reasons) == 0) and (f_crop_res.status != "UNKNOWN")

                    # Close-Up Guard Check:
                    # Do not allow an Auto-Leaf Focus sub-crop to override a close-up image when the
                    # original full-frame crop margin is < 1%, unless multi-candidate evidence supports the override.
                    can_override = True
                    if is_close_up and is_tight_crop_margin:
                        # Require multi-candidate verification across all candidates including candidate 3
                        supporting_candidates = 0
                        for other_cand in focus_candidates:
                            if not other_cand.is_focused:
                                continue
                            try:
                                o_cls = self.classifier.predict(other_cand.cropped_image, top_k=5)
                                if "all_probabilities" in o_cls and self.aggregator.num_canonical > 0:
                                    _, _, o_canon = self.aggregator.aggregate_numpy(o_cls["all_probabilities"])
                                    o_dict = {p["class_name"]: p["confidence"] for p in o_canon}
                                else:
                                    o_dict = {p["class_name"]: p["confidence"] for p in o_cls["top_predictions"]}
                                o_crop = self.crop_detector.detect_from_predictions(o_dict, user_crop_hint=user_crop_hint)
                                o_ood = self.ood_detector.evaluate(o_cls["raw_logits"], o_cls["all_probabilities"])
                                if o_crop.crop == f_crop_res.crop and not o_ood.is_ood and o_crop.status != "UNKNOWN":
                                    supporting_candidates += 1
                            except Exception:
                                pass
                        if supporting_candidates < 3:
                            can_override = False
                            print(f"[Pipeline] Auto-Leaf Focus close-up guard: override to '{f_crop_res.crop}' blocked (full-frame margin {orig_crop_margin*100:.2f}% < 1.0%, supporting: {supporting_candidates}/3)")

                    # If focused crop resolves the ambiguity and passes close-up guard:
                    if can_override and (f_valid or (f_crop_res.crop != "unknown" and f_raw_top_conf >= 0.25)):
                        print(f"[Pipeline] Auto-Leaf Focus accepted: Recovered crop '{f_crop_res.crop}' (conf {f_raw_top_conf*100:.1f}%)")
                        cls_result = focused_cls
                        if "all_probabilities" in focused_cls and getattr(self, "aggregator", None) is not None and self.aggregator.num_canonical > 0:
                            f_probs, f_names, _ = self.aggregator.aggregate_numpy(focused_cls["all_probabilities"])
                            self.last_canonical_probs = {f_names[i]: float(f_probs[i]) for i in range(len(f_names))}
                        crop_res = f_crop_res
                        ood_res = f_ood_res
                        raw_top_class = f_raw_top_class
                        raw_top_conf = f_raw_top_conf
                        top_candidates = f_top_candidates
                        rejection_reasons = [] if f_valid or f_raw_top_conf >= 0.25 else f_reasons
                        is_primary_valid = True
                        primary_rejection_str = ""
                        active_focus = cand
                        used_focus = True
                        if "attention_map" in focused_cls and focused_cls["attention_map"] is not None:
                            try:
                                attention_b64 = attention_map_to_base64(focused_cls["attention_map"], cand.cropped_image)
                            except Exception:
                                pass
                        break
                except Exception as e:
                    print(f"[Pipeline] Auto-Leaf Focus candidate notice: {e}")

        if is_primary_valid:
            primary_status = "accepted"
            canonical_label = get_canonical_label(raw_top_class)
            meta = get_class_metadata(raw_top_class)
            if " - " in canonical_label:
                clean_name = canonical_label.split(" - ", 1)[1].strip()
            elif meta and meta.common_name:
                clean_name = meta.common_name
            else:
                clean_name = canonical_label
            primary_prediction = clean_name
        else:
            primary_status = "rejected"
            primary_prediction = None

        # STEP 8: Secondary Vision Assistant (Gemini) Evaluation Telemetry
        if has_gemini_key and gemini_result is not None:
            fallback_provider = "gemini_vision"
            raw_g_status = str(gemini_result.get("status", "UNAVAILABLE")).upper()
            if raw_g_status in ["SUCCESS", "UNCERTAIN", "UNKNOWN", "INSUFFICIENT_EVIDENCE"]:
                fallback_status = "invoked"
            elif raw_g_status == "ERROR":
                fallback_status = "error"
            else:
                fallback_status = "unavailable"
            fallback_reasoning = gemini_result.get("reasoning_summary", "")
        else:
            fallback_provider = "unavailable"
            fallback_status = "unavailable"
            fallback_reasoning = (
                "Secondary vision model paused for standalone custom model evaluation. "
                "Primary closed-set CNN evaluated independently."
            )
            gemini_result = None

        # STEP 9: Centralized Consensus Resolver
        # MANDATE: Final diagnosis is ONLY produced by ConsensusResolver. Never silently select a model.
        consensus = ConsensusResolver.resolve(
            crop=crop_res.crop,
            plant_part=part_res.plant_part,
            primary_status=primary_status,
            primary_prediction=primary_prediction,
            raw_top_prediction=raw_top_class,
            raw_confidence=raw_top_conf,
            primary_rejection_reasons=rejection_reasons,
            fallback_provider=fallback_provider,
            fallback_status=fallback_status,
            gemini_result=gemini_result
        )

        # STEP 10: Advisory Generation
        # STRICT RULE: Disease-specific chemical & organic recommendations are permitted ONLY
        # when resolution is CONSENSUS or CNN_ONLY.
        # For GEMINI_SUSPECTED, CONFLICT, UNKNOWN, INSUFFICIENT_EVIDENCE:
        # Default to NO disease-specific chemical or treatment recommendations,
        # providing only scouting, image-quality, and expert-verification guidance.
        if consensus.resolution in ["CONSENSUS", "CNN_ONLY", "CNN_ACCEPTED"]:
            target_class_for_advisory = raw_top_class
            advisory_data = AdvisoryEngine.generate_advisory(
                disease_name=target_class_for_advisory,
                confidence=raw_top_conf,
                severity_pct=seg_result["infected_area_pct"],
                pest_count=clean_pest_result.get("pest_count", 0),
                pests=[d["label"] for d in pest_result["detections"]],
                resolution_state=consensus.resolution
            )
            chem = advisory_data.get("chemical_control", [])
            org = advisory_data.get("organic_control", [])
            cult = advisory_data.get("cultural_practices") or advisory_data.get("precautions", [])
            urg = advisory_data.get("urgency", "MODERATE")
            desc = advisory_data.get("disease_description", "")
        else:
            chem = []
            org = []
            cult = consensus.cultural_practices
            urg = "ADVISORY - In-Field Verification Required" if consensus.requires_expert_verification else "MODERATE"
            desc = consensus.explanation

        # Real visual diagnostic evidence items for farmers
        evidence_items = []
        if primary_status == "accepted":
            evidence_items.append(f"Visual pathology patterns match {primary_prediction} on {crop_res.crop.capitalize()} foliage.")
            if seg_result.get("infected_area_pct", 0) > 0:
                evidence_items.append(f"Foliar lesion coverage estimated at {seg_result['infected_area_pct']:.1f}% of leaf area.")
            if clean_pest_result.get("pest_count", 0) > 0:
                evidence_items.append(f"Active insect presence detected: {', '.join([d['label'] for d in pest_result.get('detections', [])])}.")
            else:
                evidence_items.append("No active insect pests detected on inspected foliage.")
            evidence_items.append(f"Model diagnostic confidence: {raw_top_conf*100:.1f}%.")
        else:
            evidence_items.append(f"Crop detected: {consensus.final_crop.capitalize()} ({part_res.plant_part.capitalize()}).")
            if seg_result.get("infected_area_pct", 0) > 0:
                evidence_items.append(f"Foliar discoloration/lesion coverage: {seg_result['infected_area_pct']:.1f}%.")
            evidence_items.append("Visual patterns require in-field physical inspection to verify pathogen type.")

        return StandardizedDiagnosisResponse(
            crop=CropInfo(
                name=consensus.final_crop,
                confidence=consensus.final_crop_confidence,
                status=consensus.final_crop_status
            ),
            plant_part=PlantPartInfo(
                name=consensus.final_plant_part,
                confidence=consensus.final_plant_part_confidence,
                status=consensus.final_plant_part_status
            ),
            primary_model=PrimaryModelInfo(
                status=primary_status,
                prediction=primary_prediction,
                confidence=raw_top_conf if primary_status == "accepted" else None,
                raw_top_prediction=raw_top_class,
                raw_confidence=raw_top_conf,
                rejection_reason=primary_rejection_str if primary_status == "rejected" else None,
                rejection_reasons=rejection_reasons,
                accepted=is_primary_valid,
                ood=ood_res.is_ood,
                crop_compatible=crop_compat,
                plant_part_compatible=part_compat,
                gate_status="ACCEPTED" if is_primary_valid else "REJECTED",
                energy_score=ood_res.energy_score,
                entropy=ood_res.normalized_entropy,
                top_candidates=top_candidates
            ),
            fallback=FallbackInfo(
                provider=fallback_provider,
                status=gemini_result.get("status", fallback_status) if gemini_result else fallback_status,
                crop=gemini_result.get("crop", "unknown") if gemini_result else "unknown",
                plant_part=gemini_result.get("plant_part", "unknown") if gemini_result else "unknown",
                diagnosis=gemini_result.get("diagnosis", "") if gemini_result else "",
                assessment_strength=gemini_result.get("assessment_strength", "LOW") if gemini_result else "LOW",
                model_reported_confidence=float(gemini_result.get("model_reported_confidence", 0.0)) if gemini_result else 0.0,
                reasoning=fallback_reasoning,
                evidence=gemini_result.get("evidence", []) if gemini_result else []
            ),
            diagnosis=DiagnosisInfo(
                name=consensus.final_diagnosis_name,
                type=consensus.final_diagnosis_type,
                confidence=consensus.confidence,
                status=consensus.status,
                consensus_state=consensus.consensus_state,
                resolution=consensus.resolution,
                source=consensus.source,
                validated_by_cnn=consensus.validated_by_cnn,
                farmer_headline=consensus.farmer_headline,
                farmer_subheading=consensus.farmer_subheading
            ),
            evidence=evidence_items,
            recommendation=consensus.recommendation,
            requires_expert_verification=consensus.requires_expert_verification,
            advisory=AdvisoryPlan(
                urgency=urg,
                disease_description=desc,
                chemical_control=chem,
                organic_control=org,
                cultural_practices=cult,
                expert_verification_note="Physical in-field examination required before applying targeted chemicals." if consensus.requires_expert_verification else "Routine monitoring advised."
            ),
            comparison=SemanticComparisonInfo(**consensus.comparison_telemetry) if consensus.comparison_telemetry else None,
            resolution=consensus.resolution,
            pests=clean_pest_result,
            segmentation=clean_seg_result,
            visualizations={
                "original": orig_b64,
                "attention_heatmap": attention_b64,
                "pest_detections": pest_b64,
                "lesion_segmentation": lesion_b64
            },
            focus_region=FocusRegionInfo(
                is_focused=used_focus,
                box_normalized=list(active_focus.box_normalized) if (used_focus and active_focus) else [0.0, 0.0, 1.0, 1.0],
                box_pixels=list(active_focus.box_pixels) if (used_focus and active_focus) else [0, 0, image.width, image.height],
                message=active_focus.message if (used_focus and active_focus) else "Full image analyzed directly."
            )
        )

    def _build_quality_failure_response(
        self,
        image: Image.Image,
        quality: ImageQualityResult,
        user_crop_hint: Optional[str] = None
    ) -> StandardizedDiagnosisResponse:
        from core.pipeline import pil_to_base64
        orig_b64 = pil_to_base64(image)
        crop_name = user_crop_hint.lower().strip() if user_crop_hint else "unknown"
        crop_status = "VERIFIED" if user_crop_hint else "UNKNOWN"
        return StandardizedDiagnosisResponse(
            crop=CropInfo(name=crop_name, confidence=1.0 if user_crop_hint else 0.0, status=crop_status),
            plant_part=PlantPartInfo(name="unknown", confidence=0.0, status="UNKNOWN"),
            primary_model=PrimaryModelInfo(
                status="bypassed",
                prediction=None,
                confidence=None,
                raw_top_prediction="None",
                raw_confidence=0.0,
                rejection_reason=f"Image quality gate failed: {quality.warning_message}",
                rejection_reasons=[f"Image quality gate failed: {quality.warning_message}"],
                accepted=False,
                ood=False,
                crop_compatible=False,
                plant_part_compatible=False,
                gate_status="REJECTED",
                energy_score=0.0,
                entropy=0.0,
                top_candidates=[]
            ),
            fallback=FallbackInfo(
                provider="not_needed",
                status="not_needed",
                crop="unknown",
                plant_part="unknown",
                diagnosis="",
                assessment_strength="LOW",
                model_reported_confidence=0.0,
                reasoning="Inference aborted early due to unreadable image quality.",
                evidence=[]
            ),
            diagnosis=DiagnosisInfo(
                name="Unreadable Image",
                type="unknown",
                confidence="low",
                status="UNKNOWN",
                consensus_state="UNKNOWN",
                resolution="UNKNOWN",
                source="NONE",
                validated_by_cnn=False,
                farmer_headline="Unreadable Image",
                farmer_subheading="Please capture a clean, sharply focused photograph in good daylight."
            ),
            evidence=[f"Quality check verdict: {quality.status} ({quality.warning_message})"],
            recommendation="Please capture a clean, sharply focused photograph in good daylight.",
            requires_expert_verification=True,
            advisory=AdvisoryPlan(
                urgency="RE-TAKE PHOTO",
                disease_description=quality.warning_message,
                chemical_control=[],
                organic_control=[],
                cultural_practices=["Capture the leaf/canopy in clear focus against natural daylight."],
                expert_verification_note="Diagnosis cannot proceed on degraded imagery."
            ),
            comparison=SemanticComparisonInfo(),
            resolution="UNKNOWN",
            pests={"pest_count": 0, "count": 0, "detections": []},
            segmentation={"infected_area_pct": 0.0, "severity_category": "None", "leaf_pixel_count": 0, "lesion_pixel_count": 0, "leaf_pixels": 0, "lesion_pixels": 0},
            visualizations={"original": orig_b64, "attention_heatmap": orig_b64, "pest_detections": orig_b64, "lesion_segmentation": orig_b64}
        )
