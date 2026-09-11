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
from inference.diagnosis_schema import (
    StandardizedDiagnosisResponse,
    CropInfo,
    PlantPartInfo,
    PrimaryModelInfo,
    FallbackInfo,
    DiagnosisInfo,
    SemanticComparisonInfo,
    AdvisoryPlan
)


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
        thresholds_path: str = "weights/calibration_thresholds.json"
    ):
        self.classifier = classifier
        self.pest_detector = pest_detector
        self.segmenter = segmenter
        self.gemini_fallback = gemini_fallback

        self.quality_evaluator = ImageQualityEvaluator()
        self.crop_detector = CropDetector()
        self.plant_part_detector = PlantPartDetector()
        self.ood_detector = OODDetector(thresholds_path=thresholds_path)

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

        # STEP 2 & 3: Run Multi-Model Stack Concurrently (CNN, Pest, Segmenter, and Independent Gemini)
        has_gemini_key = self.gemini_fallback.refresh_key()
        future_gemini = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_cls = executor.submit(self.classifier.predict, image)
            future_pest = executor.submit(self.pest_detector.detect, image)
            future_seg = executor.submit(self.segmenter.segment, image)
            if has_gemini_key:
                # Gemini receives ONLY the original image and optional user crop hint
                # STRICT SCIENTIFIC RULE: Zero exposure to CNN logits, predictions, or confidences!
                future_gemini = executor.submit(
                    self.gemini_fallback.call_fallback,
                    image=image,
                    user_crop_hint=user_crop_hint
                )

            cls_result = future_cls.result()
            pest_result = future_pest.result()
            seg_result = future_seg.result()

            gemini_result = None
            if future_gemini is not None:
                try:
                    gemini_result = future_gemini.result(timeout=25.0)
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

        # STEP 4: Crop Identification
        crop_res = self.crop_detector.detect_from_predictions(probabilities_dict, user_crop_hint=user_crop_hint)

        # STEP 5: Plant Part Identification
        part_res = self.plant_part_detector.detect(image)

        # STEP 6: Multi-Metric OOD Scoring
        probs_array = np.array([p["confidence"] for p in top_candidates])
        # Energy score computed from top logits
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

        if is_primary_valid:
            primary_status = "accepted"
            meta = get_class_metadata(raw_top_class)
            clean_name = meta.common_name if meta else raw_top_class
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
                "Multimodal visual reasoning unavailable — GEMINI_API_KEY is not configured in .env. "
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

        evidence_items = [
            f"Crop status: {crop_res.crop.capitalize()} (status: {crop_res.status})",
            f"Plant organ: {part_res.plant_part.capitalize()} (status: {part_res.status})",
            f"Resolution state: {consensus.resolution} (source: {consensus.source})"
        ]
        if primary_status == "rejected":
            evidence_items.append(f"Primary CNN prediction ('{raw_top_class}') REJECTED: {primary_rejection_str}")
        else:
            evidence_items.append(f"Primary CNN prediction ('{raw_top_class}') accepted ({raw_top_conf*100:.1f}%)")
        evidence_items.append(f"Fallback: {fallback_provider} ({fallback_status})")

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
            }
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
