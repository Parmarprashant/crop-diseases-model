"""
AgriVision AI — Hierarchical Candidate Inference Pipeline & Soft Consistency Engine.

Integrates:
1. EfficientNetB5_CBAM_Hierarchical candidate model (shared backbone, CBAM, disease head, dedicated crop head)
2. Soft Crop-Disease Consistency Re-Weighting Formula:
      S(d) = P(disease = d) * [P_crop(C(d))]^alpha
   where:
      - d is canonical disease class
      - C(d) is the botanical crop family corresponding to disease d
      - P_crop(c) is the predicted probability for crop family c from the dedicated crop head
      - alpha in [0.0, 1.0] is a frozen consistency factor tuned on val_split.csv
3. Telemetry Logging for every inference:
      - model_a_disease_score
      - candidate_disease_only_score
      - candidate_crop_conditioned_score
      - final_production_decision
4. Preserves all 10 defensive pipeline stages (Quality Gate, Auto-Leaf Focus, Energy OOD, Plant Part, etc.)
"""
import os
import sys
import json
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.hierarchical_cbam import EfficientNetB5_CBAM_Hierarchical
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
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
from inference.pipeline import CanonicalAggregator, AGGREGATOR, HierarchicalAgriDiagnosticPipeline


class HierarchicalDiseaseClassifierInference:
    """
    Inference wrapper for EfficientNetB5_CBAM_Hierarchical candidate model.
    Evaluates both the 281-class disease classifier and the 41-class botanical crop head.
    """
    def __init__(
        self,
        checkpoint_path: str = "weights/efficientnet_b5_cbam_hierarchical_candidate.pt",
        class_names_path: str = "weights/class_names.txt",
        crop_names_path: str = "weights/crop_names.txt",
        device: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = [l.strip() for l in f if l.strip()]
        self.num_disease_classes = len(self.class_names)

        with open(crop_names_path, "r", encoding="utf-8") as f:
            self.crop_names = [l.strip() for l in f if l.strip()]
        self.num_crop_classes = len(self.crop_names)

        self.model = EfficientNetB5_CBAM_Hierarchical(
            num_disease_classes=self.num_disease_classes,
            num_crop_classes=self.num_crop_classes,
            pretrained=False
        )

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[HierarchicalInference] Successfully loaded candidate weights from: {checkpoint_path}")
        else:
            print(f"[HierarchicalInference] Warning: Checkpoint {checkpoint_path} not found.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, top_k: int = 10) -> Dict[str, Any]:
        """
        Runs candidate inference on a PIL image.
        Returns disease and crop logits, probabilities, top predictions, and CBAM attention map.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda") if self.device.type == "cuda" else torch.no_grad():
                disease_logits, crop_logits = self.model(tensor, return_crop=True)

            disease_probs = F.softmax(disease_logits, dim=1).squeeze(0)
            crop_probs = F.softmax(crop_logits, dim=1).squeeze(0)

            # Disease Top-K
            top_d_probs, top_d_indices = torch.topk(disease_probs, k=min(top_k, self.num_disease_classes))
            top_d_probs = top_d_probs.cpu().tolist()
            top_d_indices = top_d_indices.cpu().tolist()

            best_d_idx = top_d_indices[0]
            best_d_class = self.class_names[best_d_idx]
            best_d_conf = float(top_d_probs[0])

            top_disease_predictions = [
                {
                    "class_name": self.class_names[idx],
                    "confidence": round(float(prob), 4)
                }
                for idx, prob in zip(top_d_indices, top_d_probs)
            ]

            # Crop Top-K
            top_c_probs, top_c_indices = torch.topk(crop_probs, k=min(top_k, self.num_crop_classes))
            top_c_probs = top_c_probs.cpu().tolist()
            top_c_indices = top_c_indices.cpu().tolist()

            best_c_idx = top_c_indices[0]
            best_c_crop = self.crop_names[best_c_idx]
            best_c_conf = float(top_c_probs[0])

            top_crop_predictions = [
                {
                    "crop": self.crop_names[idx],
                    "confidence": round(float(prob), 4)
                }
                for idx, prob in zip(top_c_indices, top_c_probs)
            ]

            all_crop_probs_dict = {
                self.crop_names[i]: float(crop_probs[i].item())
                for i in range(self.num_crop_classes)
            }

            attn_map = self.model.get_spatial_attention_map(tensor)

            raw_disease_logits = disease_logits.squeeze(0).float().cpu().numpy()
            raw_disease_probs = disease_probs.float().cpu().numpy()
            raw_crop_logits = crop_logits.squeeze(0).float().cpu().numpy()
            raw_crop_probs = crop_probs.float().cpu().numpy()
            self.last_crop_result = {
                "predicted_crop": best_c_crop,
                "crop_confidence": round(best_c_conf, 4),
                "top_crop_predictions": top_crop_predictions,
                "all_crop_probabilities": all_crop_probs_dict
            }

        return {
            "predicted_class": best_d_class,
            "confidence": round(best_d_conf, 4),
            "top_predictions": top_disease_predictions,
            "attention_map": attn_map,
            "is_confident": best_d_conf >= 0.80,
            "raw_logits": raw_disease_logits,
            "all_probabilities": raw_disease_probs,
            "predicted_crop": best_c_crop,
            "crop_confidence": round(best_c_conf, 4),
            "top_crop_predictions": top_crop_predictions,
            "crop_probabilities": raw_crop_probs,
            "all_crop_probabilities": all_crop_probs_dict,
            "raw_crop_logits": raw_crop_logits
        }


class SoftCropDiseaseConsistencyEngine:
    """
    Soft Crop-Disease Consistency Re-Weighting Engine.
    Implements: S(d) = P(disease = d) * [P_crop(C(d))]^alpha
    """
    def __init__(
        self,
        alpha: float = 0.5,
        mapping_path: str = "weights/canonical_disease_to_crop.json"
    ):
        self.alpha = float(alpha)
        self.canonical_to_crop: Dict[str, str] = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.canonical_to_crop = json.load(f)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def apply_consistency(
        self,
        canonical_probs_dict: Dict[str, float],
        crop_probs_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Applies soft consistency weighting across all canonical classes.
        Returns re-weighted canonical distribution and telemetry scores.
        """
        if self.alpha == 0.0 or not crop_probs_dict:
            # Pure disease head (exact baseline behavior)
            sorted_items = sorted(canonical_probs_dict.items(), key=lambda kv: kv[1], reverse=True)
            top_class = sorted_items[0][0] if sorted_items else "Unknown"
            top_conf = sorted_items[0][1] if sorted_items else 0.0
            return {
                "final_prediction": top_class,
                "final_confidence": round(top_conf, 4),
                "reweighted_probabilities": canonical_probs_dict,
                "candidate_disease_only_score": round(top_conf, 4),
                "candidate_crop_conditioned_score": round(top_conf, 4),
                "top_crop_from_head": max(crop_probs_dict, key=crop_probs_dict.get) if crop_probs_dict else "unknown"
            }

        unnormalized = {}
        telemetry = {}

        for d_name, p_d in canonical_probs_dict.items():
            crop_family = self.canonical_to_crop.get(d_name)
            if not crop_family:
                # Fallback to extract from prefix e.g. "Apple - Scab" -> "apple"
                crop_family = d_name.split("-")[0].strip().lower()

            # Handle rice / paddy synonym
            p_crop = crop_probs_dict.get(crop_family, 0.0)
            if crop_family == "rice" and "paddy" in crop_probs_dict:
                p_crop = max(p_crop, crop_probs_dict.get("paddy", 0.0))
            elif crop_family == "paddy" and "rice" in crop_probs_dict:
                p_crop = max(p_crop, crop_probs_dict.get("rice", 0.0))

            # Crop conditioning multiplier: [P_crop(C(d))]^alpha (with small floor for unobserved crops)
            multiplier = max(p_crop, 0.001) ** self.alpha
            score = p_d * multiplier
            unnormalized[d_name] = score

        # Re-normalize to valid probability distribution
        total_mass = sum(unnormalized.values())
        if total_mass > 0:
            reweighted = {k: v / total_mass for k, v in unnormalized.items()}
        else:
            reweighted = canonical_probs_dict

        sorted_reweighted = sorted(reweighted.items(), key=lambda kv: kv[1], reverse=True)
        final_top_class = sorted_reweighted[0][0] if sorted_reweighted else "Unknown"
        final_top_conf = sorted_reweighted[0][1] if sorted_reweighted else 0.0

        disease_only_conf = canonical_probs_dict.get(final_top_class, 0.0)

        return {
            "final_prediction": final_top_class,
            "final_confidence": round(final_top_conf, 4),
            "reweighted_probabilities": reweighted,
            "candidate_disease_only_score": round(disease_only_conf, 4),
            "candidate_crop_conditioned_score": round(final_top_conf, 4),
            "top_crop_from_head": max(crop_probs_dict, key=crop_probs_dict.get) if crop_probs_dict else "unknown"
        }


class DedicatedCropDetector(CropDetector):
    """
    Directly leverages the 41-class botanical crop head from EfficientNetB5_CBAM_Hierarchical
    while falling back safely to CropDetector's multi-class peak affinity if unasserted.
    """
    def __init__(self, classifier: HierarchicalDiseaseClassifierInference):
        super().__init__()
        self.classifier = classifier

    def detect_from_predictions(
        self,
        class_probabilities: Dict[str, float],
        user_crop_hint: Optional[str] = None
    ) -> CropDetectionResult:
        last_crop = getattr(self.classifier, "last_crop_result", None)
        if last_crop and last_crop.get("crop_confidence", 0.0) >= 0.20:
            c_name = last_crop["predicted_crop"].lower().strip()
            c_conf = last_crop["crop_confidence"]
            all_c = last_crop.get("all_crop_probabilities", {})
            return CropDetectionResult(
                crop=c_name,
                confidence=round(c_conf, 4),
                status="VERIFIED" if c_conf >= 0.25 else "TENTATIVE",
                distribution=all_c,
                margin=round(c_conf, 4),
                explanation=f"Dedicated 41-crop neural head detected '{c_name}' with {c_conf*100:.1f}% confidence."
            )
        return super().detect_from_predictions(class_probabilities, user_crop_hint=user_crop_hint)


class HierarchicalCandidatePipeline(HierarchicalAgriDiagnosticPipeline):
    """
    Full 10-step defensive pipeline for Hierarchical Candidate Model.
    Subclasses HierarchicalAgriDiagnosticPipeline to preserve all 10 defensive stages
    while plugging in the dedicated 41-crop neural head and soft consistency engine.
    """
    def __init__(
        self,
        classifier: HierarchicalDiseaseClassifierInference,
        pest_detector: YOLOv8PestDetector,
        segmenter: LesionSegmenter,
        gemini_fallback: GeminiVisionFallback,
        thresholds_path: str = "weights/calibration_thresholds.json",
        alpha: float = 0.0,
        enable_gemini: bool = False
    ):
        super().__init__(
            classifier=classifier,
            pest_detector=pest_detector,
            segmenter=segmenter,
            gemini_fallback=gemini_fallback,
            thresholds_path=thresholds_path,
            enable_gemini=enable_gemini
        )
        self.crop_detector = DedicatedCropDetector(classifier=classifier)
        self.consistency_engine = SoftCropDiseaseConsistencyEngine(alpha=alpha)
