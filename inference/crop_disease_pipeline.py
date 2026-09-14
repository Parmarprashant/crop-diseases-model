"""
AgriVision AI — Production Diagnostic Pipeline with Crop-Aware Disease Resolver.

Integrates:
1. Model A Diagnostic Expert (EfficientNet-B5+CBAM, Quality Gate, Auto-Leaf Focus, OOD).
2. Dedicated Crop Expert (ConvNeXt-Tiny calibrated with optimal T_crop).
3. Crop-Aware Disease Resolver (recovers in-crop disease evidence without hard argmax masking).
"""
import os
import sys
from typing import Optional, Dict, Any
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.pipeline import HierarchicalAgriDiagnosticPipeline
from inference.diagnosis_schema import StandardizedDiagnosisResponse
from inference.crop_fusion import CalibratedCropExpertInference, CropExpertOutput
from inference.crop_disease_resolver import (
    CropAwareDiseaseResolver,
    CropDiseaseResolverResult,
    ResolverTelemetry
)

class CropDiseasePipeline:
    """
    Master Crop-Aware Disease Resolver Pipeline.
    Combines Model A with the Calibrated Dedicated Crop Expert under
    continuous, cumulative in-crop evidence gating.
    """
    def __init__(
        self,
        base_pipeline: HierarchicalAgriDiagnosticPipeline,
        crop_expert: CalibratedCropExpertInference,
        resolver: CropAwareDiseaseResolver
    ):
        self.base_pipeline = base_pipeline
        self.crop_expert = crop_expert
        self.resolver = resolver
        self.last_resolver_result: Optional[CropDiseaseResolverResult] = None

    def diagnose(
        self,
        image: Image.Image,
        user_crop_hint: Optional[str] = None
    ) -> StandardizedDiagnosisResponse:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Run base Model A pipeline (Quality gate, Auto-Leaf Focus, EfficientNet-B5 CBAM, OOD)
        res = self.base_pipeline.diagnose(image, user_crop_hint=user_crop_hint)

        # 2. Run Independent Calibrated Crop Expert
        crop_output: CropExpertOutput = self.crop_expert.predict(image)

        # Extract top-2 crops from crop expert
        crop_dist = crop_output.distribution
        sorted_crops = sorted(crop_dist.items(), key=lambda x: x[1], reverse=True)
        top1_crop = crop_output.crop
        top2_crop = sorted_crops[1][0] if len(sorted_crops) > 1 else "unknown"

        # 3. Retrieve canonical disease probabilities from Model A
        canonical_probs = getattr(self.base_pipeline, "last_canonical_probs", {})
        if not canonical_probs and res.primary_model.top_candidates:
            canonical_probs = {c["class_name"]: c["confidence"] for c in res.primary_model.top_candidates}

        # 4. Resolve crop and disease via Crop-Aware Disease Resolver
        resolver_res = self.resolver.resolve(
            model_a_crop=res.crop.name,
            model_a_diag=res.diagnosis.name,
            model_a_conf=res.primary_model.raw_confidence or 0.0,
            model_a_accepted=res.primary_model.accepted,
            canonical_disease_probs=canonical_probs,
            crop_top1=top1_crop,
            crop_top2=top2_crop,
            crop_conf=crop_output.confidence,
            crop_margin=crop_output.margin,
            crop_entropy=crop_output.entropy,
            crop_probs=crop_dist
        )
        self.last_resolver_result = resolver_res

        # 5. Harmonize standardized response
        res.crop.name = resolver_res.final_crop
        if resolver_res.final_crop == crop_output.crop:
            res.crop.confidence = crop_output.confidence
            res.crop.status = "VERIFIED" if crop_output.is_confident else "TENTATIVE"
        
        res.diagnosis.name = resolver_res.final_disease
        res.primary_model.raw_confidence = resolver_res.final_confidence
        res.primary_model.accepted = resolver_res.final_accepted

        if not resolver_res.final_accepted:
            res.primary_model.status = "rejected"
            res.diagnosis.status = "INSUFFICIENT_EVIDENCE"
            if resolver_res.telemetry.resolver_reason not in res.primary_model.rejection_reasons:
                res.primary_model.rejection_reasons.append(resolver_res.telemetry.resolver_reason)

        return res
