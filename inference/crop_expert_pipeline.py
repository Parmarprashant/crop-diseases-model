"""
AgriVision AI — Dual-Expert Diagnostic Pipeline.

Combines:
1. Model A Disease Expert (`HierarchicalAgriDiagnosticPipeline` on `weights/efficientnet_b5_cbam_best.pt`)
2. Independent Crop Expert (`CalibratedCropExpertInference` on `weights/crop_expert_candidate.pt`)
3. Conservative Crop-Disease Consistency Engine (`ConservativeCropDiseaseConsistencyEngine`)
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
from inference.crop_fusion import (
    CalibratedCropExpertInference,
    ConservativeCropDiseaseConsistencyEngine,
    CropExpertOutput
)

class CropExpertAugmentedPipeline:
    """
    Production-grade pipeline executing Model A Disease Expert coupled with
    an Independent Dedicated Crop Expert under conservative consistency gating.
    """
    def __init__(
        self,
        base_pipeline: HierarchicalAgriDiagnosticPipeline,
        crop_expert: CalibratedCropExpertInference,
        consistency_engine: Optional[ConservativeCropDiseaseConsistencyEngine] = None
    ):
        self.base_pipeline = base_pipeline
        self.crop_expert = crop_expert
        self.consistency_engine = consistency_engine or ConservativeCropDiseaseConsistencyEngine()

    def diagnose(
        self,
        image: Image.Image,
        user_crop_hint: Optional[str] = None
    ) -> StandardizedDiagnosisResponse:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Run base Model A pipeline (Quality gate, Auto-Leaf Focus, EfficientNet-B5 CBAM, OOD)
        res = self.base_pipeline.diagnose(image, user_crop_hint=user_crop_hint)

        # 2. Run Independent Crop Expert
        crop_output: CropExpertOutput = self.crop_expert.predict(image)

        # 3. Apply Conservative Consistency Engine
        m_crop = res.crop.name
        m_diag = res.diagnosis.name
        m_conf = res.primary_model.raw_confidence or 0.0
        m_acc = res.primary_model.accepted

        f_crop, f_diag, f_conf, f_acc, rationale = self.consistency_engine.harmonize(
            model_a_crop=m_crop,
            model_a_diag=m_diag,
            model_a_conf=m_conf,
            model_a_accepted=m_acc,
            crop_expert=crop_output
        )

        # 4. Harmonize standardized response
        res.crop.name = f_crop
        if f_crop == crop_output.crop:
            res.crop.confidence = crop_output.confidence
            res.crop.status = "VERIFIED" if crop_output.is_confident else "TENTATIVE"
        
        res.diagnosis.name = f_diag
        res.primary_model.raw_confidence = f_conf
        res.primary_model.accepted = f_acc

        if not f_acc:
            res.primary_model.status = "rejected"
            res.diagnosis.status = "INSUFFICIENT_EVIDENCE"
            if rationale not in res.primary_model.rejection_reasons:
                res.primary_model.rejection_reasons.append(rationale)

        return res
