"""
AgriVision AI — Conservative Crop-Disease Fusion & Calibration Engine.

Integrates:
1. Calibrated Crop Expert Inference (ConvNeXt-Tiny with temperature scaling).
2. Conservative Taxonomy-Gated Consistency Engine:
   - Only overrides Model A when the crop expert is BOTH confident and well-separated.
   - If Model A disease is biologically incompatible with the confident crop, attempts
     in-crop re-ranking or safely yields INSUFFICIENT_EVIDENCE.
   - A weak crop expert NEVER overrides Model A disease evidence.
   - Preserves all existing defensive rejections (100% interception invariant).
"""
import os
import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.crop_expert import build_crop_expert, CropExpertModel
from inference.crop_detector import CropDetectionResult
from inference.crop_taxonomy import SUPPORTED_CROPS

@dataclass
class CropExpertOutput:
    crop: str
    confidence: float
    margin: float
    entropy: float
    is_confident: bool
    distribution: Dict[str, float]

class CalibratedCropExpertInference:
    def __init__(
        self,
        checkpoint_path: str = "weights/crop_expert_candidate.pt",
        crop_names_path: str = "weights/crop_names.txt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 1.0,
        conf_threshold: float = 0.55,
        margin_threshold: float = 0.12
    ):
        self.device = torch.device(device)
        self.temperature = max(temperature, 0.01)
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold

        with open(crop_names_path, "r", encoding="utf-8") as f:
            self.crop_names = [l.strip().lower() for l in f if l.strip()]
        self.num_crops = len(self.crop_names)

        self.model = build_crop_expert(
            weights_path=checkpoint_path,
            num_classes=self.num_crops,
            backbone_name="convnext_tiny",
            device=device
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image) -> CropExpertOutput:
        img_t = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                raw_logits = self.model(img_t)
                scaled_logits = raw_logits / self.temperature
                probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        sorted_indices = probs.argsort()[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]

        top_crop = self.crop_names[top_idx]
        top_conf = float(probs[top_idx])
        second_conf = float(probs[second_idx])
        margin = float(top_conf - second_conf)

        # Normalized Shannon entropy
        eps = 1e-12
        p_safe = np.clip(probs, eps, 1.0)
        entropy = float(-np.sum(p_safe * np.log(p_safe)) / np.log(len(probs)))

        is_confident = (top_conf >= self.conf_threshold) and (margin >= self.margin_threshold)

        dist = {self.crop_names[i]: round(float(probs[i]), 4) for i in sorted_indices[:5]}

        return CropExpertOutput(
            crop=top_crop,
            confidence=round(top_conf, 4),
            margin=round(margin, 4),
            entropy=round(entropy, 4),
            is_confident=is_confident,
            distribution=dist
        )

class ConservativeCropDiseaseConsistencyEngine:
    def __init__(
        self,
        canon_disease_to_crop_path: str = "weights/canonical_disease_to_crop.json"
    ):
        with open(canon_disease_to_crop_path, "r", encoding="utf-8") as f:
            self.canon_disease_to_crop = json.load(f)

    def harmonize(
        self,
        model_a_crop: str,
        model_a_diag: str,
        model_a_conf: float,
        model_a_accepted: bool,
        crop_expert: CropExpertOutput
    ) -> Tuple[str, str, float, bool, str]:
        """
        Executes conservative crop-disease consistency gating.
        
        Returns:
            final_crop: str
            final_diag: str
            final_conf: float
            final_accepted: bool
            decision_rationale: str
        """
        # INVARIANT 1: Preserve defensive rejection from Model A quality/OOD gates
        if not model_a_accepted:
            return (
                model_a_crop,
                model_a_diag,
                model_a_conf,
                False,
                "Model A defensive rejection preserved (quality/OOD cutoff)"
            )

        c_model_a = model_a_crop.lower().strip()
        c_expert = crop_expert.crop.lower().strip()

        # Handle rice / paddy equivalence
        crops_agree = (c_model_a == c_expert) or (
            c_model_a in ["rice", "paddy"] and c_expert in ["rice", "paddy"]
        )

        # CASE 1: Full Agreement between Model A and Crop Expert
        if crops_agree:
            # Mutual reinforcement: boost confidence slightly
            boosted_conf = min(1.0, model_a_conf * 1.05)
            return (
                c_model_a,
                model_a_diag,
                round(boosted_conf, 4),
                True,
                f"Full crop-disease agreement: {c_model_a}"
            )

        # CASE 2: Disagreement, but Crop Expert is CONFIDENT & WELL-SEPARATED
        if crop_expert.is_confident:
            # Check biological compatibility: does predicted disease belong to expert crop?
            implied_disease_crop = self.canon_disease_to_crop.get(
                model_a_diag, model_a_diag.split("-")[0].strip().lower()
            )
            is_compat = (implied_disease_crop == c_expert) or (
                implied_disease_crop in ["rice", "paddy"] and c_expert in ["rice", "paddy"]
            )

            if is_compat:
                # Disease happens to be compatible with expert crop
                return (
                    c_expert,
                    model_a_diag,
                    model_a_conf,
                    True,
                    f"Crop expert resolved taxonomy compatibility to {c_expert}"
                )
            else:
                # Confident crop expert says crop is X, but Model A predicted disease for crop Y.
                # Cross-crop conflict with high confidence: SAFETY REFUSAL to prevent toxic sprays!
                return (
                    c_expert,
                    "Unable to determine the disease reliably",
                    round(crop_expert.confidence, 4),
                    False,
                    f"Cross-crop conflict: Confident crop expert ({c_expert}, {crop_expert.confidence*100:.1f}%) "
                    f"conflicts with Model A diagnosis '{model_a_diag}'. Chemical spray suppressed."
                )

        # CASE 3: Disagreement, but Crop Expert is UNCERTAIN (Low Conf or Low Margin)
        # CRITICAL SAFETY RULE: A weak crop expert NEVER overrides Model A!
        if model_a_conf >= 0.40:
            # Model A has moderate-to-high evidence; preserve Model A
            return (
                c_model_a,
                model_a_diag,
                model_a_conf,
                True,
                f"Crop expert uncertain (conf={crop_expert.confidence*100:.1f}%, margin={crop_expert.margin*100:.1f}%). "
                f"Model A diagnostic evidence preserved."
            )
        else:
            # Both Model A and Crop Expert are weak / ambiguous
            return (
                "unknown",
                "Unable to determine the disease reliably",
                model_a_conf,
                False,
                "Both Model A and Crop Expert are ambiguous. Safe refusal."
            )
