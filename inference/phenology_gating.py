import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


@dataclass
class PhenologyGatingResult:
    """
    Diagnostic outcome of botanical phenological and plant organ masking.
    """
    biological_stage: str               # "Vegetative", "Flowering / Squaring", "Boll / Pod Formation", "Maturity / Senescence", "Unknown"
    days_after_sowing: Optional[int]
    crop_type: Optional[str]
    plant_part: Optional[str]
    masked_classes: List[str]          # Classes zeroed out due to biological impossibility
    mask_vector: List[int]             # Binary mask M in {0, 1}^K
    explanation: str


class PhenologyGating:
    """
    Enforces botanical and phenological constraints across days after sowing (DAS)
    and anatomical plant organs.
    
    Biological Stages:
    - Vegetative: 0 to 35 DAS
    - Flowering / Squaring: 35 to 65 DAS
    - Boll / Pod Formation: 65 to 110 DAS
    - Maturity / Senescence: > 110 DAS
    
    Mathematical Formulation:
        z_masked = z_i + log(M_i)
        where M_i in {0, 1}
    """
    
    # Incompatible keywords per stage
    STAGE_INCOMPATIBILITIES = {
        "Vegetative": [
            "boll rot", "pink bollworm", "pod borer", "fruit rot", 
            "panicle blight", "ear rot", "false smut", "head blight",
            "blossom end rot", "bract mosaic", "cigar end rot"
        ],
        "Flowering / Squaring": [
            "boll rot", "internal boll rot", "damping off", "damping-off", 
            "seedling blight", "ripe rot"
        ],
        "Boll / Pod Formation": [
            "damping off", "damping-off", "seedling blight"
        ],
        "Maturity / Senescence": [
            "damping off", "damping-off", "seedling blight", "seedling damping"
        ]
    }
    
    # Incompatible keywords per plant organ
    ORGAN_INCOMPATIBILITIES = {
        "leaf": [
            "bacterial panicle blight", "panicle blight", "false smut",
            "internal boll rot", "fruit rot", "ear rot", "head blight",
            "blossom end rot", "cigar end rot"
        ],
        "panicle": [
            "leaf mold", "target spot", "frog eye leaf spot", "septoria leaf spot"
        ],
        "boll": [
            "leaf mold", "frog eye leaf spot", "septoria leaf spot"
        ]
    }

    @staticmethod
    def compute_biological_stage(das: Optional[int]) -> str:
        """Determines crop biological growth stage from Days After Sowing (DAS)."""
        if das is None or das < 0:
            return "Unknown"
        if das <= 35:
            return "Vegetative"
        elif das <= 65:
            return "Flowering / Squaring"
        elif das <= 110:
            return "Boll / Pod Formation"
        else:
            return "Maturity / Senescence"

    @classmethod
    def construct_mask(
        cls,
        class_names: List[str],
        days_after_sowing: Optional[int] = None,
        crop_type: Optional[str] = None,
        plant_part: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str], str]:
        """
        Constructs binary masking vector M in {0, 1}^K across canonical classes.
        Returns:
            mask: np.ndarray of shape (K,), values in {0.0, 1.0}
            masked_classes: List of class names whose M_i = 0
            explanation: Summary of why classes were masked
        """
        K = len(class_names)
        mask = np.ones(K, dtype=np.float32)
        masked_classes = []
        reasons = []

        stage = cls.compute_biological_stage(days_after_sowing)
        
        # 1. Check phenological stage incompatibilities
        if stage in cls.STAGE_INCOMPATIBILITIES:
            banned_keywords = cls.STAGE_INCOMPATIBILITIES[stage]
            for idx, cname in enumerate(class_names):
                cname_lower = cname.lower()
                for kw in banned_keywords:
                    if kw in cname_lower:
                        mask[idx] = 0.0
                        if cname not in masked_classes:
                            masked_classes.append(cname)
            if masked_classes:
                reasons.append(f"At {days_after_sowing} DAS ({stage} stage), reproductive/seedling conditions cannot occur.")

        # 2. Check plant organ incompatibilities
        if plant_part:
            part_lower = plant_part.lower().strip()
            if part_lower in cls.ORGAN_INCOMPATIBILITIES:
                organ_banned = cls.ORGAN_INCOMPATIBILITIES[part_lower]
                for idx, cname in enumerate(class_names):
                    cname_lower = cname.lower()
                    for kw in organ_banned:
                        if kw in cname_lower:
                            mask[idx] = 0.0
                            if cname not in masked_classes:
                                masked_classes.append(cname)
                if organ_banned:
                    reasons.append(f"Organ '{plant_part}' cannot manifest organ-exclusive diseases.")

        explanation = " ".join(reasons) if reasons else "No phenological or organ incompatibilities detected."
        return mask, masked_classes, explanation

    @classmethod
    def apply_mask_to_logits(
        cls,
        logits: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        days_after_sowing: Optional[int] = None,
        crop_type: Optional[str] = None,
        plant_part: Optional[str] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], PhenologyGatingResult]:
        """
        Applies logit masking:
            z_masked = z_i + log(M_i)
        where M_i == 0 sets logit to -1e9 (-inf).
        """
        mask, masked_classes, explanation = cls.construct_mask(
            class_names=class_names,
            days_after_sowing=days_after_sowing,
            crop_type=crop_type,
            plant_part=plant_part
        )
        
        stage = cls.compute_biological_stage(days_after_sowing)
        result = PhenologyGatingResult(
            biological_stage=stage,
            days_after_sowing=days_after_sowing,
            crop_type=crop_type,
            plant_part=plant_part,
            masked_classes=masked_classes,
            mask_vector=mask.astype(int).tolist(),
            explanation=explanation
        )

        if isinstance(logits, torch.Tensor):
            mask_tensor = torch.from_numpy(mask).to(logits.device, dtype=logits.dtype)
            # Apply logit masking: where mask == 0, add -1e9
            penalty = torch.where(mask_tensor > 0.5, torch.zeros_like(mask_tensor), torch.full_like(mask_tensor, -1e9))
            masked_logits = logits + penalty
            return masked_logits, result
        else:
            penalty = np.where(mask > 0.5, 0.0, -1e9).astype(logits.dtype)
            masked_logits = logits + penalty
            return masked_logits, result
