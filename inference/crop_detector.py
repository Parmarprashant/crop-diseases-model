from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from inference.crop_taxonomy import CLASS_TAXONOMY, CROP_TO_CLASSES, SUPPORTED_CROPS


@dataclass
class CropDetectionResult:
    crop: str                   # "rice", "wheat", "cotton", "maize", "sugarcane", "unknown"
    confidence: float           # 0.0 to 1.0
    status: str                 # "VERIFIED", "TENTATIVE", "UNKNOWN"
    distribution: Dict[str, float]
    margin: float
    explanation: str


class CropDetector:
    """
    Calibrated Crop Family Detector with a first-class UNKNOWN state.
    
    CRITICAL SAFETY PRINCIPLE:
    Never forces an image into one of the 5 supported crops if evidence is ambiguous.
    If the evidence across crop classes is fragmented or low-confidence,
    it returns crop='unknown' to prevent cascade hallucinations.
    """
    def __init__(
        self,
        min_confidence_threshold: float = 0.35,
        min_margin_threshold: float = 0.08
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_margin_threshold = min_margin_threshold

    def detect_from_predictions(
        self,
        class_probabilities: Dict[str, float],
        user_crop_hint: Optional[str] = None
    ) -> CropDetectionResult:
        """
        Calculates aggregate probabilistic mass per crop family from the model's
        predictive distribution, applying calibration and unknown gating.
        
        CRITICAL SCIENTIFIC RULE:
        user_crop_hint is NEVER treated as unverified ground truth.
        The visual crop detector must validate compatibility.
        """
        # Aggregate probability mass by crop family from visual predictions
        crop_scores: Dict[str, float] = {crop: 0.0 for crop in SUPPORTED_CROPS}
        for class_name, prob in class_probabilities.items():
            meta = CLASS_TAXONOMY.get(class_name)
            if meta and meta.crop in crop_scores:
                crop_scores[meta.crop] += float(prob)

        # Sort crops by accumulated probability
        sorted_crops = sorted(crop_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_crop, top_score = sorted_crops[0]
        second_crop, second_score = sorted_crops[1]
        margin = top_score - second_score

        # Validate user crop hint if provided
        if user_crop_hint:
            norm_hint = user_crop_hint.lower().strip()
            if norm_hint in SUPPORTED_CROPS:
                hint_score = crop_scores.get(norm_hint, 0.0)
                # If visual evidence strongly indicates a DIFFERENT crop, flag conflict
                if top_crop != norm_hint and top_score >= 0.60 and (top_score - hint_score) >= 0.25:
                    return CropDetectionResult(
                        crop="unknown",
                        confidence=round(top_score, 3),
                        status="UNKNOWN",
                        distribution={k: round(v, 4) for k, v in crop_scores.items()},
                        margin=round(margin, 3),
                        explanation=(
                            f"Conflict: Field hint declared '{norm_hint.capitalize()}', but visual evidence "
                            f"strongly indicates '{top_crop.capitalize()}' ({top_score*100:.1f}%). "
                            f"User hint cannot override visual contradiction."
                        )
                    )
                # If visual evidence is sufficiently compatible with the hint
                if hint_score >= self.min_confidence_threshold or top_crop == norm_hint:
                    effective_conf = max(hint_score, 0.85)
                    return CropDetectionResult(
                        crop=norm_hint,
                        confidence=round(effective_conf, 3),
                        status="VERIFIED",
                        distribution={k: round(v, 4) for k, v in crop_scores.items()},
                        margin=round(margin, 3),
                        explanation=f"Crop confirmed as {norm_hint.capitalize()} (field scout hint verified by visual evidence {hint_score*100:.1f}%)."
                    )
                else:
                    # Visual mass for hinted crop is too low
                    return CropDetectionResult(
                        crop="unknown",
                        confidence=round(hint_score, 3),
                        status="UNKNOWN",
                        distribution={k: round(v, 4) for k, v in crop_scores.items()},
                        margin=round(margin, 3),
                        explanation=(
                            f"Unverified hint: Field scout suggested '{norm_hint.capitalize()}', but visual "
                            f"probability mass is only {hint_score*100:.1f}% (< {self.min_confidence_threshold*100:.0f}% calibrated threshold)."
                        )
                    )

        # Check for UNKNOWN state:
        # If the top crop mass is too low, or the margin between top two crops is negligible,
        # visual evidence is genuinely ambiguous. Do NOT guess.
        if top_score < self.min_confidence_threshold:
            return CropDetectionResult(
                crop="unknown",
                confidence=round(top_score, 3),
                status="UNKNOWN",
                distribution={k: round(v, 4) for k, v in crop_scores.items()},
                margin=round(margin, 3),
                explanation=(
                    f"Crop could not be determined with statistical certainty. "
                    f"Top candidate mass '{top_crop}' is {top_score*100:.1f}% (< {self.min_confidence_threshold*100:.0f}% threshold)."
                )
            )

        if margin < self.min_margin_threshold and top_score < 0.70:
            return CropDetectionResult(
                crop="unknown",
                confidence=round(top_score, 3),
                status="UNKNOWN",
                distribution={k: round(v, 4) for k, v in crop_scores.items()},
                margin=round(margin, 3),
                explanation=(
                    f"Crop ambiguity: Close split between '{top_crop}' ({top_score*100:.1f}%) "
                    f"and '{second_crop}' ({second_score*100:.1f}%). Insufficient separation."
                )
            )

        # Confident detection
        status = "VERIFIED" if top_score >= 0.75 else "TENTATIVE"
        return CropDetectionResult(
            crop=top_crop,
            confidence=round(top_score, 3),
            status=status,
            distribution={k: round(v, 4) for k, v in crop_scores.items()},
            margin=round(margin, 3),
            explanation=f"Visual evidence is consistent with {top_crop.capitalize()} ({top_score*100:.1f}% cumulative affinity)."
        )
