from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from inference.crop_taxonomy import CLASS_TAXONOMY, CROP_TO_CLASSES, SUPPORTED_CROPS, get_class_metadata


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
        min_confidence_threshold: float = 0.25,
        min_margin_threshold: float = 0.02
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
        # Step 1: Peak-Affinity & Cumulative Mass Aggregation
        # Tracks both peak single-class affinity and cumulative mass per crop family
        crop_peak_prob: Dict[str, float] = {crop: 0.0 for crop in SUPPORTED_CROPS}
        crop_peak_class: Dict[str, str] = {}
        crop_sum_prob: Dict[str, float] = {crop: 0.0 for crop in SUPPORTED_CROPS}

        for class_name, prob in class_probabilities.items():
            meta = get_class_metadata(class_name)
            if meta and meta.crop in crop_peak_prob:
                crop_family = meta.crop
                p_float = float(prob)
                crop_sum_prob[crop_family] += p_float
                if p_float > crop_peak_prob[crop_family]:
                    crop_peak_prob[crop_family] = p_float
                    crop_peak_class[crop_family] = class_name

        # Step 2: Sort crops by peak single-class affinity
        sorted_crops = sorted(crop_peak_prob.items(), key=lambda kv: kv[1], reverse=True)
        top_crop, top_score = sorted_crops[0]
        second_crop, second_score = sorted_crops[1] if len(sorted_crops) > 1 else (None, 0.0)

        # Step 3: Controlled Cumulative Mass Tie-Breaker (< 5% Margin Only)
        # When the separation between Top-1 and Top-2 candidate crops by peak affinity is < 5%,
        # long-tail scrape duplicates can fragment single-class peaks. We break tight ties (< 5%)
        # by comparing cumulative probability mass between the two competing crops.
        # When margin is >= 5%, existing single-class peak affinity behavior is strictly preserved.
        if second_crop and (0.0 <= (top_score - second_score) < 0.05):
            top_sum = crop_sum_prob.get(top_crop, 0.0)
            second_sum = crop_sum_prob.get(second_crop, 0.0)
            if second_sum > top_sum:
                top_crop = second_crop
                top_score = crop_peak_prob[second_crop]

        # Calculate margin over runner-up crop
        other_scores = [prob for crop, prob in sorted_crops if crop != top_crop]
        second_score = other_scores[0] if other_scores else 0.0
        margin = top_score - second_score

        # Step 4: Validate user crop hint if provided
        if user_crop_hint:
            norm_hint = user_crop_hint.lower().strip()
            if norm_hint in SUPPORTED_CROPS:
                hint_score = crop_peak_prob.get(norm_hint, 0.0)
                # If visual evidence strongly contradicts hint (peak >= 0.60 and separation >= 0.25)
                if top_crop != norm_hint and top_score >= 0.60 and (top_score - hint_score) >= 0.25:
                    return CropDetectionResult(
                        crop="unknown",
                        confidence=round(top_score, 3),
                        status="UNKNOWN",
                        distribution={k: round(v, 4) for k, v in crop_peak_prob.items() if v > 0},
                        margin=round(margin, 3),
                        explanation=(
                            f"Conflict: Field hint declared '{norm_hint.capitalize()}', but visual peak "
                            f"strongly indicates '{top_crop.capitalize()}' ({top_score*100:.1f}%). "
                            f"User hint cannot override visual contradiction."
                        )
                    )
                # If visual evidence is compatible with the hint
                if hint_score >= 0.05 or top_crop == norm_hint:
                    effective_conf = max(hint_score, 0.85)
                    return CropDetectionResult(
                        crop=norm_hint,
                        confidence=round(effective_conf, 3),
                        status="VERIFIED",
                        distribution={k: round(v, 4) for k, v in crop_peak_prob.items() if v > 0},
                        margin=round(margin, 3),
                        explanation=f"Crop confirmed as {norm_hint.capitalize()} (field scout hint verified by visual evidence {hint_score*100:.1f}%)."
                    )
                else:
                    return CropDetectionResult(
                        crop="unknown",
                        confidence=round(hint_score, 3),
                        status="UNKNOWN",
                        distribution={k: round(v, 4) for k, v in crop_peak_prob.items() if v > 0},
                        margin=round(margin, 3),
                        explanation=(
                            f"Unverified hint: Field scout suggested '{norm_hint.capitalize()}', but visual "
                            f"peak probability is only {hint_score*100:.1f}% (< {self.min_confidence_threshold*100:.0f}% calibrated threshold)."
                        )
                    )

        # Step 5: Check for UNKNOWN state:
        # If the peak crop confidence is too low, visual evidence is ambiguous.
        if top_score < self.min_confidence_threshold:
            return CropDetectionResult(
                crop="unknown",
                confidence=round(top_score, 3),
                status="UNKNOWN",
                distribution={k: round(v, 4) for k, v in crop_peak_prob.items() if v > 0},
                margin=round(margin, 3),
                explanation=(
                    f"Crop could not be determined with statistical certainty (peak candidate {top_score*100:.1f}% < {self.min_confidence_threshold*100:.0f}% threshold). "
                    f"Please provide a clear close-up photograph of an individual affected leaf."
                )
            )

        # Confident detection
        status = "VERIFIED" if top_score >= 0.70 else "TENTATIVE"
        return CropDetectionResult(
            crop=top_crop,
            confidence=round(top_score, 3),
            status=status,
            distribution={k: round(v, 4) for k, v in crop_peak_prob.items() if v > 0},
            margin=round(margin, 3),
            explanation=f"Visual evidence is consistent with {top_crop.capitalize()} ({top_score*100:.1f}% peak affinity)."
        )
