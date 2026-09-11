import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class PlantPartDetectionResult:
    plant_part: str            # "leaf", "panicle", "stem", "boll", "root", "unknown"
    confidence: float          # 0.0 to 1.0
    status: str                # "DETECTED", "UNCERTAIN", "UNKNOWN"
    part_scores: Dict[str, float]
    features_extracted: Dict[str, float]
    explanation: str


class PlantPartDetector:
    """
    Calibrated Plant-Part Detector with a first-class UNKNOWN state.
    
    CRITICAL SAFETY PRINCIPLE:
    Does NOT assert high-confidence plant-part classifications without strong evidence.
    If the image is ambiguous, blurry, or lacks distinct botanical signatures,
    it returns plant_part='unknown' so the pipeline refuses to run unsupported leaf classifiers.
    """
    def __init__(
        self,
        min_confidence_threshold: float = 0.38,
        min_margin_threshold: float = 0.08
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_margin_threshold = min_margin_threshold

    def detect(self, image: Image.Image) -> PlantPartDetectionResult:
        np_img = np.array(image.convert("RGB"))
        h, w, _ = np_img.shape
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # 1. Edge and texture distribution
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(w * h)

        # 2. Color profile: Green vs Yellow/Gold/Brown
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Leaf green: Hue roughly 35 - 85
        green_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel > 30)
        green_ratio = float(np.count_nonzero(green_mask)) / float(w * h)

        # Panicle / Spike golden-yellow/brown: Hue 15 - 35
        panicle_mask = (h_channel >= 15) & (h_channel < 35) & (s_channel > 40)
        panicle_color_ratio = float(np.count_nonzero(panicle_mask)) / float(w * h)

        # Cotton boll white/fibrous: Low saturation, high brightness
        boll_mask = (s_channel < 40) & (v_channel > 180)
        boll_ratio = float(np.count_nonzero(boll_mask)) / float(w * h)

        # 3. Textural granularity (Local contrast variance)
        # Panicles and spikes have dense granular bead textures
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        avg_texture_granularity = float(np.mean(np.sqrt(np.maximum(local_variance, 0))))

        # Evidence accumulation
        # Leaf evidence: Requires actual green vegetation coverage
        leaf_score = green_ratio * 0.85 + (green_ratio > 0.3) * (1.0 - min(edge_density * 4, 1.0)) * 0.15
        
        # Panicle evidence: Golden/tan/orange hue, high granular texture, distributed bead edges
        panicle_score = panicle_color_ratio * 0.70 + (panicle_color_ratio > 0.1) * (min(edge_density * 5, 1.0) * 0.15 + min(avg_texture_granularity / 40.0, 1.0) * 0.15)
        
        # Boll evidence: White fibrous mass + compact rounded regions
        boll_score = boll_ratio * 0.85

        # Stem evidence: Baseline prior
        stem_score = 0.05

        scores = {
            "leaf": leaf_score,
            "panicle": panicle_score,
            "boll": boll_score,
            "stem": stem_score
        }

        # Normalize to probability-like distribution
        total = sum(scores.values()) + 1e-8
        norm_scores = {k: v / total for k, v in scores.items()}

        sorted_parts = sorted(norm_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_part, top_conf = sorted_parts[0]
        second_part, second_conf = sorted_parts[1]
        margin = top_conf - second_conf

        features_dict = {
            "edge_density": round(edge_density, 4),
            "green_ratio": round(green_ratio, 4),
            "panicle_color_ratio": round(panicle_color_ratio, 4),
            "boll_ratio": round(boll_ratio, 4),
            "texture_granularity": round(avg_texture_granularity, 2)
        }

        # Calibration Gate: If confidence or margin is insufficient, assign UNKNOWN
        if top_conf < self.min_confidence_threshold or margin < self.min_margin_threshold:
            return PlantPartDetectionResult(
                plant_part="unknown",
                confidence=round(top_conf, 3),
                status="UNKNOWN",
                part_scores={k: round(v, 4) for k, v in norm_scores.items()},
                features_extracted=features_dict,
                explanation="Plant part could not be determined with sufficient confidence. Defaulting to 'unknown' state."
            )

        status = "DETECTED" if top_conf >= 0.65 else "UNCERTAIN"
        return PlantPartDetectionResult(
            plant_part=top_part,
            confidence=round(top_conf, 3),
            status=status,
            part_scores={k: round(v, 4) for k, v in norm_scores.items()},
            features_extracted=features_dict,
            explanation=f"Visual features indicate plant organ is consistent with {top_part} ({top_conf*100:.1f}% confidence)."
        )
