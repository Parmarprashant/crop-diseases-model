import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class OODResult:
    status: str              # "KNOWN_IN_DISTRIBUTION", "UNCERTAIN", "OUT_OF_DISTRIBUTION"
    is_ood: bool
    energy_score: float
    normalized_entropy: float
    max_softmax_probability: float
    centroid_distance: Optional[float]
    calibration_version: str
    rejection_reasons: List[str]


class OODDetector:
    """
    Multi-Metric Out-of-Distribution (OOD) Detector for Closed-Set CNNs.
    
    Combines:
    1. Energy Score (Liu et al., NeurIPS 2020: 'Energy-based Out-of-distribution Detection')
    2. Normalized Shannon Entropy (measures distributional dispersion)
    3. Maximum Softmax Probability (MSP)
    4. Feature-space distance to known class centroids
    
    All thresholds are empirical and loaded from weights/calibration_thresholds.json.
    """
    def __init__(
        self,
        thresholds_path: str = "weights/calibration_thresholds.json",
        temperature: float = 1.0
    ):
        self.temperature = temperature
        self.thresholds_path = thresholds_path
        
        # Safe default thresholds (refined by calibration script)
        self.max_energy_threshold: float = -12.0     # In-dist energy is < -14; OOD is > -10
        self.max_entropy_threshold: float = 0.55     # High entropy indicates uniform confusion
        self.min_msp_threshold: float = 0.60         # Minimum acceptable top probability
        self.max_centroid_dist_threshold: float = 35.0
        self.calibration_version: str = "default_safe_bounds_v1"
        self.class_centroids: Optional[Dict[str, List[float]]] = None

        self._load_thresholds()

    def _load_thresholds(self):
        if os.path.exists(self.thresholds_path):
            try:
                with open(self.thresholds_path, "r") as f:
                    data = json.load(f)
                    self.max_energy_threshold = data.get("energy_threshold_fpr95", self.max_energy_threshold)
                    self.max_entropy_threshold = data.get("entropy_threshold_fpr95", self.max_entropy_threshold)
                    self.min_msp_threshold = data.get("min_msp_threshold", self.min_msp_threshold)
                    self.max_centroid_dist_threshold = data.get("centroid_distance_threshold", self.max_centroid_dist_threshold)
                    self.calibration_version = data.get("version", "calibrated_v1")
                    self.class_centroids = data.get("class_centroids", None)
            except Exception as e:
                print(f"[OODDetector] Notice: using default thresholds ({e})")

    def compute_energy(self, logits: np.ndarray) -> float:
        """
        Energy score E(x; T) = -T * log(sum(exp(z_i / T))).
        More negative = higher likelihood under training distribution.
        """
        z = logits / self.temperature
        # Numerically stable LogSumExp
        max_z = np.max(z)
        lse = max_z + np.log(np.sum(np.exp(z - max_z)))
        energy = -self.temperature * lse
        return float(energy)

    def compute_normalized_entropy(self, probs: np.ndarray) -> float:
        """
        Shannon entropy normalized by log(K) to strictly lie in [0, 1].
        1.0 = completely uniform; 0.0 = completely concentrated on one class.
        """
        k = len(probs)
        if k <= 1:
            return 0.0
        p_safe = np.clip(probs, 1e-12, 1.0)
        h = -np.sum(p_safe * np.log(p_safe))
        max_h = np.log(k)
        return float(h / max_h)

    def evaluate(
        self,
        logits: np.ndarray,
        probabilities: np.ndarray,
        feature_embedding: Optional[np.ndarray] = None
    ) -> OODResult:
        """
        Evaluate an inference output for out-of-distribution properties.
        """
        energy = self.compute_energy(logits)
        entropy = self.compute_normalized_entropy(probabilities)
        msp = float(np.max(probabilities))

        rejection_reasons = []

        # 1. Energy test: OOD inputs have significantly higher (less negative) energy
        is_energy_ood = energy > self.max_energy_threshold
        if is_energy_ood:
            rejection_reasons.append(
                f"Energy score ({energy:.2f}) exceeds OOD threshold ({self.max_energy_threshold:.2f})"
            )

        # 2. Entropy test: Confused / flat distributions indicate uncertainty
        is_entropy_ood = entropy > self.max_entropy_threshold
        if is_entropy_ood:
            rejection_reasons.append(
                f"Normalized entropy ({entropy:.3f}) exceeds threshold ({self.max_entropy_threshold:.3f})"
            )

        # 3. Softmax probability test
        is_msp_low = msp < self.min_msp_threshold
        if is_msp_low:
            rejection_reasons.append(
                f"Peak confidence ({msp*100:.1f}%) is below minimum acceptance ({self.min_msp_threshold*100:.0f}%)"
            )

        # 4. Centroid distance test (if available)
        centroid_dist = None
        if feature_embedding is not None and self.class_centroids:
            # calculate minimum Euclidean distance to any class centroid
            dists = []
            for c_name, c_vec in self.class_centroids.items():
                c_np = np.array(c_vec)
                d = np.linalg.norm(feature_embedding - c_np)
                dists.append(d)
            if dists:
                centroid_dist = float(min(dists))
                if centroid_dist > self.max_centroid_dist_threshold:
                    rejection_reasons.append(
                        f"Feature space distance ({centroid_dist:.1f}) exceeds known centroid threshold ({self.max_centroid_dist_threshold:.1f})"
                    )

        # Formulate final status
        if is_energy_ood or (feature_embedding is not None and centroid_dist and centroid_dist > self.max_centroid_dist_threshold):
            status = "OUT_OF_DISTRIBUTION"
            is_ood = True
        elif is_entropy_ood or is_msp_low:
            status = "UNCERTAIN"
            is_ood = True
        else:
            status = "KNOWN_IN_DISTRIBUTION"
            is_ood = False

        return OODResult(
            status=status,
            is_ood=is_ood,
            energy_score=round(energy, 2),
            normalized_entropy=round(entropy, 3),
            max_softmax_probability=round(msp, 3),
            centroid_distance=round(centroid_dist, 2) if centroid_dist else None,
            calibration_version=self.calibration_version,
            rejection_reasons=rejection_reasons
        )
