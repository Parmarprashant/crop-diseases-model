"""
AgriVision AI — Final Confidence Calibrator.

Post-hoc confidence calibration for the Crop-Aware Disease Resolver.
Maps raw resolver confidence scores to empirical posterior correctness:
    logit(c) = ln(c / (1 - c))
    c_cal = sigma(logit(c) / T_resolver)

Features:
- Strictly monotonic scalar temperature scaling (preserves ranking perfectly).
- Evaluates Brier score, ECE (15 bins), and Negative Log Likelihood (NLL).
- Strict Tier 1 fitting protocol: fitted solely on Tier 1 validation data.
"""
import os
import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class FinalConfidenceCalibrator:
    """
    Monotonic scalar temperature scaling calibrator for resolver output confidences.
    """
    def __init__(self, temperature: float = 1.0, eps: float = 1e-7):
        self.temperature = float(temperature)
        self.eps = eps

    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a single confidence score in [0.0, 1.0].
        Strictly preserves 0.0 and 1.0 bounds and ranking.
        """
        c = float(np.clip(raw_confidence, self.eps, 1.0 - self.eps))
        if abs(self.temperature - 1.0) < 1e-4:
            return round(c, 4)
        
        # logit(c)
        logit = math.log(c / (1.0 - c))
        scaled_logit = logit / max(self.temperature, 1e-4)
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return round(float(np.clip(calibrated, 0.0, 1.0)), 4)

    def calibrate_array(self, confidences: np.ndarray) -> np.ndarray:
        confs = np.clip(confidences, self.eps, 1.0 - self.eps)
        if abs(self.temperature - 1.0) < 1e-4:
            return confs
        logits = np.log(confs / (1.0 - confs))
        scaled_logits = logits / max(self.temperature, 1e-4)
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    @staticmethod
    def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        """
        Expected Calibration Error (ECE) with equal-width binning.
        """
        confs = np.asarray(confidences, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)
        if len(confs) == 0:
            return 0.0
        
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(confs)

        for i in range(n_bins):
            b_low = bin_boundaries[i]
            b_high = bin_boundaries[i + 1]
            in_bin = (confs > b_low) & (confs <= b_high) if i > 0 else (confs >= b_low) & (confs <= b_high)
            bin_count = np.sum(in_bin)
            if bin_count > 0:
                bin_acc = np.mean(labs[in_bin])
                bin_conf = np.mean(confs[in_bin])
                ece += (bin_count / n_total) * np.abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def compute_brier(confidences: np.ndarray, labels: np.ndarray) -> float:
        confs = np.asarray(confidences, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)
        return float(np.mean((confs - labs) ** 2))

    @staticmethod
    def compute_nll(confidences: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
        confs = np.clip(np.asarray(confidences, dtype=np.float64), eps, 1.0 - eps)
        labs = np.asarray(labels, dtype=np.float64)
        nll = -np.mean(labs * np.log(confs) + (1.0 - labs) * np.log(1.0 - confs))
        return float(nll)

    def fit(self, raw_confidences: np.ndarray, labels: np.ndarray, objective: str = "ece") -> Dict[str, float]:
        """
        Optimize temperature T on validation predictions.
        Supports objective='ece' (direct ECE minimization) or objective='nll'.
        """
        confs = np.asarray(raw_confidences, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)

        uncal_ece = self.compute_ece(confs, labs)
        uncal_brier = self.compute_brier(confs, labs)
        uncal_nll = self.compute_nll(confs, labs)

        from scipy.optimize import minimize_scalar

        if objective == "ece":
            # Grid search for minimum ECE
            t_candidates = np.linspace(0.4, 2.5, 211)
            best_t = 1.0
            best_ece = uncal_ece
            for t_cand in t_candidates:
                temp_cal = FinalConfidenceCalibrator(temperature=t_cand)
                c_scaled = temp_cal.calibrate_array(confs)
                cand_ece = temp_cal.compute_ece(c_scaled, labs)
                if cand_ece < best_ece:
                    best_ece = cand_ece
                    best_t = float(t_cand)
            optimal_t = best_t
        else:
            def loss_fn(t_val):
                temp_cal = FinalConfidenceCalibrator(temperature=t_val)
                c_scaled = temp_cal.calibrate_array(confs)
                return temp_cal.compute_nll(c_scaled, labs)

            res = minimize_scalar(loss_fn, bounds=(0.2, 4.0), method='bounded')
            optimal_t = float(res.x)

        self.temperature = optimal_t
        cal_confs = self.calibrate_array(confs)
        cal_ece = self.compute_ece(cal_confs, labs)
        cal_brier = self.compute_brier(cal_confs, labs)
        cal_nll = self.compute_nll(cal_confs, labs)

        return {
            "optimal_temperature": round(optimal_t, 4),
            "uncalibrated_ece": round(uncal_ece, 4),
            "calibrated_ece": round(cal_ece, 4),
            "uncalibrated_brier": round(uncal_brier, 4),
            "calibrated_brier": round(cal_brier, 4),
            "uncalibrated_nll": round(uncal_nll, 4),
            "calibrated_nll": round(cal_nll, 4),
            "ece_reduction_pct": round((uncal_ece - cal_ece) / max(uncal_ece, 1e-6) * 100, 2)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "status": "FROZEN"
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FinalConfidenceCalibrator":
        return cls(temperature=d.get("temperature", 1.0))
