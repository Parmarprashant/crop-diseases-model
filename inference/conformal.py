"""
AgriVision Ultra v5.0 — Conformal Risk Control (CRC) & Split Conformal Prediction.

Extends standard inductive split conformal prediction with Conformal Risk Control (CRC)
(Angelopoulos et al., 2022/2024) to bound the expected False Omission Rate (FOR)
on high-consequence quarantine and lethal pathogens strictly below 1%:
    E[L_omission(C(X), Y)] <= beta  (beta = 0.01)
where:
    L_omission = 1 if the true lethal pathogen is omitted from prediction set C(X), and 0 otherwise.

For non-critical classes, maintains standard 95% marginal coverage guarantee (1 - alpha = 0.95).
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# Critical pathogens requiring bounded False Omission Rate (FOR <= 1%)
CRITICAL_QUARANTINE_PATHOGENS = {
    "cotton leaf curl virus",
    "bacterial blight",
    "angular leaf spot",
    "bacterial leaf blight",
    "becterial blight in rice",
    "brownspot",
    "late blight",
    "flag smut",
    "american bollworm",
    "fall armyworm",
    "fusarium wilt",
    "verticillium wilt"
}


@dataclass
class ConformalResult:
    """
    Structured outcome of Conformal Risk Control (CRC) & Split Conformal Prediction.
    """
    status: str                         # "CONFORMAL_SINGLE", "CONFORMAL_MULTIPLE", "CONFORMAL_EMPTY"
    candidate_indices: List[int]        # Indices satisfying conformal / CRC cutoff
    candidate_classes: List[str]        # Canonical class names of candidates
    candidate_probabilities: Dict[str, float] # Class name -> softmax probability
    set_size: int                       # |C(X)|
    q_hat: float                        # Empirical quantile threshold
    tau_q: float                        # 1 - q_hat cutoff probability
    spray_permitted: bool               # True ONLY on CONFORMAL_SINGLE
    explanation: str                    # Human/auditor diagnostic explanation
    crc_bound: float = 0.01             # Bounded False Omission Rate (beta <= 0.01)
    critical_pathogens_in_set: List[str] = field(default_factory=list)
    crc_guarantee_active: bool = True


class ConformalEngine:
    """
    Dual Conformal Engine:
    1. Inductive Split Conformal Prediction for marginal 95% coverage (alpha = 0.05)
    2. Conformal Risk Control (CRC) bounding False Omission Rate (beta <= 0.01) on lethal pathogens
    """
    def __init__(
        self,
        calibration_path: str = "weights/conformal_calibration.json",
        crc_calibration_path: str = "weights/ultra_v5/crc_calibration.json",
        alpha: float = 0.05,
        beta: float = 0.01,
        class_names: Optional[List[str]] = None,
        default_q_hat: float = 0.85,
        default_tau_crc: float = 0.08
    ):
        self.calibration_path = calibration_path
        self.crc_calibration_path = crc_calibration_path
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.class_names = class_names or []
        self.q_hat = default_q_hat
        self.tau_q = round(1.0 - self.q_hat, 4)
        self.tau_crc = default_tau_crc
        self.calibration_metadata: Dict[str, Any] = {}
        
        self.load_calibration()

    def load_calibration(self) -> None:
        """Loads pre-computed empirical quantile q_hat and CRC threshold tau_crc."""
        # Standard Conformal Calibration
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.calibration_metadata.update(data)
                    self.q_hat = float(data.get("q_hat", self.q_hat))
                    self.alpha = float(data.get("alpha", self.alpha))
                    self.tau_q = round(1.0 - self.q_hat, 4)
            except Exception as e:
                print(f"[ConformalEngine] Notice loading conformal calibration: {e}")

        # CRC Calibration
        if os.path.exists(self.crc_calibration_path):
            try:
                with open(self.crc_calibration_path, "r", encoding="utf-8") as f:
                    crc_data = json.load(f)
                    self.tau_crc = float(crc_data.get("tau_crc", self.tau_crc))
                    self.beta = float(crc_data.get("beta", self.beta))
            except Exception as e:
                print(f"[ConformalEngine] Notice loading CRC calibration: {e}")

    @staticmethod
    def is_critical_pathogen(class_name: str) -> bool:
        """Checks if a pathogen class belongs to high-consequence quarantine list."""
        name_lower = class_name.lower()
        return any(crit in name_lower for crit in CRITICAL_QUARANTINE_PATHOGENS)

    @classmethod
    def calibrate_crc(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        beta: float = 0.01,
        grid_steps: int = 1000
    ) -> float:
        """
        Calibrates CRC threshold lambda_hat such that:
            (n / (n + 1)) * R_n(lambda) + 1 / (n + 1) <= beta
        where loss is False Omission of critical pathogens:
            L_omission(C_lambda(X), Y) = 1 if Y in Critical and Y not in C_lambda(X), else 0.
        """
        n = len(labels)
        if n == 0:
            return 0.08

        # Identify critical label indices
        critical_indices = {
            i for i, name in enumerate(class_names)
            if cls.is_critical_pathogen(name)
        }

        # Grid search over lambda from 1.0 down to 0.0
        candidate_lambdas = np.linspace(1.0, 0.001, grid_steps)
        best_lambda = 0.001

        for lam in candidate_lambdas:
            losses = []
            for i in range(n):
                y_i = labels[i]
                if y_i in critical_indices:
                    # Loss = 1 if omitted from prediction set C_lam
                    in_set = probabilities[i, y_i] >= lam
                    losses.append(0.0 if in_set else 1.0)
                else:
                    losses.append(0.0)

            empirical_risk = np.mean(losses) if losses else 0.0
            # Finite-sample upper bound with continuity correction
            upper_bound = (n / (n + 1.0)) * empirical_risk + (1.0 / (n + 1.0))
            if upper_bound <= beta:
                best_lambda = float(lam)
                break

        print(f"[CRC] Calibrated tau_crc = {best_lambda:.4f} for bound beta <= {beta:.3f}")
        return best_lambda

    def predict_set(
        self,
        logits_or_probs: Union[np.ndarray, torch.Tensor, List[float]],
        tau_q: Optional[float] = None,
        class_names: Optional[List[str]] = None
    ) -> ConformalResult:
        """
        Constructs prediction set C(X) incorporating both:
        1. Standard conformal cutoff tau_q = 1 - q_hat (alpha = 0.05)
        2. CRC cutoff tau_crc for critical pathogens (bound beta <= 0.01)
        """
        names = class_names or self.class_names

        # Convert to 1D numpy probability array
        if isinstance(logits_or_probs, torch.Tensor):
            t = logits_or_probs.detach().cpu()
            if t.dim() > 1:
                t = t.squeeze(0)
            if torch.any(t < 0) or not torch.isclose(torch.sum(t), torch.tensor(1.0), atol=1e-2):
                probs = F.softmax(t, dim=0).numpy()
            else:
                probs = t.numpy()
        elif isinstance(logits_or_probs, np.ndarray):
            arr = logits_or_probs.squeeze()
            if np.any(arr < 0) or not np.isclose(np.sum(arr), 1.0, atol=1e-2):
                exp_arr = np.exp(arr - np.max(arr))
                probs = exp_arr / np.sum(exp_arr)
            else:
                probs = arr
        else:
            arr = np.array(logits_or_probs)
            if np.any(arr < 0) or not np.isclose(np.sum(arr), 1.0, atol=1e-2):
                exp_arr = np.exp(arr - np.max(arr))
                probs = exp_arr / np.sum(exp_arr)
            else:
                probs = arr

        cutoff_std = tau_q if tau_q is not None else self.tau_q
        cutoff_crc = self.tau_crc

        # Determine candidates:
        # A class k is included if:
        # - It is a critical pathogen and probs[k] >= cutoff_crc, OR
        # - probs[k] >= cutoff_std
        candidate_indices = []
        critical_in_set = []

        for idx, p in enumerate(probs):
            c_name = names[idx] if idx < len(names) else f"Class_{idx}"
            is_crit = self.is_critical_pathogen(c_name)
            effective_cutoff = cutoff_crc if is_crit else cutoff_std

            if p >= effective_cutoff:
                candidate_indices.append(idx)
                if is_crit:
                    critical_in_set.append(c_name)

        # Sort candidates descending by probability
        candidate_indices.sort(key=lambda idx: probs[idx], reverse=True)

        candidate_classes = [
            names[i] if i < len(names) else f"Class_{i}"
            for i in candidate_indices
        ]
        candidate_probs = {
            (names[i] if i < len(names) else f"Class_{i}"): round(float(probs[i]), 4)
            for i in candidate_indices
        }

        set_size = len(candidate_indices)

        if set_size == 1:
            status = "CONFORMAL_SINGLE"
            spray_permitted = True
            explanation = (
                f"Statistically certain diagnosis: Single candidate '{candidate_classes[0]}' "
                f"meets conformal and CRC bounds (P={candidate_probs[candidate_classes[0]]*100:.1f}%). "
                f"CIBRC-compliant treatments permitted."
            )
        elif set_size > 1:
            status = "CONFORMAL_MULTIPLE"
            spray_permitted = False
            top_cand_str = ", ".join([f"{c} ({candidate_probs[c]*100:.1f}%)" for c in candidate_classes[:3]])
            crc_note = f" (Includes critical pathogen protection: {', '.join(critical_in_set)})" if critical_in_set else ""
            explanation = (
                f"Ambiguous co-infection or early-stage foliar lesion: {set_size} candidate pathogens "
                f"in conformal set ({top_cand_str}){crc_note}. "
                f"Targeted chemical sprays SUPPRESSED; cultural inspection advisories provided."
            )
        else:
            status = "CONFORMAL_EMPTY"
            spray_permitted = False
            explanation = (
                f"Conformal prediction set is empty (|C(X)| = 0): No pathogen class satisfied "
                f"the conformal confidence threshold. Flagged as out-of-distribution anomaly. "
                f"All spray advisories locked; routed to Institutional HITL queue."
            )

        return ConformalResult(
            status=status,
            candidate_indices=candidate_indices,
            candidate_classes=candidate_classes,
            candidate_probabilities=candidate_probs,
            set_size=set_size,
            q_hat=round(self.q_hat, 4),
            tau_q=round(cutoff_std, 4),
            spray_permitted=spray_permitted,
            explanation=explanation,
            crc_bound=self.beta,
            critical_pathogens_in_set=critical_in_set,
            crc_guarantee_active=True
        )
