"""
AgriVision Ultra v5.0 — Evidential Uncertainty Estimation.

Parameterizes the model output as a Dirichlet distribution using Subjective Logic /
Evidential Deep Learning (Sensoy et al., NeurIPS 2018):
    Evidence: e_k = max(0, z_k)
    Dirichlet strength: S = sum_{k=1}^K (e_k + 1)
    Belief mass: b_k = e_k / S
    Epistemic uncertainty (vacuity): u = K / S

Note:
    sum_{k=1}^K b_k + u = 1.0

If epistemic uncertainty u > 0.30, the input is classified as UNCERTAIN_ANOMALY
and routed to the university scientist queue.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import torch


@dataclass
class EvidentialResult:
    predicted_class: str
    predicted_idx: int
    belief_masses: Dict[str, float]
    epistemic_uncertainty: float
    total_evidence: float
    is_anomaly: bool
    action_route: str
    explanation: str


class EvidentialUncertaintyEngine:
    """
    Computes Dirichlet belief masses and epistemic uncertainty from logits.
    """
    def __init__(
        self,
        uncertainty_threshold: float = 0.30,
        class_names: Optional[List[str]] = None
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.class_names = class_names or []

    def estimate_uncertainty(
        self,
        logits: Union[np.ndarray, torch.Tensor, List[float]],
        class_names: Optional[List[str]] = None,
        uncertainty_threshold: Optional[float] = None
    ) -> EvidentialResult:
        """
        Computes evidential belief masses and epistemic uncertainty u = K / S.
        """
        names = class_names or self.class_names
        tau_u = uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_threshold

        if isinstance(logits, torch.Tensor):
            z = logits.detach().cpu().squeeze().float().numpy()
        else:
            z = np.array(logits, dtype=np.float32).squeeze()

        K = len(z)
        if K == 0:
            return EvidentialResult(
                predicted_class="Unknown",
                predicted_idx=0,
                belief_masses={},
                epistemic_uncertainty=1.0,
                total_evidence=0.0,
                is_anomaly=True,
                action_route="UNIVERSITY_SCIENTIST_QUEUE",
                explanation="Zero logits provided: maximum epistemic uncertainty."
            )

        # Evidence: e_k = max(0, z_k)
        evidence = np.maximum(0.0, z)
        # Dirichlet strength: S = sum(e_k + 1) = sum(e_k) + K
        S = float(np.sum(evidence) + K)

        # Belief masses: b_k = e_k / S
        belief_masses = evidence / S
        # Epistemic uncertainty: u = K / S
        u = float(K / S)

        best_idx = int(np.argmax(belief_masses))
        best_name = names[best_idx] if best_idx < len(names) else f"Class_{best_idx}"

        # Top belief masses for clean reporting
        top_indices = np.argsort(belief_masses)[::-1][:5]
        top_beliefs = {
            (names[i] if i < len(names) else f"Class_{i}"): round(float(belief_masses[i]), 4)
            for i in top_indices
        }

        is_anomaly = u > tau_u
        if is_anomaly:
            action_route = "UNIVERSITY_SCIENTIST_QUEUE"
            explanation = (
                f"High epistemic uncertainty (u = {u:.4f} > {tau_u:.2f}): "
                f"Visual features exhibit high vacuity across known disease Dirichlet priors. "
                f"Direct chemical intervention locked; routed to University Scientist Queue."
            )
        else:
            action_route = "ROUTED_EXPERT"
            explanation = (
                f"Sufficient evidence observed (u = {u:.4f} <= {tau_u:.2f}): "
                f"Belief mass b('{best_name}') = {belief_masses[best_idx]:.4f}. "
                f"Proceeding with autonomous expert diagnosis."
            )

        return EvidentialResult(
            predicted_class=best_name,
            predicted_idx=best_idx,
            belief_masses=top_beliefs,
            epistemic_uncertainty=round(u, 4),
            total_evidence=round(S, 4),
            is_anomaly=is_anomaly,
            action_route=action_route,
            explanation=explanation
        )
