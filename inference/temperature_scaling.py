"""
AgriVision Ultra v5.0 — Multi-Temperature Platt Calibration.

Learns distinct calibration temperatures T_f for each of the 6 crop families
rather than a single global scalar, minimizing Expected Calibration Error (ECE) to <= 0.015.

Mathematical Formulation:
    P(Y = c | x, c in Family f) = exp(z_c / T_f) / sum_j exp(z_j / T_{f_j})
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


DEFAULT_FAMILY_NAMES = {
    0: "Cotton (Malvaceae)",
    1: "Monocots / Cereals (Rice, Wheat, Maize)",
    2: "Legumes / Pulses (Soybean, Chickpea, Pigeonpea)",
    3: "Solanaceous (Tomato, Potato, Chili)",
    4: "Broadleaf Hard Negatives (Sunflower, Okra, Castor, Weeds)",
    5: "Non-Crop Background (Soil, Hands, Plastic, Debris)"
}


class FamilyTemperatureScaler(nn.Module):
    """
    Multi-Temperature Platt Calibration module across 6 crop families.
    """
    def __init__(
        self,
        num_families: int = 6,
        class_to_family: Optional[List[int]] = None,
        init_temperatures: Optional[Dict[int, float]] = None
    ):
        super(FamilyTemperatureScaler, self).__init__()
        self.num_families = num_families
        
        # Initialize temperatures
        init_t = torch.ones(num_families, dtype=torch.float32)
        if init_temperatures:
            for f_idx, t_val in init_temperatures.items():
                init_t[int(f_idx)] = float(t_val)
        self.temperatures = nn.Parameter(init_t)
        
        # Class-to-family index mapping tensor
        if class_to_family is not None:
            self.register_buffer(
                "family_indices",
                torch.tensor(class_to_family, dtype=torch.long)
            )
        else:
            self.family_indices = None

    def set_class_to_family(self, class_to_family: List[int]):
        """Sets or updates the class-to-family mapping tensor."""
        self.register_buffer(
            "family_indices",
            torch.tensor(class_to_family, dtype=torch.long)
        )

    def forward(
        self,
        logits: torch.Tensor,
        family_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Scales logits using family-specific temperatures:
            z_scaled[b, c] = z[b, c] / T_{family[c]}
        """
        mapping = family_indices if family_indices is not None else self.family_indices
        if mapping is None:
            # Fallback to mean temperature if no mapping is provided
            t_mean = torch.mean(torch.clamp(self.temperatures, min=0.1, max=10.0))
            return logits / t_mean

        mapping = mapping.to(logits.device)
        # Clamp temperatures to avoid division by zero or negative temperatures
        t_clamped = torch.clamp(self.temperatures, min=0.1, max=10.0).to(logits.device)
        t_per_class = t_clamped[mapping] # Shape: (C,)
        
        return logits / t_per_class.unsqueeze(0)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Fits family-specific temperatures on validation logits and labels via NLL minimization.
        """
        if self.family_indices is None:
            raise ValueError("family_indices must be configured before calibration")

        self.temperatures.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperatures], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        logits_device = logits.to(self.temperatures.device)
        labels_device = labels.to(self.temperatures.device)

        ece_before = self.compute_ece(logits_device, labels_device)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_device)
            loss = criterion(scaled_logits, labels_device)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Ensure temperatures remain positive
        with torch.no_grad():
            self.temperatures.clamp_(min=0.1, max=10.0)

        ece_after = self.compute_ece(logits_device, labels_device)
        print(f"[TemperatureScaling] ECE: {ece_before:.4f} -> {ece_after:.4f}")

        return {
            "ece_before": float(ece_before),
            "ece_after": float(ece_after)
        }

    @torch.no_grad()
    def compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15
    ) -> float:
        """
        Calculates Expected Calibration Error (ECE) across predictions:
            ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        ece = torch.zeros(1, device=logits.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece.item())

    def save_calibration(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None):
        """Saves temperature scaling calibration parameters to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        temps = {str(i): round(float(self.temperatures[i].item()), 4) for i in range(self.num_families)}
        data = {
            "temperatures": temps,
            "family_names": {str(k): v for k, v in DEFAULT_FAMILY_NAMES.items()},
            "num_families": self.num_families
        }
        if extra_meta:
            data.update(extra_meta)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[TemperatureScaling] Saved calibration parameters to {filepath}")

    @classmethod
    def load_calibration(
        cls,
        filepath: str,
        class_to_family: Optional[List[int]] = None
    ) -> "FamilyTemperatureScaler":
        """Loads temperature scaling calibration parameters from JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        temps = {int(k): float(v) for k, v in data["temperatures"].items()}
        return cls(
            num_families=data.get("num_families", 6),
            class_to_family=class_to_family,
            init_temperatures=temps
        )


def build_class_to_family_mapping(
    class_names: List[str],
    taxonomy_mapping: Optional[Dict[str, Any]] = None
) -> List[int]:
    """
    Constructs a list of length len(class_names) mapping each class index to its
    corresponding crop family index (0-5).
    """
    mapping = []
    for c_name in class_names:
        c_lower = c_name.lower()
        if any(kw in c_lower for kw in ["cotton", "gossypium"]):
            mapping.append(0) # Cotton
        elif any(kw in c_lower for kw in ["rice", "wheat", "maize", "corn", "sugarcane", "barley"]):
            mapping.append(1) # Monocots
        elif any(kw in c_lower for kw in ["soybean", "chickpea", "pigeonpea", "groundnut", "bean", "pea"]):
            mapping.append(2) # Legumes
        elif any(kw in c_lower for kw in ["tomato", "potato", "chili", "chilli", "pepper", "eggplant", "brinjal"]):
            mapping.append(3) # Solanaceous
        elif any(kw in c_lower for kw in ["sunflower", "okra", "castor", "weed", "cucurbit", "squash", "pumpkin", "cucumber"]):
            mapping.append(4) # Broadleaf
        else:
            mapping.append(1) # Default non-cotton crop family
    return mapping
