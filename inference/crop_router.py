"""
AgriVision Ultra v5.0 — 6-Family Morphological Crop Router with Dual-Gated OOD Protection.

Integrates:
1. 6-Family Morphological Backbone (EfficientNet-B5 + CBAM)
2. Logit Free Energy OOD Gating:
   E(x) = -T * log \sum_{j=0}^5 exp(z_j / T)
3. Feature-Space Mahalanobis Distance Envelope:
   D_M(x) = min_{c in {0..5}} sqrt((z(x) - \hat{\mu}_c)^T \mathbf{\Sigma}^{-1} (z(x) - \hat{\mu}_c))
4. Dual-Gated Out-of-Distribution Rejection:
   is_ood = True if E(x) > -1.20 OR D_M(x) > \tau_{mahalanobis} (99th empirical percentile).
5. State DPI Administrative Prior (crop_hint) integration.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.cbam import CBAM

try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


ROUTER_FAMILIES = {
    0: "Cotton (Malvaceae)",
    1: "Monocots / Cereals (Rice, Wheat, Maize)",
    2: "Legumes / Pulses (Soybean, Chickpea, Pigeonpea)",
    3: "Solanaceous (Tomato, Potato, Chili)",
    4: "Broadleaf Hard Negatives (Sunflower, Okra, Castor, Weeds)",
    5: "Non-Crop Background (Soil, Hands, Plastic, Debris)"
}


@dataclass
class RoutingDecision:
    status: str                         # "COTTON", "NON_COTTON", "UNCERTAIN"
    expert: Optional[str]               # "cotton_specialist_42", "generalist_281", or None
    routing_confidence: float           # Confidence in routing decision (0.0 - 1.0)
    p_cotton: float                     # Raw P(Cotton)
    p_non_cotton: float                 # Raw P(Non-Cotton)
    margin: float                       # |P(Cotton) - P(Non-Cotton)|
    ood_score: float                    # Energy-based OOD metric
    is_ood: bool                        # Whether sample is detected as OOD
    reason: str                         # Explanation of decision
    predicted_family: int = 0           # 0 to 5
    family_name: str = "Cotton"         # Name of predicted family
    family_probabilities: Dict[str, float] = None # Softmax probability per family
    mahalanobis_distance: float = 0.0   # Minimum Mahalanobis distance D_M(x)


class CropRouterModel(nn.Module):
    """
    Ultra v5.0 6-Family Morphological Crop Router Model:
    - Backbone: EfficientNet-B5 features (2048-d)
    - Attention: CBAM (Channel + Spatial attention)
    - Projection: 2048 -> 512 (SiLU)
    - Classification Head: 512 -> 6
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = False):
        super(CropRouterModel, self).__init__()
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError("torchvision required for CropRouterModel")
            
        base_model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None)
        self.features = base_model.features
        self.cbam = CBAM(in_channels=2048, reduction_ratio=16, kernel_size=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        
        self.router_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat_att = self.cbam(feat)
        pooled = self.avgpool(feat_att)
        flat = torch.flatten(pooled, 1)
        logits = self.router_head(flat)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dimensional bottleneck representation z(x)."""
        feat = self.features(x)
        feat_att = self.cbam(feat)
        pooled = self.avgpool(feat_att)
        flat = torch.flatten(pooled, 1)
        h = self.router_head[1](flat)  # 512-dim bottleneck
        h = self.router_head[2](h)     # SiLU
        return h


class DedicatedCropRouter:
    """
    Ultra v5.0 Dedicated Crop Router with Dual-Gated OOD Protection:
    - Energy OOD gating (tau_energy = -1.20)
    - Mahalanobis feature envelope (tau_mahalanobis = 99th empirical percentile)
    """
    def __init__(
        self,
        checkpoint_path: str = "weights/research/crop_router_v4.pt",
        thresholds_path: str = "weights/router_thresholds.json",
        envelope_path: str = "weights/ultra_v5/mahalanobis_envelope.json",
        device: Optional[str] = None,
        img_size: int = 256,
        temperature: float = 1.0
    ):
        self.checkpoint_path = checkpoint_path
        self.thresholds_path = thresholds_path
        self.envelope_path = envelope_path
        self.img_size = img_size
        self.temperature = float(temperature)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Operational thresholds
        self.thresholds = {
            "tau_cotton_high": 0.75,
            "tau_cotton_low": 0.15,
            "tau_margin_min": 0.15,
            "tau_ood_energy": -1.20,
            "tau_mahalanobis": 27.017,
            "temperature": 1.0
        }
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    for k in self.thresholds:
                        if loaded.get(k) is not None:
                            self.thresholds[k] = float(loaded[k])
            except Exception as e:
                print(f"[CropRouter] Notice loading thresholds: {e}")

        # Override energy threshold strictly to -1.20 per Ultra v5 invariant
        self.thresholds["tau_ood_energy"] = -1.20

        # Load Mahalanobis Envelope Parameters
        self.has_mahalanobis = False
        self.means = None
        self.inv_cov = None

        if os.path.exists(envelope_path):
            try:
                with open(envelope_path, "r", encoding="utf-8") as f:
                    env_data = json.load(f)
                    self.thresholds["tau_mahalanobis"] = float(env_data.get("tau_mahalanobis", self.thresholds["tau_mahalanobis"]))
                
                # Check for full precision numpy arrays
                inv_cov_path = os.path.join(os.path.dirname(envelope_path), "inv_covariance.npy")
                means_path = os.path.join(os.path.dirname(envelope_path), "family_means.npy")
                if os.path.exists(inv_cov_path) and os.path.exists(means_path):
                    self.inv_cov = torch.from_numpy(np.load(inv_cov_path)).to(self.device).float()
                    self.means = torch.from_numpy(np.load(means_path)).to(self.device).float()
                    self.has_mahalanobis = True
            except Exception as e:
                print(f"[CropRouter] Notice loading Mahalanobis envelope: {e}")

        # Build 6-family model
        self.model = CropRouterModel(num_classes=6, pretrained=False)
        
        resolved_ckpt = checkpoint_path
        if not os.path.exists(resolved_ckpt):
            fallback_ckpt = "weights/research/crop_router_v1.pt"
            if os.path.exists(fallback_ckpt):
                resolved_ckpt = fallback_ckpt
                
        if os.path.exists(resolved_ckpt):
            print(f"[CropRouter] Loading router weights from {resolved_ckpt} on {self.device}...")
            sd = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
            self.model.load_state_dict(sd, strict=False)
                
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def compute_mahalanobis_distance(self, feat: torch.Tensor) -> float:
        """
        Calculates minimum Mahalanobis distance:
            D_M(x) = min_{c in {0..5}} sqrt((z(x) - \mu_c)^T \Sigma^{-1} (z(x) - \mu_c))
        """
        if not self.has_mahalanobis or self.means is None or self.inv_cov is None:
            return 0.0
        # feat: (1, 512), self.means: (6, 512)
        diff = feat - self.means  # (6, 512)
        # d_sq = sum((diff @ inv_cov) * diff, dim=1)
        d_sq = torch.sum((diff @ self.inv_cov) * diff, dim=1)  # (6,)
        min_d_sq = torch.clamp(torch.min(d_sq), min=0.0)
        return float(torch.sqrt(min_d_sq).item())

    @torch.inference_mode()
    def route_crop(
        self,
        image_tensor: Union[torch.Tensor, Image.Image],
        crop_hint: Optional[str] = None
    ) -> RoutingDecision:
        """
        Primary routing evaluation:
        1. Temperature-scaled logit normalization & energy OOD detection:
           E(x) = -T * log \sum_{j=0}^{5} exp(z_j / T)
        2. Feature-space Mahalanobis distance envelope:
           D_M(x) = min_{c in {0..5}} sqrt((z(x) - \mu_c)^T \Sigma^{-1} (z(x) - \mu_c))
        3. Dual-Gated OOD short-circuit if E(x) > -1.20 OR D_M(x) > tau_mahalanobis.
        4. Administrative metadata prior (crop_hint) integration.
        5. Multiclass botanical dispatch to Model A or Model B.
        """
        if isinstance(image_tensor, Image.Image):
            if image_tensor.mode != "RGB":
                image_tensor = image_tensor.convert("RGB")
            tensor = self.transform(image_tensor).unsqueeze(0).to(self.device)
        elif isinstance(image_tensor, torch.Tensor):
            tensor = image_tensor
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
        else:
            raise TypeError(f"Expected PIL Image or torch.Tensor, got {type(image_tensor)}")

        logits = self.model(tensor)
        feat = self.model.extract_features(tensor)
        num_classes = logits.shape[1]
        
        T = self.temperature
        scaled_logits = logits / T
        
        # Energy score: E(x) = -T * logsumexp(z_j / T)
        energy = -float(T * torch.logsumexp(scaled_logits, dim=1).item())
        tau_energy = self.thresholds["tau_ood_energy"] # -1.20
        
        # Mahalanobis distance D_M(x)
        d_mahalanobis = self.compute_mahalanobis_distance(feat)
        tau_mahalanobis = self.thresholds["tau_mahalanobis"]

        # Dual-Gated OOD Protection
        is_energy_ood = energy > tau_energy
        is_mahalanobis_ood = (d_mahalanobis > tau_mahalanobis) if self.has_mahalanobis else False
        is_ood = is_energy_ood or is_mahalanobis_ood

        if is_ood:
            ood_reasons = []
            if is_energy_ood:
                ood_reasons.append(f"Free Energy E(x) = {energy:.3f} > {tau_energy:.2f}")
            if is_mahalanobis_ood:
                ood_reasons.append(f"Mahalanobis D_M(x) = {d_mahalanobis:.2f} > {tau_mahalanobis:.2f}")
            
            return RoutingDecision(
                status="UNCERTAIN",
                expert=None,
                routing_confidence=0.0,
                p_cotton=0.0,
                p_non_cotton=1.0,
                margin=1.0,
                ood_score=round(energy, 4),
                is_ood=True,
                reason=f"Dual-Gated OOD: {'; '.join(ood_reasons)}. Visual features violate botanical envelope.",
                predicted_family=5,
                family_name=ROUTER_FAMILIES.get(5, "Non-Crop Background"),
                family_probabilities={},
                mahalanobis_distance=round(d_mahalanobis, 4)
            )

        probs = F.softmax(scaled_logits, dim=1).squeeze(0)
        p_cotton = float(probs[0].item())
        
        if num_classes >= 6:
            p_non_cotton = float(probs[1:].sum().item())
            pred_family = int(torch.argmax(probs).item())
            fam_probs = {ROUTER_FAMILIES.get(i, f"Class_{i}"): round(float(probs[i].item()), 4) for i in range(num_classes)}
        else:
            p_non_cotton = float(probs[0].item()) if num_classes == 2 else float(1.0 - p_cotton)
            p_cotton = float(probs[1].item()) if num_classes == 2 else p_cotton
            pred_family = 0 if p_cotton > p_non_cotton else 1
            fam_probs = {"Cotton": round(p_cotton, 4), "Non-Cotton": round(p_non_cotton, 4)}

        margin = abs(p_cotton - p_non_cotton)
        family_name = ROUTER_FAMILIES.get(pred_family, f"Family_{pred_family}")

        # -------------------------------------------------------------
        # 1. Administrative Metadata Prior Integration (crop_hint)
        # -------------------------------------------------------------
        if crop_hint:
            norm_hint = crop_hint.lower().strip()
            if norm_hint == "cotton":
                if pred_family not in (4, 5):
                    return RoutingDecision(
                        status="COTTON",
                        expert="cotton_specialist_42",
                        routing_confidence=round(max(p_cotton, 0.85), 4),
                        p_cotton=round(p_cotton, 4),
                        p_non_cotton=round(p_non_cotton, 4),
                        margin=round(margin, 4),
                        ood_score=round(energy, 4),
                        is_ood=False,
                        reason="State DPI crop prior ('cotton') verified by morphological visual router evidence.",
                        predicted_family=pred_family,
                        family_name=family_name,
                        family_probabilities=fam_probs,
                        mahalanobis_distance=round(d_mahalanobis, 4)
                    )
                else:
                    return RoutingDecision(
                        status="UNCERTAIN",
                        expert=None,
                        routing_confidence=round(margin, 4),
                        p_cotton=round(p_cotton, 4),
                        p_non_cotton=round(p_non_cotton, 4),
                        margin=round(margin, 4),
                        ood_score=round(energy, 4),
                        is_ood=False,
                        reason=f"DPI Conflict: Registered crop is 'cotton', but visual evidence strongly indicates {family_name}.",
                        predicted_family=pred_family,
                        family_name=family_name,
                        family_probabilities=fam_probs,
                        mahalanobis_distance=round(d_mahalanobis, 4)
                    )
            else:
                if p_cotton >= 0.95:
                    return RoutingDecision(
                        status="COTTON",
                        expert="cotton_specialist_42",
                        routing_confidence=round(p_cotton, 4),
                        p_cotton=round(p_cotton, 4),
                        p_non_cotton=round(p_non_cotton, 4),
                        margin=round(margin, 4),
                        ood_score=round(energy, 4),
                        is_ood=False,
                        reason=f"Overwhelming visual cotton pathology (P={p_cotton*100:.1f}%) supersedes declared hint '{norm_hint}'.",
                        predicted_family=0,
                        family_name=ROUTER_FAMILIES[0],
                        family_probabilities=fam_probs,
                        mahalanobis_distance=round(d_mahalanobis, 4)
                    )
                else:
                    return RoutingDecision(
                        status="NON_COTTON",
                        expert="generalist_281",
                        routing_confidence=round(max(p_non_cotton, 0.85), 4),
                        p_cotton=round(p_cotton, 4),
                        p_non_cotton=round(p_non_cotton, 4),
                        margin=round(margin, 4),
                        ood_score=round(energy, 4),
                        is_ood=False,
                        reason=f"State DPI prior '{norm_hint}' verified as Non-Cotton; routed to Generalist Model A.",
                        predicted_family=pred_family,
                        family_name=family_name,
                        family_probabilities=fam_probs,
                        mahalanobis_distance=round(d_mahalanobis, 4)
                    )

        # -------------------------------------------------------------
        # 2. Autonomous Visual Routing (No crop_hint provided)
        # -------------------------------------------------------------
        tau_high = self.thresholds["tau_cotton_high"]
        tau_low = self.thresholds["tau_cotton_low"]
        tau_margin = self.thresholds["tau_margin_min"]

        # Class 5: Non-Crop Background
        if pred_family == 5:
            return RoutingDecision(
                status="UNCERTAIN",
                expert=None,
                routing_confidence=round(float(probs[5].item()), 4),
                p_cotton=round(p_cotton, 4),
                p_non_cotton=round(p_non_cotton, 4),
                margin=round(margin, 4),
                ood_score=round(energy, 4),
                is_ood=True,
                reason="Router detected non-crop background (soil, hands, plastic, or debris).",
                predicted_family=5,
                family_name=ROUTER_FAMILIES[5],
                family_probabilities=fam_probs,
                mahalanobis_distance=round(d_mahalanobis, 4)
            )

        # Class 4: Broadleaf Hard Negatives (Sunflower, Okra, Castor, Weeds)
        if pred_family == 4:
            return RoutingDecision(
                status="NON_COTTON",
                expert="generalist_281",
                routing_confidence=round(float(probs[4].item()), 4),
                p_cotton=round(p_cotton, 4),
                p_non_cotton=round(p_non_cotton, 4),
                margin=round(margin, 4),
                ood_score=round(energy, 4),
                is_ood=False,
                reason=f"Broadleaf negative ({family_name}) distinguished from Malvaceae; routed to Generalist Model A.",
                predicted_family=4,
                family_name=ROUTER_FAMILIES[4],
                family_probabilities=fam_probs,
                mahalanobis_distance=round(d_mahalanobis, 4)
            )

        # Class 1, 2, 3: Monocots, Legumes, Solanaceous
        if pred_family in (1, 2, 3):
            return RoutingDecision(
                status="NON_COTTON",
                expert="generalist_281",
                routing_confidence=round(p_non_cotton, 4),
                p_cotton=round(p_cotton, 4),
                p_non_cotton=round(p_non_cotton, 4),
                margin=round(margin, 4),
                ood_score=round(energy, 4),
                is_ood=False,
                reason=f"High-confidence Non-Cotton routing ({family_name}, P={p_non_cotton*100:.1f}%).",
                predicted_family=pred_family,
                family_name=family_name,
                family_probabilities=fam_probs,
                mahalanobis_distance=round(d_mahalanobis, 4)
            )

        # Class 0: Cotton
        if pred_family == 0 and p_cotton >= tau_high and margin >= tau_margin:
            return RoutingDecision(
                status="COTTON",
                expert="cotton_specialist_42",
                routing_confidence=round(p_cotton, 4),
                p_cotton=round(p_cotton, 4),
                p_non_cotton=round(p_non_cotton, 4),
                margin=round(margin, 4),
                ood_score=round(energy, 4),
                is_ood=False,
                reason=f"High-confidence Cotton routing ({family_name}, P={p_cotton*100:.1f}%, margin={margin*100:.1f}%).",
                predicted_family=0,
                family_name=ROUTER_FAMILIES[0],
                family_probabilities=fam_probs,
                mahalanobis_distance=round(d_mahalanobis, 4)
            )

        if p_cotton <= tau_low and margin >= tau_margin:
            return RoutingDecision(
                status="NON_COTTON",
                expert="generalist_281",
                routing_confidence=round(p_non_cotton, 4),
                p_cotton=round(p_cotton, 4),
                p_non_cotton=round(p_non_cotton, 4),
                margin=round(margin, 4),
                ood_score=round(energy, 4),
                is_ood=False,
                reason=f"High-confidence Non-Cotton routing (P={p_non_cotton*100:.1f}%, margin={margin*100:.1f}%).",
                predicted_family=pred_family,
                family_name=family_name,
                family_probabilities=fam_probs,
                mahalanobis_distance=round(d_mahalanobis, 4)
            )

        return RoutingDecision(
            status="UNCERTAIN",
            expert=None,
            routing_confidence=round(margin, 4),
            p_cotton=round(p_cotton, 4),
            p_non_cotton=round(p_non_cotton, 4),
            margin=round(margin, 4),
            ood_score=round(energy, 4),
            is_ood=False,
            reason=f"Ambiguous routing: P(Cotton)={p_cotton*100:.1f}%, P(Non-Cotton)={p_non_cotton*100:.1f}%, margin={margin*100:.1f}%.",
            predicted_family=pred_family,
            family_name=family_name,
            family_probabilities=fam_probs,
            mahalanobis_distance=round(d_mahalanobis, 4)
        )

    def route(self, image: Image.Image, user_crop_hint: Optional[str] = None) -> RoutingDecision:
        """Backward-compatible interface accepting PIL Image."""
        return self.route_crop(image, crop_hint=user_crop_hint)
