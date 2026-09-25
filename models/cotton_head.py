"""
AgriVision Ultra v5.0 — Sub-Class ArcFace (Additive Angular Margin) Head.

Forces cotton foliar disease feature vectors onto a hypersphere with wide inter-class angular margins:
    L_ArcFace = -log ( exp(s * cos(theta_{y_i} + m)) / (exp(s * cos(theta_{y_i} + m)) + sum_{j != y_i} exp(s * cos(theta_j))) )
where:
    scale factor s = 30.0
    angular margin m = 0.35 rad (~20 degrees)

Drives distinct clustering among visually overlapping leaf lesions:
Target Spot vs. Alternaria vs. Cercospora vs. Bacterial Blight.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


COTTON_DISEASE_CLASSES = [
    "Target Spot (Corynespora cassiicola)",
    "Alternaria Leaf Spot (Alternaria macrospora)",
    "Cercospora Leaf Spot (Cercospora gossypina)",
    "Bacterial Blight / Angular Leaf Spot (Xanthomonas)",
    "Anthracnose on Cotton (Colletotrichum gossypii)",
    "Cotton Rust (Puccinia schedonnardi)",
    "Ramularia / Gray Mildew (Ramularia areola)",
    "Cotton Leaf Curl Virus (CLCuV)",
    "Boll Rot (Mature stage)",
    "Cotton Aphid / Sooty Mold (Aphis gossypii)",
    "Fusarium / Verticillium Wilt",
    "Healthy Cotton Leaf"
]


class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin (ArcFace) projection layer.
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 12,
        s: float = 30.0,
        m: float = 0.35,
        easy_margin: bool = False
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Weight matrix representing class centers on hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute trigonometric constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with normalized features and weights.
        If labels are provided, applies angular margin m to the target class angle theta_{y_i}.
        If labels is None (inference mode), outputs scaled cosine similarity s * cos(theta).
        """
        # Cosine similarity: cos(theta) = <x_norm, w_norm>
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm) # Shape: (B, num_classes)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None or not self.training:
            # Inference: return s * cos(theta)
            return cosine * self.s

        # Training with Additive Angular Margin: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0.0, 1.0)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding for ground truth class
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CottonArcFaceModel(nn.Module):
    """
    Model B Cotton Disease Classifier with ArcFace head.
    Backbone: EfficientNet-B5 features + CBAM attention
    Bottleneck: Linear 2048 -> 512 (SiLU)
    Head: ArcMarginProduct (512 -> 12)
    """
    def __init__(
        self,
        num_classes: int = 12,
        s: float = 30.0,
        m: float = 0.35,
        pretrained: bool = False
    ):
        super(CottonArcFaceModel, self).__init__()
        from models.efficientnet_cbam import EfficientNetB5_CBAM
        self.base = EfficientNetB5_CBAM(num_classes=num_classes, pretrained=pretrained)
        
        # Replace final classification head with 512-dim projection + ArcMarginProduct
        self.bottleneck = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True)
        )
        self.arcface_head = ArcMarginProduct(
            in_features=512,
            num_classes=num_classes,
            s=s,
            m=m
        )

    def extract_bottleneck(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts 512-dim bottleneck features and CBAM spatial attention map."""
        feat = self.base.features(x)
        feat_att = self.base.cbam(feat)
        pooled = self.base.avgpool(feat_att)
        flat = torch.flatten(pooled, 1)
        h = self.bottleneck(flat)
        return h, feat_att

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h, _ = self.extract_bottleneck(x)
        logits = self.arcface_head(h, labels)
        return logits
