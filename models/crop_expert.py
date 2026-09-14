"""
AgriVision AI — Dedicated Crop Expert Model Architecture.

Implements an independent Crop Expert visual classifier based on ConvNeXt-Tiny
(or configurable backbone), outputting calibrated probabilities P(crop | image)
across the canonical crop taxonomy.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from torchvision import models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class CropExpertModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 41,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        drop_rate: float = 0.2
    ):
        super(CropExpertModel, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "convnext_tiny":
            if TIMM_AVAILABLE:
                try:
                    self.encoder = timm.create_model(
                        "convnext_tiny",
                        pretrained=pretrained,
                        num_classes=0, # Remove default head
                        drop_rate=drop_rate
                    )
                    in_features = self.encoder.num_features
                except Exception:
                    self.encoder = None
            else:
                self.encoder = None

            if self.encoder is None and TORCHVISION_AVAILABLE:
                weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                base = tv_models.convnext_tiny(weights=weights)
                self.encoder = base.features
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                in_features = 768
            elif self.encoder is None:
                raise RuntimeError("Neither timm nor torchvision could load convnext_tiny")
            else:
                self.avgpool = None

            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, num_classes)
            )

        elif backbone_name == "efficientnet_b0":
            if TORCHVISION_AVAILABLE:
                weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base = tv_models.efficientnet_b0(weights=weights)
                self.encoder = base.features
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                in_features = 1280
                self.head = nn.Sequential(
                    nn.Dropout(p=drop_rate),
                    nn.Linear(in_features, num_classes)
                )
            else:
                raise RuntimeError("torchvision required for efficientnet_b0")
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.avgpool is not None:
            feat = self.encoder(x)
            feat = self.avgpool(feat)
            feat = torch.flatten(feat, 1)
        else:
            feat = self.encoder(x)
        logits = self.head(feat)
        return logits

def build_crop_expert(
    weights_path: Optional[str] = None,
    num_classes: int = 41,
    backbone_name: str = "convnext_tiny",
    device: str = "cuda"
) -> CropExpertModel:
    """Builds and optionally loads CropExpertModel checkpoint."""
    model = CropExpertModel(num_classes=num_classes, backbone_name=backbone_name, pretrained=weights_path is None)
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[CropExpert] Loaded checkpoint from: {weights_path}")
    model.to(device)
    model.eval()
    return model
