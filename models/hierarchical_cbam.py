"""
AgriVision AI — Hierarchical Multi-Task Neural Network Architecture.
Integrates an EfficientNet-B5 backbone + CBAM attention module with:
1. Disease Head (281 raw classes -> aggregated to 167 canonical classes)
   - Exactly matches the Model A classifier structure (Linear 2048 -> 512 -> 281)
   - Initialized with pre-trained Model A weights.
2. Dedicated Crop Head (41 botanical crop families)
   - Dedicated multi-layer perceptron (Linear 2048 -> 256 -> 41)
   - Directly supervised on verified ground-truth crop labels.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from .cbam import CBAM

class EfficientNetB5_CBAM_Hierarchical(nn.Module):
    def __init__(
        self,
        num_disease_classes: int = 281,
        num_crop_classes: int = 41,
        pretrained: bool = False,
        drop_rate: float = 0.4
    ):
        super(EfficientNetB5_CBAM_Hierarchical, self).__init__()
        self.num_disease_classes = num_disease_classes
        self.num_crop_classes = num_crop_classes
        
        # 1. Shared EfficientNet-B5 Backbone
        if pretrained and TORCHVISION_AVAILABLE:
            weights = models.EfficientNet_B5_Weights.DEFAULT
            base_model = models.efficientnet_b5(weights=weights)
        elif TORCHVISION_AVAILABLE:
            base_model = models.efficientnet_b5(weights=None)
        else:
            raise RuntimeError("torchvision is required for EfficientNet-B5 architecture")

        self.features = base_model.features
        in_features = 2048 # EfficientNet-B5 top stage channel depth
        
        # 2. Shared Integrated CBAM Attention Block
        self.cbam = CBAM(in_channels=in_features, reduction_ratio=16, kernel_size=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Disease Classification Head (Identical architecture to Model A)
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(p=drop_rate * 0.5, inplace=False),
            nn.Linear(512, num_disease_classes)
        )

        # 4. Dedicated Botanical Crop-Family Classification Head
        self.crop_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.15, inplace=False),
            nn.Linear(256, num_crop_classes)
        )

        # Telemetry hooks
        self.last_features: Optional[torch.Tensor] = None
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        return_crop: bool = True
    ) -> Any:
        """
        Forward pass through shared visual representation.
        If return_crop is True: returns (disease_logits, crop_logits)
        If return_crop is False: returns disease_logits (for exact Model A compatibility)
        """
        feat = self.features(x)
        if not self.training:
            self.last_features = feat
            
        feat_attended = self.cbam(feat)
        if not self.training:
            self.attention_weights = feat_attended

        pooled = self.avgpool(feat_attended)
        flattened = torch.flatten(pooled, 1)

        disease_logits = self.classifier(flattened)

        if return_crop:
            crop_logits = self.crop_classifier(flattened)
            return disease_logits, crop_logits
        return disease_logits

    def load_model_a_weights(self, checkpoint_path: str):
        """
        Loads pre-trained Model A checkpoint into shared backbone, CBAM,
        and the disease classifier head. The new crop head is untouched.
        """
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_dict = self.state_dict()
        
        matched_keys = 0
        for k, v in sd.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                matched_keys += 1
                
        self.load_state_dict(model_dict)
        print(f"[HierarchicalModel] Loaded {matched_keys} / {len(sd)} weights from Model A: {checkpoint_path}")

    def get_spatial_attention_map(self, x: torch.Tensor) -> np.ndarray:
        """Extract spatial attention intensity map from CBAM."""
        self.eval()
        with torch.no_grad():
            feat = self.features(x)
            ca_feat = self.cbam.channel_attention(feat)
            avg_out = torch.mean(ca_feat, dim=1, keepdim=True)
            max_out, _ = torch.max(ca_feat, dim=1, keepdim=True)
            combined = torch.cat([avg_out, max_out], dim=1)
            spatial_weights = self.cbam.spatial_attention.sigmoid(
                self.cbam.spatial_attention.conv(combined)
            )
            upsampled = F.interpolate(
                spatial_weights,
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
            attn_map = upsampled.squeeze().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            return attn_map
