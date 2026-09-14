import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from .cbam import CBAM

try:
    from torchvision import models, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class EfficientNetB5_CBAM(nn.Module):
    """
    EfficientNet-B5 architecture augmented with Convolutional Block Attention Module (CBAM)
    for high-precision plant disease detection.
    """
    def __init__(self, num_classes: int = 42, pretrained: bool = True, drop_rate: float = 0.4):
        super(EfficientNetB5_CBAM, self).__init__()
        self.num_classes = num_classes
        
        # Load EfficientNet-B5 backbone
        if pretrained and TORCHVISION_AVAILABLE:
            weights = models.EfficientNet_B5_Weights.DEFAULT
            base_model = models.efficientnet_b5(weights=weights)
        elif TORCHVISION_AVAILABLE:
            base_model = models.efficientnet_b5(weights=None)
        else:
            raise RuntimeError("torchvision is required for EfficientNet-B5 architecture")

        self.features = base_model.features
        # EfficientNet-B5 top feature map channel depth is 2048
        in_features = 2048
        
        # Integrated CBAM Attention Block
        self.cbam = CBAM(in_channels=in_features, reduction_ratio=16, kernel_size=7)
        
        # Pooling & Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate * 0.5, inplace=True),
            nn.Linear(512, num_classes)
        )
        
        # Hook holders for Grad-CAM / Attention visualization
        self.last_features: Optional[torch.Tensor] = None
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features through EfficientNet stages
        feat = self.features(x)
        if not self.training:
            self.last_features = feat
        
        # Apply Channel & Spatial Attention
        feat_attended = self.cbam(feat)
        if not self.training:
            self.attention_weights = feat_attended
        
        # Global Pooling and dense classification
        pooled = self.avgpool(feat_attended)
        flattened = torch.flatten(pooled, 1)
        out = self.classifier(flattened)
        return out

    def get_spatial_attention_map(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract spatial attention intensity from the CBAM module to visualize
        lesion focus areas.
        """
        self.eval()
        with torch.no_grad():
            feat = self.features(x)
            # Channel attention first
            ca_feat = self.cbam.channel_attention(feat)
            # Spatial attention forward pass
            avg_out = torch.mean(ca_feat, dim=1, keepdim=True)
            max_out, _ = torch.max(ca_feat, dim=1, keepdim=True)
            combined = torch.cat([avg_out, max_out], dim=1)
            spatial_weights = self.cbam.spatial_attention.sigmoid(
                self.cbam.spatial_attention.conv(combined)
            )
            
            # Upsample spatial attention map to input size
            upsampled = F.interpolate(
                spatial_weights,
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
            attn_map = upsampled.squeeze().cpu().numpy()
            # Normalize to 0.0 - 1.0
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            return attn_map


def build_efficientnet_cbam(
    num_classes: int = 42,
    weights_path: Optional[str] = None,
    device: str = "cpu"
) -> EfficientNetB5_CBAM:
    """
    Factory function to instantiate EfficientNetB5_CBAM model on the target device
    with optional pre-trained checkpoints.
    """
    model = EfficientNetB5_CBAM(num_classes=num_classes, pretrained=(weights_path is None))
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class DiseaseClassifierInference:
    """
    End-to-end inference wrapper for EfficientNet-B5 + CBAM with preprocessing,
    softmax probabilities, and explainability maps.
    """
    def __init__(
        self,
        model: EfficientNetB5_CBAM,
        class_names: List[str],
        device: str = "cpu",
        img_size: int = 256,
        use_tta: Optional[bool] = None
    ):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size
        self.use_tta = use_tta if use_tta is not None else (os.environ.get("TTA_ENABLED", "1").lower() in ("1", "true", "yes"))
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Calibrated image resolution matching checkpoint (256x256)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, top_k: int = 3, use_tta: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run inference on a PIL Image with optional Horizontal Flip TTA (Test-Time Augmentation).
        Returns predicted class, confidence, top-k predictions, and CBAM attention map.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        enable_tta = self.use_tta if use_tta is None else use_tta
        
        with torch.no_grad():
            logits_orig = self.model(tensor)
            probs_orig = F.softmax(logits_orig, dim=1)

            if enable_tta:
                # Horizontal flip TTA: flip along width dimension (dim 3)
                tensor_flipped = torch.flip(tensor, dims=[3])
                logits_flipped = self.model(tensor_flipped)
                probs_flipped = F.softmax(logits_flipped, dim=1)
                probs_ensemble = (probs_orig + probs_flipped) / 2.0
            else:
                probs_ensemble = probs_orig

            probs = probs_ensemble.squeeze(0)
            
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
            top_probs = top_probs.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            
            best_idx = top_indices[0]
            best_class = self.class_names[best_idx] if best_idx < len(self.class_names) else f"Class_{best_idx}"
            best_conf = float(top_probs[0])
            
            top_predictions = [
                {
                    "class_name": self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}",
                    "confidence": round(float(prob), 4)
                }
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            # Generate CBAM spatial attention heatmap from original un-flipped image
            attn_map = self.model.get_spatial_attention_map(tensor)

            # CRITICAL OOD SAFETY CONTRACT:
            # Always pass original logits to OOD detector since calibration was done on original orientation
            raw_logits = logits_orig.squeeze(0).cpu().numpy()
            all_probs = probs.cpu().numpy()

        return {
            "predicted_class": best_class,
            "confidence": round(best_conf, 4),
            "top_predictions": top_predictions,
            "attention_map": attn_map,  # 2D numpy array [456, 456]
            "is_confident": best_conf >= 0.80,
            "raw_logits": raw_logits,
            "all_probabilities": all_probs
        }
