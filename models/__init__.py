"""
Model architectures for the Agriculture Diagnostic Stack:
- EfficientNet-B5 with CBAM Attention
- YOLOv8n Pest Detection
- ResNet-34 U-Net Lesion Segmentation
"""

from .cbam import CBAM, ChannelAttention, SpatialAttention
from .efficientnet_cbam import EfficientNetB5_CBAM, build_efficientnet_cbam
from .yolo_pest import YOLOv8PestDetector
from .unet_segmenter import ResNet34_UNet, LesionSegmenter

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "EfficientNetB5_CBAM",
    "build_efficientnet_cbam",
    "YOLOv8PestDetector",
    "ResNet34_UNet",
    "LesionSegmenter",
]
