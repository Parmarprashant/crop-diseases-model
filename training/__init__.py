"""
Training pipeline for agricultural model stack:
- dataset.py: Dataset loaders sourcing strictly from MAIN DATA
- train_classifier.py: EfficientNet-B5 + CBAM Attention trainer
- train_yolo.py: YOLOv8n pest trainer
- train_unet.py: ResNet-34 U-Net lesion segmentation trainer
"""

from .dataset import MainDataCropDataset, get_data_loaders

__all__ = ["MainDataCropDataset", "get_data_loaders"]
