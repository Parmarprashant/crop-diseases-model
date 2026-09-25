"""
AgriVision Ultra v5.0 — Saliency-to-Lesion Consistency Regularization.

Enforces that Model B's CBAM spatial attention map M_s(x) in [0, 1]^{H x W}
strictly aligns with the binary lesion segmentation mask produced by the ResNet-34 U-Net:
    L_consistency = BCE(M_s(x), Downsample(UNet(x)))

Guarantees that diagnosis is anchored on actual necrotic/chlorotic tissue lesions
rather than specular reflections, background soil, veins, or healthy leaf margins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional


class SaliencyLesionConsistencyLoss(nn.Module):
    """
    Consistency loss between CBAM spatial attention and U-Net lesion segmentation.
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 0.5):
        super(SaliencyLesionConsistencyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss(reduction="mean")

    def forward(
        self,
        spatial_attention: torch.Tensor,
        unet_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        spatial_attention: (B, 1, H, W) or (B, H, W), values in [0, 1]
        unet_mask: (B, 1, H_u, W_u) or (B, 2, H_u, W_u) or (B, H_u, W_u)
        """
        if spatial_attention.dim() == 3:
            spatial_attention = spatial_attention.unsqueeze(1)

        if unet_mask.dim() == 3:
            unet_mask = unet_mask.unsqueeze(1)
        elif unet_mask.dim() == 4 and unet_mask.size(1) == 2:
            # If 2 channels (background, lesion), take lesion channel (channel 1)
            unet_mask = unet_mask[:, 1:2, :, :]

        # Ensure spatial attention is clamped in [1e-7, 1 - 1e-7]
        spatial_attention = torch.clamp(spatial_attention, 1e-7, 1.0 - 1e-7)

        # Downsample / resize U-Net mask to match spatial attention resolution
        target_h, target_w = spatial_attention.size(2), spatial_attention.size(3)
        if unet_mask.size(2) != target_h or unet_mask.size(3) != target_w:
            downsampled_mask = F.interpolate(
                unet_mask.float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )
        else:
            downsampled_mask = unet_mask.float()

        downsampled_mask = torch.clamp(downsampled_mask, 0.0, 1.0)

        # 1. Binary Cross-Entropy Loss
        loss_bce = self.bce(spatial_attention, downsampled_mask)

        # 2. Soft Dice Loss for foreground lesion alignment
        intersection = (spatial_attention * downsampled_mask).sum(dim=(2, 3))
        union = spatial_attention.sum(dim=(2, 3)) + downsampled_mask.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        loss_dice = (1.0 - dice_score).mean()

        total_loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_dice": float(loss_dice.item()),
            "mean_dice_score": float(dice_score.mean().item())
        }

        return total_loss, metrics


def compute_saliency_iou(
    attention_map: np.ndarray,
    lesion_mask: np.ndarray,
    attn_threshold: float = 0.5,
    mask_threshold: float = 0.5
) -> float:
    """
    Computes Intersection-over-Union (IoU) between binarized attention and lesion mask.
    attention_map: 2D numpy array [H, W], normalized [0, 1]
    lesion_mask: 2D numpy array [H, W], normalized [0, 1] or binary {0, 1}
    """
    # Resize attention map if needed
    if attention_map.shape != lesion_mask.shape:
        import cv2
        attention_map = cv2.resize(
            attention_map,
            (lesion_mask.shape[1], lesion_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    bin_attn = attention_map >= attn_threshold
    bin_mask = lesion_mask >= mask_threshold

    intersection = np.logical_and(bin_attn, bin_mask).sum()
    union = np.logical_or(bin_attn, bin_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)
