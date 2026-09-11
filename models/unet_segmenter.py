import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple

try:
    from torchvision import models, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34_UNet(nn.Module):
    """
    U-Net architecture with pre-trained ResNet-34 encoder for agricultural
    leaf & lesion semantic segmentation.
    Outputs 2 channels:
      Channel 0: Leaf area mask
      Channel 1: Lesion infected area mask
    """
    def __init__(self, out_channels: int = 2, pretrained: bool = True):
        super(ResNet34_UNet, self).__init__()
        
        if pretrained and TORCHVISION_AVAILABLE:
            weights = models.ResNet34_Weights.DEFAULT
            encoder = models.resnet34(weights=weights)
        elif TORCHVISION_AVAILABLE:
            encoder = models.resnet34(weights=None)
        else:
            raise RuntimeError("torchvision required for ResNet34_UNet")

        # Encoder stages
        self.init_conv = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        ) # [B, 64, H/2, W/2]
        self.maxpool = encoder.maxpool # [B, 64, H/4, W/4]
        
        self.enc1 = encoder.layer1 # [B, 64, H/4, W/4]
        self.enc2 = encoder.layer2 # [B, 128, H/8, W/8]
        self.enc3 = encoder.layer3 # [B, 256, H/16, W/16]
        self.enc4 = encoder.layer4 # [B, 512, H/32, W/32]
        
        # Center bridge
        self.bridge = ConvBlock(512, 512)
        
        # Decoder stages
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        
        # Final head (upsample to original input resolution)
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]
        
        # Encoder
        x0 = self.init_conv(x)      # 64, H/2, W/2
        x1 = self.maxpool(x0)       # 64, H/4, W/4
        x1 = self.enc1(x1)          # 64, H/4, W/4
        x2 = self.enc2(x1)          # 128, H/8, W/8
        x3 = self.enc3(x2)          # 256, H/16, W/16
        x4 = self.enc4(x3)          # 512, H/32, W/32
        
        # Bridge
        b = self.bridge(x4)
        
        # Decoder with skip connections
        d4 = self.dec4(b, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        if out.shape[2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
            
        return out


class LesionSegmenter:
    """
    Inference orchestrator for ResNet-34 U-Net.
    Segments leaf tissue and diseased lesions, calculates severity percentage,
    and constructs segmented visual overlays.
    """
    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = ResNet34_UNet(out_channels=2, pretrained=(weights_path is None))
        
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment(self, image: Image.Image) -> Dict[str, Any]:
        """
        Segment a PIL leaf image into healthy leaf tissue and lesions.
        Returns:
            - infected_area_pct: float percentage
            - severity_category: 'Mild', 'Moderate', or 'Severe'
            - leaf_pixel_count: int
            - lesion_pixel_count: int
            - overlay_image: PIL Image with color-coded masks
        """
        w_orig, h_orig = image.size
        rgb_img = image.convert("RGB")
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor)
            probs = torch.sigmoid(preds).squeeze(0).cpu().numpy()
            
            leaf_mask = (probs[0] > 0.45).astype(np.uint8)
            lesion_mask = (probs[1] > 0.50).astype(np.uint8)
            
            # Constrain lesions to be within leaf area
            lesion_mask = (lesion_mask * leaf_mask).astype(np.uint8)
            
            leaf_pixels = int(np.sum(leaf_mask))
            lesion_pixels = int(np.sum(lesion_mask))
            
            # If leaf mask is too small or model uncalibrated, use adaptive color-threshold heuristic
            if leaf_pixels < 500:
                leaf_mask, lesion_mask, leaf_pixels, lesion_pixels = self._adaptive_fallback_segmentation(rgb_img)

            pct = (lesion_pixels / max(leaf_pixels, 1)) * 100.0
            pct = round(min(max(pct, 0.0), 100.0), 2)
            
            if pct < 10.0:
                severity = "Mild"
            elif pct <= 30.0:
                severity = "Moderate"
            else:
                severity = "Severe"
                
            overlay = self._create_overlay(rgb_img.resize((256, 256)), leaf_mask, lesion_mask)

        return {
            "infected_area_pct": pct,
            "severity_category": severity,
            "leaf_pixel_count": leaf_pixels,
            "lesion_pixel_count": lesion_pixels,
            "overlay_image": overlay.resize((w_orig, h_orig), Image.Resampling.BILINEAR)
        }

    def _adaptive_fallback_segmentation(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Adaptive HSV segmentation for leaf and necrotic lesion boundaries."""
        arr = np.array(img.resize((256, 256)))
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # Leaf region: pixels where green is dominant or leaf hue
        leaf_mask = ((g > r * 0.9) | (g > b * 1.1) | (g > 60)).astype(np.uint8)
        
        # Lesion region: necrotic brown, yellowing, or dark spots inside leaf
        diff_rg = np.abs(r.astype(int) - g.astype(int))
        lesion_mask = (leaf_mask & ((r > 130) & (b < 100) & (diff_rg > 25) | ((r < 70) & (g < 70) & (b < 70)))).astype(np.uint8)
        
        leaf_px = int(np.sum(leaf_mask))
        lesion_px = int(np.sum(lesion_mask))
        return leaf_mask, lesion_mask, leaf_px, lesion_px

    def _create_overlay(self, base_img: Image.Image, leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> Image.Image:
        """Create high-contrast transparent overlay: green for leaf, bright red for lesions."""
        base_arr = np.array(base_img).astype(np.float32)
        
        # Color overlays:
        # Healthy leaf: slight green tint
        healthy = (leaf_mask == 1) & (lesion_mask == 0)
        base_arr[healthy, 1] = np.clip(base_arr[healthy, 1] * 1.15 + 20, 0, 255)
        
        # Lesions: bright coral/red highlight
        lesions = (lesion_mask == 1)
        base_arr[lesions, 0] = np.clip(base_arr[lesions, 0] * 0.4 + 210, 0, 255)
        base_arr[lesions, 1] = np.clip(base_arr[lesions, 1] * 0.4 + 40, 0, 255)
        base_arr[lesions, 2] = np.clip(base_arr[lesions, 2] * 0.4 + 40, 0, 255)
        
        return Image.fromarray(base_arr.astype(np.uint8))
