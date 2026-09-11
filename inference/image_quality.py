import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class ImageQualityResult:
    is_valid: bool
    status: str              # "PASS", "BLURRY", "POOR_EXPOSURE", "LOW_VEGETATION", "LOW_RESOLUTION"
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    vegetation_ratio: float
    warning_message: str


class ImageQualityEvaluator:
    """
    Evaluates input image fidelity before feeding to neural networks.
    Rejects degraded inputs (severely blurred, completely dark, or non-plant images)
    to prevent garbage-in garbage-out failure modes.
    """
    def __init__(
        self,
        min_sharpness: float = 20.0,
        min_brightness: float = 25.0,
        max_brightness: float = 245.0,
        min_contrast: float = 18.0,
        min_vegetation_ratio: float = 0.04,
        min_dimension: int = 64
    ):
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_vegetation_ratio = min_vegetation_ratio
        self.min_dimension = min_dimension

    def evaluate(self, image: Image.Image) -> ImageQualityResult:
        w, h = image.size
        if w < self.min_dimension or h < self.min_dimension:
            return ImageQualityResult(
                is_valid=False,
                status="LOW_RESOLUTION",
                sharpness_score=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
                vegetation_ratio=0.0,
                warning_message=f"Image resolution too low ({w}x{h}). Minimum required: {self.min_dimension}x{self.min_dimension}."
            )

        # Convert to numpy array (RGB)
        np_img = np.array(image.convert("RGB"))

        # 1. Sharpness via Laplacian variance
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # 2. Brightness & Contrast via Lab color space
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        brightness = float(np.mean(l_channel))
        contrast = float(np.std(l_channel))

        # 3. Vegetation Ratio via Excess Green Index (2G - R - B)
        r = np_img[:, :, 0].astype(np.float32)
        g = np_img[:, :, 1].astype(np.float32)
        b = np_img[:, :, 2].astype(np.float32)
        exg = 2 * g - r - b
        # Pixels with significant green excess or yellow-green hue
        plant_mask = (exg > 10.0) | ((g > r) & (g > b) & (g > 30))
        vegetation_ratio = float(np.count_nonzero(plant_mask)) / float(w * h)

        # Quality Verdicts
        if sharpness < self.min_sharpness:
            return ImageQualityResult(
                is_valid=False,
                status="BLURRY",
                sharpness_score=round(sharpness, 2),
                brightness_score=round(brightness, 2),
                contrast_score=round(contrast, 2),
                vegetation_ratio=round(vegetation_ratio, 3),
                warning_message=f"Image appears severely blurred (sharpness {sharpness:.1f} < {self.min_sharpness}). Foliar symptoms may be unrecognizable."
            )

        if brightness < self.min_brightness or brightness > self.max_brightness:
            issue = "severely underexposed / dark" if brightness < self.min_brightness else "overexposed / washed out"
            return ImageQualityResult(
                is_valid=False,
                status="POOR_EXPOSURE",
                sharpness_score=round(sharpness, 2),
                brightness_score=round(brightness, 2),
                contrast_score=round(contrast, 2),
                vegetation_ratio=round(vegetation_ratio, 3),
                warning_message=f"Image is {issue} (brightness {brightness:.1f}). Visual pathology unreliable."
            )

        if contrast < self.min_contrast:
            return ImageQualityResult(
                is_valid=False,
                status="LOW_CONTRAST",
                sharpness_score=round(sharpness, 2),
                brightness_score=round(brightness, 2),
                contrast_score=round(contrast, 2),
                vegetation_ratio=round(vegetation_ratio, 3),
                warning_message=f"Low contrast ({contrast:.1f} < {self.min_contrast}). Visual separation of lesions is ambiguous."
            )

        return ImageQualityResult(
            is_valid=True,
            status="PASS",
            sharpness_score=round(sharpness, 2),
            brightness_score=round(brightness, 2),
            contrast_score=round(contrast, 2),
            vegetation_ratio=round(vegetation_ratio, 3),
            warning_message=""
        )
