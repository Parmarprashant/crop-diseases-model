import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PEST_CLASSES = [
    "American Bollworm",
    "Army worm",
    "Cotton Aphid",
    "Cotton Mealy Bug",
    "Cotton Whitefly",
    "Pink Bollworm",
    "Red Cotton Bug",
    "Rice Stem Borer",
    "Wheat Aphid",
    "Wheat Mite",
    "Thrips",
    "Maize Fall Armyworm",
    "Maize Stem Borer"
]


class YOLOv8PestDetector:
    """
    YOLOv8n-based object detection model specialized in identifying and localizing
    crop pests with bounding box coordinates and species labels.
    """
    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.25, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        
        if weights_path and os.path.exists(weights_path) and ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(weights_path)
                self.is_loaded = True
            except Exception as e:
                print(f"[YOLOv8] Warning loading custom weights {weights_path}: {e}")
        elif ULTRALYTICS_AVAILABLE:
            try:
                # Default to yolov8n.pt pretrained backbone
                self.model = YOLO("yolov8n.pt")
                self.is_loaded = True
            except Exception as e:
                print(f"[YOLOv8] Notice: Default YOLOv8n initialized ({e})")

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run pest detection on a PIL Image.
        Returns:
            - detections: list of { "label": str, "confidence": float, "box": [x1, y1, x2, y2] }
            - count: total pest instances found
            - annotated_image: PIL Image with rendered bounding boxes & tags
        """
        w, h = image.size
        detections: List[Dict[str, Any]] = []

        if self.is_loaded and self.model is not None:
            try:
                results = self.model(image, conf=self.conf_threshold, device=self.device, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = r.names.get(cls_id, f"Pest_{cls_id}")
                        
                        detections.append({
                            "label": cls_name,
                            "confidence": round(conf, 4),
                            "box": [round(c, 1) for c in coords]
                        })
            except Exception as e:
                print(f"[YOLOv8] Inference warning: {e}")

        # If no detections found or ultralytics fallback:
        image.load()
        annotated_img = self._draw_detections(image.copy(), detections)

        return {
            "pests_detected": len(detections) > 0,
            "pest_count": len(detections),
            "detections": detections,
            "annotated_image": annotated_img
        }

    def _draw_detections(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """Draw bounding boxes and class tags with high-visibility styling."""
        draw = ImageDraw.Draw(image)
        for det in detections:
            box = det["box"]
            label = f"{det['label']} {det['confidence']*100:.1f}%"
            
            # Bright warning border for pests
            draw.rectangle(box, outline="#FF3B30", width=3)
            
            # Label badge background
            x1, y1, _, _ = box
            tag_y = max(0, y1 - 18)
            draw.rectangle([x1, tag_y, x1 + len(label) * 8 + 8, tag_y + 18], fill="#FF3B30")
            draw.text((x1 + 4, tag_y + 2), label, fill="#FFFFFF")
            
        return image
