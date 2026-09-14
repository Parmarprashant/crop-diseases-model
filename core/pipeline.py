import io
import base64
from typing import Dict, Any, List, Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import concurrent.futures
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np

from models.efficientnet_cbam import DiseaseClassifierInference, build_efficientnet_cbam
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.advisory_engine import AdvisoryEngine
from core.gemini_fallback import GeminiVisionFallback


def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Helper to convert PIL Image to base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def attention_map_to_base64(attn_map: np.ndarray, original_img: Image.Image) -> str:
    """Convert a 2D attention heatmap into an overlay on the original image."""
    w, h = original_img.size
    if max(w, h) > 512:
        scale = 512.0 / max(w, h)
        w, h = max(int(w * scale), 1), max(int(h * scale), 1)
        orig_small = original_img.resize((w, h), Image.Resampling.BILINEAR).convert("RGB")
    else:
        orig_small = original_img.convert("RGB")

    attn_resized = Image.fromarray((attn_map * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    norm_attn = np.array(attn_resized, dtype=np.float32) / 255.0
    
    # Apply JET or Turbo colormap
    try:
        colormap = matplotlib.colormaps["jet"]
    except Exception:
        colormap = cm.get_cmap("jet")
    heatmap = colormap(norm_attn)[:, :, :3]  # drop alpha
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    
    # Blend with original
    blended = Image.blend(orig_small, heatmap_img, alpha=0.45)
    return pil_to_base64(blended)


class AgricultureDiagnosticPipeline:
    """
    Master Cloud Inference Pipeline implementing the specified flow:
    1. Parallel execution:
       - EfficientNet-B5 + CBAM (Disease Classification)
       - YOLOv8n (Pest Detection)
       - U-Net ResNet-34 (Lesion Segmentation)
    2. Confidence Gate:
       - If confidence >= 0.70 -> Accepted directly + Advisory Engine
       - If confidence <  0.70 -> Smart Fallback to Gemini 2.0 Flash / 1.5 Pro Vision
    """
    def __init__(
        self,
        classifier: DiseaseClassifierInference,
        pest_detector: YOLOv8PestDetector,
        segmenter: LesionSegmenter,
        gemini_fallback: GeminiVisionFallback,
        confidence_threshold: float = 0.70
    ):
        self.classifier = classifier
        self.pest_detector = pest_detector
        self.segmenter = segmenter
        self.gemini_fallback = gemini_fallback
        self.confidence_threshold = confidence_threshold

    def diagnose(self, image: Image.Image, threshold_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute full cloud diagnostic stack on the input leaf image.
        threshold_override is request-scoped (no shared-state mutation, concurrency-safe).
        """
        effective_threshold = self.confidence_threshold
        if threshold_override is not None:
            effective_threshold = float(threshold_override)

        if image.mode != "RGB":
            image = image.convert("RGB")
        image.load()

        # Step 1: Run EfficientNet-B5 + CBAM, YOLOv8n, and U-Net concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_cls = executor.submit(self.classifier.predict, image)
            future_pest = executor.submit(self.pest_detector.detect, image)
            future_seg = executor.submit(self.segmenter.segment, image)

            cls_result = future_cls.result()
            pest_result = future_pest.result()
            seg_result = future_seg.result()

        primary_confidence = cls_result["confidence"]
        primary_disease = cls_result["predicted_class"]
        severity_pct = seg_result["infected_area_pct"]
        pests_list = [d["label"] for d in pest_result["detections"]]

        # Visualizations to Base64
        attention_overlay_b64 = attention_map_to_base64(cls_result["attention_map"], image)
        pest_annotated_b64 = pil_to_base64(pest_result["annotated_image"])
        lesion_overlay_b64 = pil_to_base64(seg_result["overlay_image"])

        # Step 2: Confidence Check Gate
        if primary_confidence >= effective_threshold:
            # High confidence branch -> Accepted
            advisory = AdvisoryEngine.generate_advisory(
                disease_name=primary_disease,
                confidence=primary_confidence,
                severity_pct=severity_pct,
                pest_count=pest_result["pest_count"],
                pests=pests_list
            )
            
            final_response = {
                "decision_path": "PRIMARY_MODEL_ACCEPTED",
                "is_fallback": False,
                "confidence_threshold": effective_threshold,
                "diagnosis": {
                    "disease_name": advisory.get("disease_identified", primary_disease),
                    "raw_disease_name": primary_disease,
                    "crop": advisory.get("crop", "Field Crop"),
                    "category": advisory.get("category", "Foliar Pathology"),
                    "confidence": primary_confidence,
                    "top_candidates": cls_result["top_predictions"],
                    "model_source": "EfficientNet-B5 + CBAM Attention"
                },
                "pests": {
                    "count": pest_result["pest_count"],
                    "detections": pest_result["detections"]
                },
                "segmentation": {
                    "infected_area_pct": severity_pct,
                    "severity_category": seg_result["severity_category"],
                    "leaf_pixels": seg_result["leaf_pixel_count"],
                    "lesion_pixels": seg_result["lesion_pixel_count"]
                },
                "advisory": advisory,
                "visualizations": {
                    "original": pil_to_base64(image),
                    "attention_heatmap": attention_overlay_b64,
                    "pest_detections": pest_annotated_b64,
                    "lesion_segmentation": lesion_overlay_b64
                }
            }
        else:
            # Low confidence branch (< 0.80) -> Call Gemini Vision Smart Fallback
            fallback_result = self.gemini_fallback.call_fallback(
                image=image,
                preliminary_predictions=cls_result["top_predictions"],
                primary_confidence=primary_confidence,
                severity_pct=severity_pct,
                pests_detected=pests_list
            )

            final_response = {
                "decision_path": "GEMINI_VISION_FALLBACK",
                "is_fallback": True,
                "confidence_threshold": effective_threshold,
                "diagnosis": {
                    "disease_name": fallback_result.get("disease_name", primary_disease),
                    "crop": fallback_result.get("crop", "Field Crop"),
                    "confidence": fallback_result.get("confidence", 0.88),
                    "reasoning": fallback_result.get("visual_reasoning", ""),
                    "model_source": fallback_result.get("fallback_source", "Gemini 2.0 Flash / 1.5 Pro Vision"),
                    "primary_model_tentative": primary_disease,
                    "primary_model_confidence": primary_confidence,
                    "top_candidates": cls_result["top_predictions"]
                },
                "pests": {
                    "count": pest_result["pest_count"],
                    "detections": pest_result["detections"]
                },
                "segmentation": {
                    "infected_area_pct": severity_pct,
                    "severity_category": seg_result["severity_category"],
                    "leaf_pixels": seg_result["leaf_pixel_count"],
                    "lesion_pixels": seg_result["lesion_pixel_count"]
                },
                "advisory": {
                    "disease_identified": fallback_result.get("disease_name", primary_disease),
                    "urgency": "Review advised by Smart Fallback",
                    "chemical_control": fallback_result.get("treatment_advice", {}).get("chemical", []),
                    "organic_control": fallback_result.get("treatment_advice", {}).get("organic", []),
                    "cultural_practices": fallback_result.get("treatment_advice", {}).get("cultural", [])
                },
                "visualizations": {
                    "original": pil_to_base64(image),
                    "attention_heatmap": attention_overlay_b64,
                    "pest_detections": pest_annotated_b64,
                    "lesion_segmentation": lesion_overlay_b64
                }
            }

        return final_response
