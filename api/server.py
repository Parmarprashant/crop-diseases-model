import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

import io
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse
from models.efficientnet_cbam import build_efficientnet_cbam, DiseaseClassifierInference
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.gemini_fallback import GeminiVisionFallback
from inference.pipeline import HierarchicalAgriDiagnosticPipeline
from inference.diagnosis_schema import StandardizedDiagnosisResponse
from inference.crop_taxonomy import CLASS_TAXONOMY

# Server startup metadata for traceability
STARTUP_TIMESTAMP = datetime.utcnow().isoformat() + "Z"

# Discover classes strictly from weights/class_names.txt or MAIN DATA/Train
MAIN_DATA_PATH = os.environ.get("MAIN_DATA_PATH", "MAIN DATA")
TRAIN_DIR = os.path.join(MAIN_DATA_PATH, "Train")
CLASS_NAMES_FILE = os.path.join("weights", "class_names.txt")

DEFAULT_CLASSES: List[str] = []
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, "r") as f:
        DEFAULT_CLASSES = [line.strip() for line in f if line.strip()]
elif os.path.exists(TRAIN_DIR):
    DEFAULT_CLASSES = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])

# Setup configuration
def get_safe_device() -> str:
    env_device = os.environ.get("DEVICE", "").lower()
    if env_device in ["cpu", "cuda"]:
        return env_device
    return "cpu"

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
DEVICE = get_safe_device()
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "efficientnet_b5_cbam_best.pt")
THRESHOLDS_PATH = os.path.join(WEIGHTS_DIR, "calibration_thresholds.json")

# Model File Hash & Mtime calculation
CHECKPOINT_HASH = "unavailable"
CHECKPOINT_MTIME = "unavailable"
if os.path.exists(CHECKPOINT_PATH):
    try:
        mtime = os.path.getmtime(CHECKPOINT_PATH)
        CHECKPOINT_MTIME = datetime.utcfromtimestamp(mtime).isoformat() + "Z"
        h = hashlib.sha256()
        with open(CHECKPOINT_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        CHECKPOINT_HASH = h.hexdigest()[:16]
    except Exception as e:
        print(f"[Server] Checkpoint hash calculation notice: {e}")

CLASS_MANIFEST_HASH = "unavailable"
if os.path.exists(CLASS_NAMES_FILE):
    try:
        with open(CLASS_NAMES_FILE, "rb") as f:
            CLASS_MANIFEST_HASH = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        pass

# Instantiate models with calibrated 256px resolution
print(f"[Server] Initializing models on device: {DEVICE} (Checkpoint: {CHECKPOINT_PATH})...")

classifier_model = build_efficientnet_cbam(
    num_classes=len(DEFAULT_CLASSES),
    weights_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None,
    device=DEVICE
)
classifier_inf = DiseaseClassifierInference(classifier_model, DEFAULT_CLASSES, device=DEVICE, img_size=256)

pest_detector = YOLOv8PestDetector(
    weights_path=os.path.join(WEIGHTS_DIR, "yolov8n_pest.pt") if os.path.exists(os.path.join(WEIGHTS_DIR, "yolov8n_pest.pt")) else None,
    device="cpu"
)

lesion_segmenter = LesionSegmenter(
    weights_path=os.path.join(WEIGHTS_DIR, "unet_resnet34_best.pt") if os.path.exists(os.path.join(WEIGHTS_DIR, "unet_resnet34_best.pt")) else None,
    device=DEVICE
)

gemini_fallback = GeminiVisionFallback(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
)

# Instantiate defensive hierarchical pipeline
pipeline = HierarchicalAgriDiagnosticPipeline(
    classifier=classifier_inf,
    pest_detector=pest_detector,
    segmenter=lesion_segmenter,
    gemini_fallback=gemini_fallback,
    thresholds_path=THRESHOLDS_PATH
)

# Setup FastAPI App
app = FastAPI(
    title="AgriVision Cloud AI - Multi-Model Diagnostic Server",
    description="Hierarchical Agricultural Disease Diagnosis Pipeline with OOD & Compatibility Gating.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ONLINE",
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        models_loaded={
            "EfficientNet-B5+CBAM": classifier_model is not None,
            "YOLOv8n_Pest": pest_detector is not None,
            "ResNet34_UNet": lesion_segmenter is not None,
            "Gemini_Vision_Fallback": True
        },
        confidence_threshold=pipeline.ood_detector.min_msp_threshold
    )


@app.get("/api/v1/model-info")
def get_model_info():
    """Diagnostic endpoint exposing loaded model, hashes, versions, and calibration status."""
    calib_data = {}
    if os.path.exists(THRESHOLDS_PATH):
        try:
            with open(THRESHOLDS_PATH) as f:
                calib_data = json.load(f)
        except Exception:
            pass

    return {
        "system_name": "AgriVision Cloud AI",
        "system_version": "2.1.0-hierarchical-defensive",
        "server_start_timestamp": STARTUP_TIMESTAMP,
        "primary_model": {
            "architecture": "EfficientNet-B5 + CBAM Attention",
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_mtime": CHECKPOINT_MTIME,
            "checkpoint_sha256_short": CHECKPOINT_HASH,
            "input_resolution": "256x256",
            "device": DEVICE,
            "class_count": len(DEFAULT_CLASSES),
            "class_manifest_hash": CLASS_MANIFEST_HASH
        },
        "secondary_models": {
            "pest_detector": "YOLOv8n Pest Intelligence",
            "lesion_segmenter": "U-Net (ResNet-34 Encoder)",
            "multimodal_fallback": "Gemini 2.0 Flash Vision"
        },
        "ood_calibration": {
            "version": calib_data.get("version", "default"),
            "energy_threshold_fpr95": calib_data.get("energy_threshold_fpr95", -5.327),
            "entropy_threshold_fpr95": calib_data.get("entropy_threshold_fpr95", 0.466),
            "min_msp_threshold": calib_data.get("min_msp_threshold", 0.486),
            "auroc": calib_data.get("metrics", {}).get("auroc", 0.9802),
            "fpr_at_95_tpr": calib_data.get("metrics", {}).get("fpr_at_95_tpr", 0.0542)
        },
        "advisory_database_version": "2.1.0-with-rice-panicle-safety"
    }


@app.get("/api/v1/classes")
def get_supported_classes():
    return {
        "dataset_source": "MAIN DATA/Train",
        "total_classes": len(DEFAULT_CLASSES),
        "classes": DEFAULT_CLASSES
    }


@app.post("/api/v1/diagnose", response_model=StandardizedDiagnosisResponse)
async def run_diagnosis(
    file: UploadFile = File(...),
    crop_hint: Optional[str] = Query(None, description="Optional manual crop specification from farmer")
):
    """
    Primary Cloud Inference Endpoint:
    Executes hierarchical diagnosis pipeline:
    Quality Gating -> Crop Identification -> Plant Part Detection -> Domain Compatibility ->
    EfficientNet-B5+CBAM -> Energy OOD Gating -> Safe Fallback.
    """
    filename_lower = (file.filename or "").lower()
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.jfif', '.heic')
    is_valid_type = (
        (file.content_type and file.content_type.startswith("image/")) or
        filename_lower.endswith(valid_exts)
    )
    if not is_valid_type:
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG, PNG, WEBP, etc.)")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    try:
        result = pipeline.diagnose(image, user_crop_hint=crop_hint)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis pipeline error: {str(e)}")


@app.get("/")
def root():
    """
    Cloud Model API Gateway Discovery Endpoint
    """
    return {
        "service": "AgriVision Cloud Model Inference Engine",
        "version": "2.0.0",
        "status": "ONLINE",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "diagnose": "POST /api/v1/diagnose",
            "health": "GET /api/v1/health",
            "config": "GET /api/v1/config"
        },
        "device": DEVICE,
        "models": {
            "classifier": "EfficientNet-B5 + CBAM (42 classes)",
            "lesion_segmenter": "ResNet-34 U-Net",
            "pest_detector": "YOLOv8n (14 pest classes)",
            "secondary_vision": "Gemini 2.5 Flash"
        }
    }

