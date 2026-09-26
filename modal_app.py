import os
import sys
import shutil
from pathlib import Path
import modal

# 1. Initialize Modal App
app = modal.App("agrivision-diagnostic-engine")

# 2. Build Cloud Container Image (Only code + dependencies; weights stream cloud-to-cloud)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cpu"
    )
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "timm>=0.9.0",
        "ultralytics>=8.0.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "opencv-python-headless>=4.7.0",
        "huggingface_hub>=0.20.0",
        "google-generativeai>=0.8.0"
    )
    # Add application code modules (only ~500 KB, uploads in 1 second)
    .add_local_file(".env", remote_path="/root/.env")
    .add_local_dir("api", remote_path="/root/api")
    .add_local_dir("core", remote_path="/root/core")
    .add_local_dir("inference", remote_path="/root/inference")
    .add_local_dir("models", remote_path="/root/models")
)

# Persistent volume for model weights
weights_volume = modal.Volume.from_name("agrivision-weights-vol", create_if_missing=True)

# 3. Define 24/7 Cloud ASGI FastAPI Endpoint
@app.function(
    image=image,
    volumes={"/root/weights": weights_volume},
    secrets=[modal.Secret.from_dotenv()],
    cpu=2.0,
    memory=4096,   # 4 GB RAM in cloud
    timeout=180,
)
@modal.asgi_app()
def fastapi_app():
    os.chdir("/root")
    sys.path.insert(0, "/root")
    
    from dotenv import load_dotenv
    load_dotenv("/root/.env", override=True)

    weights_dir = "/root/weights"
    ckpt = os.path.join(weights_dir, "efficientnet_b5_cbam_best.pt")
    
    # Download weights cloud-to-cloud from Hugging Face if not cached in volume
    if not os.path.exists(ckpt):
        print("[Modal] Initializing weights cloud-to-cloud from Hugging Face...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Blueeee001/Crop-Diseases",
            repo_type="space",
            allow_patterns=["weights/*"],
            local_dir="/root"
        )
        # Ensure yolov8n_pest.pt exists
        if not os.path.exists(os.path.join(weights_dir, "yolov8n_pest.pt")):
            yolo_src = os.path.join(weights_dir, "yolov8n.pt")
            if os.path.exists(yolo_src):
                shutil.copy(yolo_src, os.path.join(weights_dir, "yolov8n_pest.pt"))
        weights_volume.commit()
        print("[Modal] Weights successfully cached in persistent volume!")

    os.environ["DEVICE"] = "cpu"
    os.environ["WEIGHTS_DIR"] = weights_dir
    os.environ["CHECKPOINT_PATH"] = os.path.join(weights_dir, "efficientnet_b5_cbam_best.pt")
    os.environ["CONFIDENCE_THRESHOLD"] = "0.70"
    os.environ["ENABLE_GEMINI_FALLBACK"] = "1"
    os.environ["GEMINI_MODEL"] = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    from api.server import app as web_app
    return web_app
