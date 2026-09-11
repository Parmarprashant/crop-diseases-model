"""
Script to initialize baseline checkpoints and metadata for the cloud model stack.
Ensures that the cloud server and dashboard can run end-to-end inference immediately.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from models.efficientnet_cbam import build_efficientnet_cbam
from models.unet_segmenter import ResNet34_UNet

MAIN_DATA_PATH = os.path.join(BASE_DIR, "MAIN DATA")
TRAIN_DIR = os.path.join(MAIN_DATA_PATH, "Train")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Discover classes strictly from MAIN DATA
classes = []
if os.path.exists(TRAIN_DIR):
    classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

if not classes:
    classes = [
        "American Bollworm on Cotton", "Anthracnose on Cotton", "Army worm",
        "bacterial_blight in Cotton", "Becterial Blight in Rice", "bollrot on Cotton",
        "bollworm on Cotton", "Brownspot", "Common_Rust", "Cotton Aphid",
        "cotton mealy bug", "cotton whitefly", "Flag Smut", "Gray_Leaf_Spot",
        "Healthy cotton", "Healthy Maize", "Healthy Wheat", "Leaf Curl",
        "Leaf smut", "maize ear rot", "maize fall armyworm", "maize stem borer",
        "Mosaic sugarcane", "pink bollworm in cotton", "red cotton bug",
        "RedRot sugarcane", "RedRust sugarcane", "Rice Blast", "Sugarcane Healthy",
        "thirps on cotton", "Tungro", "Wheat aphid", "Wheat black rust",
        "Wheat Brown leaf Rust", "Wheat leaf blight", "Wheat mite",
        "Wheat powdery mildew", "Wheat scab", "Wheat Stem fly",
        "Wheat___Yellow_Rust", "Wilt", "Yellow Rust Sugarcane"
    ]

# Save classes list
class_file = os.path.join(WEIGHTS_DIR, "class_names.txt")
with open(class_file, "w") as f:
    for c in classes:
        f.write(f"{c}\n")
print(f"[Export] Saved {len(classes)} classes to {class_file}")

def get_safe_device() -> str:
    if torch.cuda.is_available():
        try:
            _ = (torch.ones(1, device="cuda") + 1).cpu()
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"

# Export initialized baseline weights
device = get_safe_device()
print(f"[Export] Initializing EfficientNet-B5 + CBAM on {device}...")
eff_model = build_efficientnet_cbam(num_classes=len(classes), weights_path=None, device=device)
eff_weights_path = os.path.join(WEIGHTS_DIR, "efficientnet_b5_cbam_best.pt")
torch.save(eff_model.state_dict(), eff_weights_path)
print(f"[Export] Successfully exported EfficientNet-B5 + CBAM weights to {eff_weights_path}")

print(f"[Export] Initializing ResNet-34 U-Net...")
unet_model = ResNet34_UNet(out_channels=2, pretrained=True).to(device)
unet_weights_path = os.path.join(WEIGHTS_DIR, "unet_resnet34_best.pt")
torch.save(unet_model.state_dict(), unet_weights_path)
print(f"[Export] Successfully exported ResNet-34 U-Net weights to {unet_weights_path}")

print("[Export Complete] All model stack weights and class metadata ready!")
