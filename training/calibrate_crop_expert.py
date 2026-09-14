"""
AgriVision AI — Tier 1 Temperature-Scaled Calibration for Dedicated Crop Expert.

Calibrates CropExpertModel (ConvNeXt-Tiny) strictly on Data/outputs/outputs/val_split.csv.
Finds optimal T_crop > 0 minimizing NLL and Expected Calibration Error (ECE).
Freezes T_crop into validation/crop_disease_resolver_config.json.
"""
import os
import sys
import json
import time
import numpy as np
from typing import Tuple, List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.optimize import minimize_scalar

from models.crop_expert import build_crop_expert
from training.crop_expert_dataset import CropExpertDataset

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.sum(in_bin) / total_samples

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence - avg_accuracy)

    return float(ece)

def nll_criterion(logits: torch.Tensor, labels: torch.Tensor, temp: float) -> float:
    scaled_logits = logits / max(temp, 1e-4)
    loss = F.cross_entropy(scaled_logits, labels).item()
    return loss

def main():
    print("=" * 80)
    print("   AGRIVISION AI: TIER 1 CROP EXPERT TEMPERATURE CALIBRATION")
    print("=" * 80)
    
    val_csv = "Data/outputs/outputs/val_split.csv"
    crop_names_path = "weights/crop_names.txt"
    weights_path = "weights/crop_expert_candidate.pt"
    config_path = "validation/crop_disease_resolver_config.json"

    with open(crop_names_path, "r", encoding="utf-8") as f:
        crop_names = [l.strip().lower() for l in f if l.strip()]
    num_crops = len(crop_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Checkpoint: {weights_path} | Crops: {num_crops}")

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = CropExpertDataset(
        csv_path=val_csv,
        crop_names_path=crop_names_path,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = build_crop_expert(
        weights_path=weights_path,
        num_classes=num_crops,
        backbone_name="convnext_tiny",
        device=device
    )
    model.eval()

    print("\nCollecting validation logits and targets across 6,670 samples...")
    all_logits = []
    all_targets = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(images)
            all_logits.append(logits.float().cpu())
            all_targets.append(targets)

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  Processed {min((batch_idx+1)*64, len(val_dataset))}/{len(val_dataset)} samples...")

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    elapsed = time.time() - t0
    print(f"Logits collection complete in {elapsed:.1f}s. Shape: {logits_tensor.shape}")

    # Compute uncalibrated metrics (T = 1.0)
    uncal_probs = F.softmax(logits_tensor, dim=1).numpy()
    targets_np = targets_tensor.numpy()

    uncal_nll = float(F.cross_entropy(logits_tensor, targets_tensor).item())
    uncal_ece = compute_ece(uncal_probs, targets_np)
    uncal_acc = float(np.mean(np.argmax(uncal_probs, axis=1) == targets_np) * 100.0)

    print(f"\n--- UNCALIBRATED CROP EXPERT (T = 1.000) ---")
    print(f"  Top-1 Accuracy : {uncal_acc:.2f}%")
    print(f"  NLL Loss       : {uncal_nll:.4f}")
    print(f"  ECE (15 bins)  : {uncal_ece:.4f}")

    # Optimize Temperature T > 0
    print("\nOptimizing Temperature Scaling parameter T_crop...")
    res = minimize_scalar(
        lambda t: nll_criterion(logits_tensor, targets_tensor, t),
        bounds=(0.1, 5.0),
        method="bounded"
    )
    optimal_t = float(res.x)

    # Compute calibrated metrics
    cal_logits = logits_tensor / optimal_t
    cal_probs = F.softmax(cal_logits, dim=1).numpy()
    cal_nll = float(F.cross_entropy(cal_logits, targets_tensor).item())
    cal_ece = compute_ece(cal_probs, targets_np)
    cal_acc = float(np.mean(np.argmax(cal_probs, axis=1) == targets_np) * 100.0)

    print(f"\n--- CALIBRATED CROP EXPERT (T = {optimal_t:.4f}) ---")
    print(f"  Top-1 Accuracy : {cal_acc:.2f}% (Invariant preserved)")
    print(f"  NLL Loss       : {cal_nll:.4f} (Delta: {cal_nll - uncal_nll:.4f})")
    print(f"  ECE (15 bins)  : {cal_ece:.4f} (Delta: {cal_ece - uncal_ece:.4f}, {((cal_ece - uncal_ece)/uncal_ece)*100:.1f}%)")

    # Load existing config or create new
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    config["calibration"] = {
        "dataset": val_csv,
        "sample_count": len(val_dataset),
        "temperature": round(optimal_t, 4),
        "uncalibrated_nll": round(uncal_nll, 4),
        "calibrated_nll": round(cal_nll, 4),
        "uncalibrated_ece": round(uncal_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "crop_accuracy": round(cal_acc, 2),
        "status": "FROZEN"
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved calibration configuration to: {config_path}")

    # Also save cached val logits for fast resolver tuning
    cache_path = "validation/val_crop_logits_cache.pt"
    torch.save({
        "logits": logits_tensor,
        "targets": targets_tensor,
        "samples": val_dataset.samples,
        "crop_strings": val_dataset.crop_strings,
        "temperature": optimal_t
    }, cache_path)
    print(f"Saved cached validation logits to: {cache_path}")

if __name__ == "__main__":
    main()
