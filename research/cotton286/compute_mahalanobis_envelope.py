"""
AgriVision Ultra v5.0 — Compute Mahalanobis Distance Envelope.

Extracts 512-dimensional bottleneck features across 6 morphological families:
    Class 0: Cotton (Malvaceae)
    Class 1: Monocots / Cereals (Rice, Wheat, Maize)
    Class 2: Legumes / Pulses (Soybean, Chickpea, Pigeonpea)
    Class 3: Solanaceous (Tomato, Potato, Chili)
    Class 4: Broadleaf Hard Negatives (Sunflower, Okra, Castor, Weeds)
    Class 5: Non-Crop Background (Soil, Hands, Plastic, Debris)

Computes:
    mu_c: Class-conditional mean vectors (6 x 512)
    Sigma: Pooled covariance matrix (512 x 512)
    tau_mahalanobis: 99th empirical percentile distance threshold
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import DedicatedCropRouter, ROUTER_FAMILIES


def collect_samples_by_family():
    samples_by_family = {i: [] for i in range(6)}

    # Family 0: Cotton
    manifest_path = "research/cotton286/cotton_split_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            train_items = data.get("splits", {}).get("train", [])
            dev_items = data.get("splits", {}).get("dev", [])
            for item in train_items[:100] + dev_items:
                p = item["file_path"]
                if os.path.exists(p):
                    samples_by_family[0].append(p)

    # Field cohort for Family 1 (Monocots), Family 2 (Legumes), Family 3 (Solanaceous)
    field_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/field_test_cohort"
    if os.path.exists(field_dir):
        for f in os.listdir(field_dir):
            p = os.path.join(field_dir, f)
            fl = f.lower()
            if any(k in fl for k in ["rice", "wheat", "maize", "corn"]):
                samples_by_family[1].append(p)
            elif any(k in fl for k in ["soybean", "pulse", "chickpea", "pigeonpea", "groundnut", "bean", "pea"]):
                samples_by_family[2].append(p)
            elif any(k in fl for k in ["tomato", "potato", "chili", "chilli", "pepper", "eggplant"]):
                samples_by_family[3].append(p)
            else:
                # Distribute other field crops
                samples_by_family[1].append(p)

    # Family 4: Broadleaf negatives (sunflower)
    sunflower_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/sunflower_holdout"
    if os.path.exists(sunflower_dir):
        for f in os.listdir(sunflower_dir):
            p = os.path.join(sunflower_dir, f)
            samples_by_family[4].append(p)

    # Family 5: Non-crop background & OOD
    ood_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/open_set_v2_cohort"
    if os.path.exists(ood_dir):
        for f in os.listdir(ood_dir):
            p = os.path.join(ood_dir, f)
            samples_by_family[5].append(p)

    return samples_by_family


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Mahalanobis] Running on device: {device}")

    router = DedicatedCropRouter(
        checkpoint_path="weights/research/crop_router_v4.pt",
        device=device
    )
    router.model.eval()

    samples_by_family = collect_samples_by_family()
    for fam_id, paths in samples_by_family.items():
        print(f"Family {fam_id} ({ROUTER_FAMILIES.get(fam_id)}): {len(paths)} samples")

    features_by_family = {i: [] for i in range(6)}

    with torch.no_grad():
        for fam_id, paths in samples_by_family.items():
            for p in paths:
                try:
                    with Image.open(p) as img:
                        img = img.convert("RGB")
                    tensor = router.transform(img).unsqueeze(0).to(device)
                    feat = router.model.extract_features(tensor) # (1, 512)
                    features_by_family[fam_id].append(feat.cpu().numpy().squeeze(0))
                except Exception as e:
                    continue

    # Compute class-conditional means mu_c
    means = {}
    all_centered_features = []
    
    for fam_id in range(6):
        feats = np.array(features_by_family[fam_id])
        if len(feats) == 0:
            # Fallback mean if empty
            mu_c = np.zeros(512, dtype=np.float32)
        else:
            mu_c = np.mean(feats, axis=0)
            centered = feats - mu_c
            all_centered_features.append(centered)
        means[fam_id] = mu_c.tolist()

    # Compute pooled shared covariance Sigma
    all_centered = np.vstack(all_centered_features)
    N = all_centered.shape[0]
    # Sigma = (1 / N) * sum (z_i - mu_c)(z_i - mu_c)^T
    cov = np.cov(all_centered, rowvar=False, bias=True) # (512, 512)
    
    # Regularize covariance matrix to ensure stable inversion: Sigma + eps * I
    eps = 1e-4
    cov_reg = cov + eps * np.eye(512, dtype=np.float32)
    inv_cov = np.linalg.pinv(cov_reg)

    # Compute in-distribution Mahalanobis distances to establish 99th percentile threshold
    mahalanobis_distances = []
    means_arr = np.array([means[i] for i in range(6)]) # (6, 512)

    for fam_id in range(6):
        feats = np.array(features_by_family[fam_id])
        if len(feats) == 0:
            continue
        for feat in feats:
            # Calculate distance to all class centers and take minimum
            diffs = feat - means_arr # (6, 512)
            # D_M^2 = diff @ inv_cov @ diff.T
            # For each class:
            d_sq = np.sum((diffs @ inv_cov) * diffs, axis=1) # (6,)
            d_m = np.sqrt(np.maximum(0.0, np.min(d_sq)))
            mahalanobis_distances.append(float(d_m))

    mahalanobis_distances = np.array(mahalanobis_distances)
    tau_mahalanobis = float(np.percentile(mahalanobis_distances, 99.0))
    mean_dist = float(np.mean(mahalanobis_distances))
    max_dist = float(np.max(mahalanobis_distances))

    print(f"[Mahalanobis] N={N} samples, Mean Dist={mean_dist:.3f}, Max Dist={max_dist:.3f}")
    print(f"[Mahalanobis] Calibrated 99th percentile threshold tau_mahalanobis = {tau_mahalanobis:.3f}")

    # Output JSON configuration
    envelope_data = {
        "feature_dim": 512,
        "num_families": 6,
        "tau_mahalanobis": round(tau_mahalanobis, 4),
        "tau_free_energy": -1.20,
        "mean_distances": {
            str(i): round(float(np.mean([np.linalg.norm(np.array(means[i]))])), 4)
            for i in range(6)
        },
        "percentile_99": round(tau_mahalanobis, 4),
        "means": means,
        "inv_cov_diag": np.diag(inv_cov).tolist() # Store diagonal for compact fast fallback
    }

    output_json = "weights/ultra_v5/mahalanobis_envelope.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(envelope_data, f)
    print(f"[Mahalanobis] Saved envelope parameters to {output_json}")

    # Also save full precision precision matrix (inverse covariance) as numpy binary for full inference
    np.save("weights/ultra_v5/inv_covariance.npy", inv_cov.astype(np.float32))
    np.save("weights/ultra_v5/family_means.npy", means_arr.astype(np.float32))
    print("[Mahalanobis] Saved full precision matrices: inv_covariance.npy, family_means.npy")


if __name__ == "__main__":
    main()
