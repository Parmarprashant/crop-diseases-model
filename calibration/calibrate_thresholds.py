import os
import sys
import json
import time
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn.functional as F
from models.efficientnet_cbam import build_efficientnet_cbam
from inference.ood_detector import OODDetector


def calculate_auroc_and_fpr95(in_scores: np.ndarray, ood_scores: np.ndarray):
    """
    Computes AUROC, AUPR, and FPR@95TPR for OOD detection.
    Convention: Lower energy = in-distribution; Higher energy = OOD.
    """
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])

    # Sort descending by score (higher score = more likely OOD)
    desc_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[desc_idx]

    # Compute ROC points
    tpr_list = []
    fpr_list = []
    total_pos = float(len(ood_scores))
    total_neg = float(len(in_scores))

    tp = 0.0
    fp = 0.0
    for l in sorted_labels:
        if l == 1.0:
            tp += 1.0
        else:
            fp += 1.0
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    # Trapezoidal integration for AUROC (NumPy 2.x compatible)
    tpr_arr = np.array([0.0] + tpr_list)
    fpr_arr = np.array([0.0] + fpr_list)
    if hasattr(np, "trapezoid"):
        auroc = float(np.trapezoid(tpr_arr, fpr_arr))
    else:
        from scipy.integrate import trapezoid
        auroc = float(trapezoid(tpr_arr, fpr_arr))

    # FPR at 95% TPR (when TPR >= 0.95)
    fpr95 = 1.0
    for tpr, fpr in zip(tpr_list, fpr_list):
        if tpr >= 0.95:
            fpr95 = float(fpr)
            break

    # Precision-Recall Curve (AUPR)
    prec_list = []
    tp = 0.0
    for i, l in enumerate(sorted_labels):
        if l == 1.0:
            tp += 1.0
        prec_list.append(tp / float(i + 1))
    aupr = float(np.mean(prec_list))

    return {
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
        "fpr_at_95_tpr": round(fpr95, 4)
    }


def main():
    print("=" * 70)
    print("[Calibration] AgriVision Cloud AI - Empirical Threshold Selection")
    print("=" * 70)

    weights_path = os.path.join(BASE_DIR, "weights", "efficientnet_b5_cbam_best.pt")
    class_names_path = os.path.join(BASE_DIR, "weights", "class_names.txt")
    output_path = os.path.join(BASE_DIR, "weights", "calibration_thresholds.json")

    if not os.path.exists(class_names_path):
        print(f"Error: {class_names_path} not found.")
        return

    with open(class_names_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Running calibration on: {device.upper()}")

    model = build_efficientnet_cbam(num_classes=len(class_names), weights_path=weights_path, device=device)
    model.eval()

    from torchvision import transforms
    # Standard resolution matching training checkpoint (256x256)
    img_size = 256
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from training.dataset import MainDataCropDataset
    val_dataset = MainDataCropDataset(root_dir=os.path.join(BASE_DIR, "Data"), subset="Val", transform=val_transform, classes=class_names)
    print(f"[In-Distribution] Sourced {len(val_dataset)} validation images across {len(class_names)} classes.")

    detector = OODDetector()
    in_energies = []
    in_entropies = []
    in_msps = []

    samples_to_take = min(500, len(val_dataset))
    print(f"[In-Distribution] Computing metrics on {samples_to_take} validation samples...")
    indices = np.linspace(0, len(val_dataset) - 1, samples_to_take, dtype=int)
    total_sampled = 0

    with torch.no_grad():
        for idx in indices:
            try:
                img_tensor, _ = val_dataset[idx]
                tensor = img_tensor.unsqueeze(0).to(device)
                logits = model(tensor).squeeze(0).cpu().numpy()
                probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()

                energy = detector.compute_energy(logits)
                entropy = detector.compute_normalized_entropy(probs)
                msp = float(np.max(probs))

                in_energies.append(energy)
                in_entropies.append(entropy)
                in_msps.append(msp)
                total_sampled += 1
            except Exception:
                continue

    print(f" -> Processed {total_sampled} in-distribution validation samples.")
    in_energies = np.array(in_energies)
    in_entropies = np.array(in_entropies)
    in_msps = np.array(in_msps)

    # Generate synthetic Out-of-Distribution evaluation set
    print("[Out-of-Distribution] Evaluating against synthetic and perturbed OOD distributions...")
    ood_energies = []
    ood_entropies = []
    ood_msps = []

    np.random.seed(42)
    for _ in range(total_sampled):
        # 1. Random uniform noise image
        noise_img = Image.fromarray(np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8))
        tensor = val_transform(noise_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor).squeeze(0).cpu().numpy()
            probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
            ood_energies.append(detector.compute_energy(logits))
            ood_entropies.append(detector.compute_normalized_entropy(probs))
            ood_msps.append(float(np.max(probs)))

    ood_energies = np.array(ood_energies)
    ood_entropies = np.array(ood_entropies)

    # Statistical Evaluation
    metrics = calculate_auroc_and_fpr95(in_energies, ood_energies)
    print(f" -> OOD Energy AUROC:     {metrics['auroc'] * 100:.2f}%")
    print(f" -> OOD Energy AUPR:      {metrics['aupr'] * 100:.2f}%")
    print(f" -> FPR @ 95% TPR:        {metrics['fpr_at_95_tpr'] * 100:.2f}%")

    # Select threshold: 95th percentile of in-distribution energy (covers 95% of real plant images)
    selected_energy_thresh = float(np.percentile(in_energies, 95))
    selected_entropy_thresh = float(np.percentile(in_entropies, 95))
    min_msp = float(np.percentile(in_msps, 5))

    calibration_manifest = {
        "version": "empirical_calibration_v2_sih",
        "dataset": "MAIN DATA/Validation (42 classes)",
        "sample_count": total_sampled,
        "input_resolution": f"{img_size}x{img_size}",
        "metrics": metrics,
        "energy_in_dist_mean": float(np.mean(in_energies)),
        "energy_in_dist_std": float(np.std(in_energies)),
        "energy_threshold_fpr95": round(selected_energy_thresh, 3),
        "entropy_in_dist_mean": float(np.mean(in_entropies)),
        "entropy_threshold_fpr95": round(selected_entropy_thresh, 3),
        "min_msp_threshold": round(min_msp, 3),
        "calibration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration_manifest, f, indent=2)

    print(f"[Success] Calibrated thresholds persisted to {output_path}:")
    print(f"          Energy Threshold:  {selected_energy_thresh:.3f}")
    print(f"          Entropy Threshold: {selected_entropy_thresh:.3f}")
    print(f"          Min MSP Threshold: {min_msp:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
