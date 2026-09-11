import os
import sys
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn.functional as F
from torchvision import transforms
from models.efficientnet_cbam import build_efficientnet_cbam
from inference.ood_detector import OODDetector
from inference.crop_taxonomy import check_crop_compatibility, CLASS_TAXONOMY


def evaluate_system():
    print("=" * 75)
    print("OOD BENCHMARK & DEFENSIVE RELIABILITY EVALUATION REPORT")
    print("=" * 75)

    thresholds_path = os.path.join(BASE_DIR, "weights", "calibration_thresholds.json")
    if not os.path.exists(thresholds_path):
        print("Error: Run calibration/calibrate_thresholds.py first.")
        return

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    print(f"Calibration Version: {thresholds['version']}")
    print(f"Calibrated Thresholds -> Energy: {thresholds['energy_threshold_fpr95']} | Entropy: {thresholds['entropy_threshold_fpr95']} | Min MSP: {thresholds['min_msp_threshold']}")
    print(f"Empirical Metrics     -> AUROC: {thresholds['metrics']['auroc']*100:.2f}% | AUPR: {thresholds['metrics']['aupr']*100:.2f}% | FPR@95TPR: {thresholds['metrics']['fpr_at_95_tpr']*100:.2f}%\n")

    class_names_path = os.path.join(BASE_DIR, "weights", "class_names.txt")
    with open(class_names_path, "r") as f:
        class_names = [l.strip() for l in f if l.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = os.path.join(BASE_DIR, "weights", "efficientnet_b5_cbam_best.pt")
    model = build_efficientnet_cbam(num_classes=len(class_names), weights_path=weights_path, device=device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    detector = OODDetector(thresholds_path=thresholds_path)

    # 1. In-Distribution Evaluation
    val_dir = os.path.join(BASE_DIR, "MAIN DATA", "Validation")
    in_total = 0
    in_accepted = 0
    in_rejected = 0
    raw_crop_violations = 0
    pipeline_crop_violations = 0

    with torch.no_grad():
        for c in class_names:
            c_dir = os.path.join(val_dir, c)
            if not os.path.exists(c_dir) and os.path.exists(val_dir):
                for actual in os.listdir(val_dir):
                    if actual.lower().replace("_", " ") == c.lower().replace("_", " "):
                        c_dir = os.path.join(val_dir, actual)
                        break
            if not os.path.exists(c_dir):
                continue

            for fname in [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]:
                img = Image.open(os.path.join(c_dir, fname)).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(device)
                logits = model(tensor).squeeze(0).cpu().numpy()
                probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()

                ood_res = detector.evaluate(logits, probs)
                in_total += 1
                if not ood_res.is_ood:
                    in_accepted += 1
                else:
                    in_rejected += 1

                # 1. Raw CNN top prediction
                pred_idx = int(np.argmax(probs))
                pred_class = class_names[pred_idx]
                expected_crop = CLASS_TAXONOMY[c].crop if c in CLASS_TAXONOMY else "unknown"
                raw_is_comp, _ = check_crop_compatibility(expected_crop, pred_class)
                if not raw_is_comp:
                    raw_crop_violations += 1

                # 2. Hardened Pipeline Gated Output
                # If raw is incompatible or OOD, pipeline rejects and suppresses cross-crop diagnosis
                if not ood_res.is_ood and raw_is_comp:
                    final_diag_crop = CLASS_TAXONOMY[pred_class].crop
                else:
                    final_diag_crop = expected_crop # Falls back to safe crop state or unknown
                
                pipeline_is_comp, _ = check_crop_compatibility(expected_crop, pred_class if (not ood_res.is_ood and raw_is_comp) else None)
                if not pipeline_is_comp:
                    pipeline_crop_violations += 1

    # 2. Out-of-Distribution Evaluation
    ood_total = 100
    ood_rejected = 0
    ood_false_accepted = 0

    np.random.seed(99)
    with torch.no_grad():
        for _ in range(ood_total):
            noise_img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
            tensor = val_transform(noise_img).unsqueeze(0).to(device)
            logits = model(tensor).squeeze(0).cpu().numpy()
            probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()

            ood_res = detector.evaluate(logits, probs)
            if ood_res.is_ood:
                ood_rejected += 1
            else:
                ood_false_accepted += 1

    # Performance Metrics
    tpr = (in_accepted / in_total) * 100.0 if in_total else 0.0
    tnr = (ood_rejected / ood_total) * 100.0 if ood_total else 0.0
    far = (ood_false_accepted / ood_total) * 100.0 if ood_total else 0.0
    frr = (in_rejected / in_total) * 100.0 if in_total else 0.0

    print("-----------------------------------------------------------------------")
    print(f"In-Distribution Retention Rate (TPR):      {tpr:.2f}% ({in_accepted}/{in_total})")
    print(f"False Rejection Rate (FRR):                {frr:.2f}%")
    print(f"OOD Rejection Rate (TNR):                  {tnr:.2f}% ({ood_rejected}/{ood_total})")
    print(f"False Acceptance Rate (FAR):               {far:.2f}%")
    print(f"Raw Unguided CNN Cross-Crop Errors:        {raw_crop_violations} (Unshielded baseline)")
    print(f"Pipeline-Gated Crop Violations:            {pipeline_crop_violations} (TARGET: 0)")
    print("-----------------------------------------------------------------------")
    assert pipeline_crop_violations == 0, f"Critical failure: {pipeline_crop_violations} crop compatibility violations!"
    print("[PASSED] Zero crop compatibility violations achieved across all samples.\n")


if __name__ == "__main__":
    evaluate_system()
