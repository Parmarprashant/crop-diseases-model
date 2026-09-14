"""
AgriVision AI — Step 7: Freeze Consistency Exponent Alpha on Development Validation Set.

Evaluates alpha in {0.0, 0.25, 0.50, 0.75, 1.00} strictly on val_split.csv:
    S(d) = P(disease = d) * [P_crop(C(d))]^alpha

Computes:
    Score_dev = Acc_disease,dev + 0.5 * Acc_crop,dev - 2.0 * Rate_cross-crop,dev
Selects the alpha that maximizes Score_dev and freezes it into:
    validation/hierarchical_frozen_alpha.json

STRICT SCIENTIFIC RULE:
Alpha is chosen ONLY on val_split.csv.
Alpha is NEVER tuned on Tier 2 (220) or Tier 4 (150).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.hierarchical_pipeline import (
    HierarchicalDiseaseClassifierInference,
    SoftCropDiseaseConsistencyEngine
)
from inference.pipeline import CanonicalAggregator

def main():
    parser = argparse.ArgumentParser(description="Tune alpha on val_split.csv")
    parser.add_argument("--checkpoint", type=str, default="weights/efficientnet_b5_cbam_hierarchical_candidate.pt")
    parser.add_argument("--val_csv", type=str, default="Data/outputs/outputs/val_split.csv")
    parser.add_argument("--output_json", type=str, default="validation/hierarchical_frozen_alpha.json")
    parser.add_argument("--max_samples", type=int, default=1500, help="Representative sample size for alpha tuning")
    args = parser.parse_args()

    print("=" * 80)
    print("      AGRIVISION AI: CONSISTENCY EXPONENT (ALPHA) SELECTION ON DEV SET")
    print("=" * 80)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Validation Split : {args.val_csv}")
    print(f"Candidate Alphas : [0.0, 0.25, 0.50, 0.75, 1.00]\n")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} does not exist yet. Please wait for training to finish.")
        sys.exit(1)

    classifier = HierarchicalDiseaseClassifierInference(checkpoint_path=args.checkpoint)
    aggregator = CanonicalAggregator()

    # Load validation images
    df = pd.read_csv(args.val_csv)
    if len(df) > args.max_samples:
        # Stratified sample across crops
        df_sample = df.groupby("crop", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(args.max_samples * len(x) / len(df)))), random_state=42)
        ).reset_index(drop=True)
    else:
        df_sample = df

    print(f"Loaded {len(df_sample):,} representative samples across {df_sample['crop'].nunique()} crops for alpha tuning.")

    # Master directory lookup
    master_dir = "Data/master_images/master_images/images"
    available_files = {f.lower(): f for f in os.listdir(master_dir)}

    valid_records = []
    print("Running candidate forward passes...")
    t0 = time.time()

    for idx, row in df_sample.iterrows():
        img_id = str(row["image_id"]).lower()
        true_crop = str(row["crop"]).strip().lower()
        true_disease = str(row["canonical_class"]).strip()

        phys_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            cand = f"{img_id}{ext}"
            if cand in available_files:
                phys_file = os.path.join(master_dir, available_files[cand])
                break

        if not phys_file or not os.path.exists(phys_file):
            continue

        try:
            image = Image.open(phys_file).convert("RGB")
        except Exception:
            continue

        res = classifier.predict(image, top_k=10)
        canon_probs_arr, canon_names, _ = aggregator.aggregate_numpy(res["all_probabilities"])
        canonical_probs_dict = {canon_names[i]: float(canon_probs_arr[i]) for i in range(len(canon_names))}
        crop_probs_dict = res.get("all_crop_probabilities", {})

        valid_records.append({
            "image_id": img_id,
            "true_crop": true_crop,
            "true_disease": true_disease,
            "pred_crop_head": res["predicted_crop"],
            "canonical_probs": canonical_probs_dict,
            "crop_probs": crop_probs_dict
        })

        if (len(valid_records)) % 250 == 0:
            print(f"  Processed {len(valid_records):,} samples ({time.time() - t0:.1f}s)...")

    print(f"Completed inference on {len(valid_records):,} samples in {time.time() - t0:.1f}s.\n")

    # Evaluate each alpha candidate
    alphas = [0.0, 0.25, 0.50, 0.75, 1.00]
    alpha_results = []
    best_score = -float("inf")
    best_alpha = 0.5

    with open("weights/canonical_disease_to_crop.json", "r", encoding="utf-8") as f:
        canon_to_crop = json.load(f)

    for alpha in alphas:
        engine = SoftCropDiseaseConsistencyEngine(alpha=alpha)
        correct_disease = 0
        correct_crop = 0
        cross_crop_errors = 0
        total = len(valid_records)

        for r in valid_records:
            t_crop = r["true_crop"]
            t_disease = r["true_disease"]

            cons = engine.apply_consistency(r["canonical_probs"], r["crop_probs"])
            pred_diag = cons["final_prediction"]

            # Predicted crop derived from predicted disease
            pred_disease_crop = canon_to_crop.get(pred_diag, pred_diag.split("-")[0].strip().lower())
            head_crop = r["pred_crop_head"].lower()

            # Crop accuracy (using dedicated crop head)
            crop_match = (head_crop == t_crop) or (t_crop in ["rice", "paddy"] and head_crop in ["rice", "paddy"])
            if crop_match:
                correct_crop += 1

            # Disease accuracy
            d_norm_true = t_disease.lower().replace("-", " ").replace("_", " ").strip()
            d_norm_pred = pred_diag.lower().replace("-", " ").replace("_", " ").strip()
            if d_norm_true == d_norm_pred or (d_norm_pred in d_norm_true and len(d_norm_pred) > 5):
                correct_disease += 1

            # Cross-crop error: predicted disease belongs to a different crop family
            diag_crop_match = (pred_disease_crop == t_crop) or (t_crop in ["rice", "paddy"] and pred_disease_crop in ["rice", "paddy"])
            if not diag_crop_match:
                cross_crop_errors += 1

        acc_disease = (correct_disease / total) * 100.0
        acc_crop = (correct_crop / total) * 100.0
        rate_cross_crop = (cross_crop_errors / total) * 100.0
        score_dev = acc_disease + (0.5 * acc_crop) - (2.0 * rate_cross_crop)

        res_record = {
            "alpha": alpha,
            "acc_disease": round(acc_disease, 2),
            "acc_crop": round(acc_crop, 2),
            "rate_cross_crop": round(rate_cross_crop, 2),
            "score_dev": round(score_dev, 2),
            "total_samples": total
        }
        alpha_results.append(res_record)

        print(f"Alpha = {alpha:4.2f} | Disease Acc: {acc_disease:5.2f}% | Crop Acc: {acc_crop:5.2f}% | Cross-Crop: {rate_cross_crop:5.2f}% | Score_dev: {score_dev:6.2f}")

        if score_dev > best_score:
            best_score = score_dev
            best_alpha = alpha

    print("=" * 80)
    print(f" >>> OPTIMAL FROZEN ALPHA: {best_alpha:4.2f} (Score_dev: {best_score:6.2f})")
    print("=" * 80)

    frozen_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_alpha": best_alpha,
        "best_score_dev": best_score,
        "evaluation_split": args.val_csv,
        "samples_evaluated": len(valid_records),
        "selection_formula": "Score_dev = Acc_disease + 0.5*Acc_crop - 2.0*Rate_cross_crop",
        "results": alpha_results
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(frozen_config, f, indent=2)
    print(f"Saved frozen alpha configuration to: {args.output_json}")

if __name__ == "__main__":
    main()
