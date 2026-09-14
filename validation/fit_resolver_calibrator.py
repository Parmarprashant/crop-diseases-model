"""
AgriVision AI — Tier 1 Resolver Confidence Calibration & Optimization.

Fits the monotonic scalar temperature calibrator T_resolver on Tier 1
validation predictions (Data/outputs/outputs/val_split.csv).
Fitted strictly prior to unsealing Tier 4 / Tier 5 evaluation.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from models.efficientnet_cbam import build_efficientnet_cbam, DiseaseClassifierInference
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.gemini_fallback import GeminiVisionFallback
from inference.pipeline import HierarchicalAgriDiagnosticPipeline
from inference.crop_fusion import CalibratedCropExpertInference
from inference.crop_disease_resolver import CropAwareDiseaseResolver
from inference.final_confidence_calibrator import FinalConfidenceCalibrator

def normalize_disease_str(s: str) -> str:
    clean = s.lower().replace("-", " ").replace("_", " ")
    clean = clean.replace("leafcurl", "leaf curl")
    return " ".join(clean.split())

def match_disease(crop_true: str, disease_true: str, diag_pred: str) -> bool:
    d_true_norm = normalize_disease_str(disease_true)
    d_pred_norm = normalize_disease_str(diag_pred)
    if "healthy" in d_true_norm:
        return "healthy" in d_pred_norm
    if "healthy" in d_pred_norm and "healthy" not in d_true_norm:
        return False
    if d_pred_norm in d_true_norm or d_true_norm in d_pred_norm:
        return True
    core_pred_words = [w for w in d_pred_norm.split() if w not in [crop_true, "leaf", "spot", "disease", "rot", "blight", "virus"]]
    if core_pred_words and any(w in d_true_norm for w in core_pred_words):
        return True
    return False

def main():
    print("=" * 80)
    print("   AGRIVISION AI: FITTING TIER 1 RESOLVER CONFIDENCE CALIBRATOR")
    print("=" * 80)

    val_csv = "Data/outputs/outputs/val_split.csv"
    master_images_dir = "Data/master_images/master_images/images"
    prior_config_path = "validation/crop_disease_resolver_config.json"

    with open(prior_config_path, "r", encoding="utf-8") as f:
        prior_cfg = json.load(f)
    t_crop = prior_cfg["calibration"]["temperature"]
    print(f"Loaded frozen T_crop = {t_crop:.4f}")

    df_val = pd.read_csv(val_csv)
    available_files = {f.lower(): f for f in os.listdir(master_images_dir)}

    valid_rows = []
    for idx, row in df_val.iterrows():
        img_id = str(row["image_id"]).lower()
        for ext in [".jpg", ".jpeg", ".png"]:
            if f"{img_id}{ext}" in available_files:
                phys_path = os.path.join(master_images_dir, available_files[f"{img_id}{ext}"])
                valid_rows.append({
                    "image_id": img_id,
                    "file_path": phys_path,
                    "crop_true": str(row["crop"]).lower().strip(),
                    "disease_true": str(row["canonical_class"]).strip()
                })
                break

    df_valid = pd.DataFrame(valid_rows)
    # Stratified cohort: up to 10 per crop across all 41 crops
    strat_samples = []
    for crop_name, group in df_valid.groupby("crop_true"):
        n_take = min(len(group), 10)
        strat_samples.append(group.sample(n=n_take, random_state=42))
    df_eval = pd.concat(strat_samples).reset_index(drop=True)
    print(f"Evaluating {len(df_eval)} stratified Tier 1 development samples...")

    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    with open("weights/class_names.txt", "r") as f:
        class_names = [l.strip() for l in f if l.strip()]

    model_a_net = build_efficientnet_cbam(num_classes=len(class_names), weights_path="weights/efficientnet_b5_cbam_best.pt", device=dev_str)
    model_a_cls = DiseaseClassifierInference(model=model_a_net, class_names=class_names, device=dev_str)
    pest_det = YOLOv8PestDetector()
    seg = LesionSegmenter()
    gemini = GeminiVisionFallback()

    base_pipeline = HierarchicalAgriDiagnosticPipeline(
        classifier=model_a_cls,
        pest_detector=pest_det,
        segmenter=seg,
        gemini_fallback=gemini,
        enable_gemini=False
    )

    crop_expert = CalibratedCropExpertInference(
        checkpoint_path="weights/crop_expert_candidate.pt",
        crop_names_path="weights/crop_names.txt",
        device=dev_str,
        temperature=t_crop
    )

    # Initialize uncalibrated Mode B resolver
    resolver = CropAwareDiseaseResolver(
        mode="mode_b",
        tau_crop_conf=0.45,
        tau_compat_mass=0.08,
        tau_within_crop_dom=0.25,
        tau_single_compat=0.05,
        tau_crop_margin=0.08,
        tau_crop_entropy=0.85,
        tau_disease_conf=0.35,
        calibrator=None # uncalibrated for fitting
    )

    cache_file = "validation/tier1_calibration_cache.npz"
    if os.path.exists(cache_file):
        print(f"Loading cached Tier 1 predictions from {cache_file}...")
        data = np.load(cache_file)
        raw_confs = data["raw_confs"]
        binary_correctness = data["binary_correctness"]
    else:
        print("Extracting predictions on Tier 1 development cohort...")
        raw_confs = []
        binary_correctness = []

        t0 = time.time()
        for idx, row in df_eval.iterrows():
            img = Image.open(row["file_path"]).convert("RGB")
            res = base_pipeline.diagnose(img)
            crop_out = crop_expert.predict(img)

            crop_dist = crop_out.distribution
            sorted_crops = sorted(crop_dist.items(), key=lambda x: x[1], reverse=True)
            top1_crop = crop_out.crop
            top2_crop = sorted_crops[1][0] if len(sorted_crops) > 1 else "unknown"

            canonical_probs = getattr(base_pipeline, "last_canonical_probs", {})
            if not canonical_probs and res.primary_model.top_candidates:
                canonical_probs = {c["class_name"]: c["confidence"] for c in res.primary_model.top_candidates}

            resolver_res = resolver.resolve(
                model_a_crop=res.crop.name,
                model_a_diag=res.diagnosis.name,
                model_a_conf=res.primary_model.raw_confidence or 0.0,
                model_a_accepted=res.primary_model.accepted,
                canonical_disease_probs=canonical_probs,
                crop_top1=top1_crop,
                crop_top2=top2_crop,
                crop_conf=crop_out.confidence,
                crop_margin=crop_out.margin,
                crop_entropy=crop_out.entropy,
                crop_probs=crop_dist
            )

            c_true = row["crop_true"]
            d_true = row["disease_true"]

            crop_corr = resolver.are_crops_compatible(resolver_res.final_crop, c_true)
            diag_corr = match_disease(c_true, d_true, resolver_res.final_disease)
            is_full_correct = 1.0 if (crop_corr and diag_corr and resolver_res.final_accepted) else 0.0

            raw_confs.append(resolver_res.final_confidence)
            binary_correctness.append(is_full_correct)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(df_eval):
                print(f"  Processed {idx + 1}/{len(df_eval)} in {time.time() - t0:.1f}s...")

        raw_confs = np.array(raw_confs)
        binary_correctness = np.array(binary_correctness)
        np.savez(cache_file, raw_confs=raw_confs, binary_correctness=binary_correctness)
        print(f"Saved prediction cache to {cache_file}")

    # Fit temperature calibrator for minimum ECE
    calibrator = FinalConfidenceCalibrator()
    metrics = calibrator.fit(raw_confs, binary_correctness, objective="ece")

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS (Tier 1 Development Cohort)")
    print("=" * 60)
    print(f"  Optimal Temperature T_resolver: {metrics['optimal_temperature']:.4f}")
    print(f"  Uncalibrated ECE:              {metrics['uncalibrated_ece']:.4f}")
    print(f"  Calibrated ECE:                {metrics['calibrated_ece']:.4f} (Reduction: {metrics['ece_reduction_pct']:.1f}%)")
    print(f"  Uncalibrated Brier Score:      {metrics['uncalibrated_brier']:.4f}")
    print(f"  Calibrated Brier Score:        {metrics['calibrated_brier']:.4f}")
    print(f"  Uncalibrated NLL:              {metrics['uncalibrated_nll']:.4f}")
    print(f"  Calibrated NLL:                {metrics['calibrated_nll']:.4f}")
    print("=" * 60)

    # Save fitted calibration parameters
    calibration_summary = {
        "dataset": val_csv,
        "sample_count": len(df_eval),
        "temperature": metrics["optimal_temperature"],
        "uncalibrated_ece": metrics["uncalibrated_ece"],
        "calibrated_ece": metrics["calibrated_ece"],
        "uncalibrated_brier": metrics["uncalibrated_brier"],
        "calibrated_brier": metrics["calibrated_brier"],
        "uncalibrated_nll": metrics["uncalibrated_nll"],
        "calibrated_nll": metrics["calibrated_nll"],
        "status": "FROZEN"
    }

    out_path = "validation/resolver_calibration_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calibration_summary, f, indent=2)
    print(f"\nCalibration summary saved to {out_path}")

if __name__ == "__main__":
    main()
