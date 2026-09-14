"""
AgriVision AI — Tier 1 Three-Arm Control Ablation.

Evaluates:
  Arm 1 (Control A): Model A Baseline alone
  Arm 2 (Control B): Model A + Dedicated Crop Expert (Advisory-Only)
  Arm 3 (Candidate C): Model A + Dedicated Crop Expert + Crop-Aware Disease Resolver (Mode B) + Calibrator

Saves ablation results to validation/tier1_control_ablation_results.json.
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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

def compute_ece(confidences: List[float], accuracies: List[float], n_bins: int = 15) -> float:
    if not confidences or not accuracies:
        return 0.0
    confs = np.array(confidences)
    accs = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confs)

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        in_bin = (confs > b_low) & (confs <= b_high) if i > 0 else (confs >= b_low) & (confs <= b_high)
        prop = np.sum(in_bin) / total
        if prop > 0:
            ece += prop * np.abs(np.mean(confs[in_bin]) - np.mean(accs[in_bin]))
    return float(ece)

def main():
    print("=" * 80)
    print("   AGRIVISION AI: TIER 1 THREE-ARM CONTROL ABLATION")
    print("=" * 80)

    # 1. Load Tier 1 dataset and predictions
    val_csv = "Data/outputs/outputs/val_split.csv"
    master_images_dir = "Data/master_images/master_images/images"
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
    strat_samples = []
    for crop_name, group in df_valid.groupby("crop_true"):
        n_take = min(len(group), 10)
        strat_samples.append(group.sample(n=n_take, random_state=42))
    df_eval = pd.concat(strat_samples).reset_index(drop=True)

    # Load frozen configuration
    frozen_cfg_path = "validation/final_candidate_frozen_config.json"
    with open(frozen_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    t_crop = cfg["crop_expert"]["calibration_temperature"]
    t_res = cfg["confidence_calibrator"]["temperature"]

    # We need predictions for all 3 arms.
    # To get exact Model A and Crop Expert outputs, let's load or compute
    import torch
    from PIL import Image
    from models.efficientnet_cbam import build_efficientnet_cbam, DiseaseClassifierInference
    from models.yolo_pest import YOLOv8PestDetector
    from models.unet_segmenter import LesionSegmenter
    from core.gemini_fallback import GeminiVisionFallback
    from inference.pipeline import HierarchicalAgriDiagnosticPipeline
    from inference.crop_fusion import CalibratedCropExpertInference

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

    # Initialize Resolvers
    calibrator = FinalConfidenceCalibrator(temperature=t_res)

    resolver_advisory = CropAwareDiseaseResolver(mode="mode_a")
    resolver_candidate = CropAwareDiseaseResolver(
        mode="mode_b",
        tau_crop_conf=cfg["resolver"]["tau_crop_conf"],
        tau_compat_mass=cfg["resolver"]["tau_compat_mass"],
        tau_within_crop_dom=cfg["resolver"]["tau_within_crop_dom"],
        tau_single_compat=cfg["resolver"]["tau_single_compat"],
        tau_crop_margin=cfg["resolver"]["tau_crop_margin"],
        tau_crop_entropy=cfg["resolver"]["tau_crop_entropy"],
        tau_disease_conf=cfg["resolver"]["tau_disease_conf"],
        calibrator=calibrator
    )

    print(f"Executing 3-arm ablation across {len(df_eval)} Tier 1 samples...")
    t0 = time.time()

    records_a = []
    records_b = []
    records_c = []

    for idx, row in df_eval.iterrows():
        img = Image.open(row["file_path"]).convert("RGB")
        res_a = base_pipeline.diagnose(img)
        crop_out = crop_expert.predict(img)

        crop_dist = crop_out.distribution
        sorted_crops = sorted(crop_dist.items(), key=lambda x: x[1], reverse=True)
        top1_crop = crop_out.crop
        top2_crop = sorted_crops[1][0] if len(sorted_crops) > 1 else "unknown"

        canonical_probs = getattr(base_pipeline, "last_canonical_probs", {})
        if not canonical_probs and res_a.primary_model.top_candidates:
            canonical_probs = {c["class_name"]: c["confidence"] for c in res_a.primary_model.top_candidates}

        # Arm 1: Model A baseline
        records_a.append({
            "crop_pred": res_a.crop.name,
            "diag_pred": res_a.diagnosis.name,
            "conf": res_a.primary_model.raw_confidence or 0.0,
            "accepted": res_a.primary_model.accepted,
            "crop_true": row["crop_true"],
            "disease_true": row["disease_true"]
        })

        # Arm 2: Advisory (Control B)
        res_b = resolver_advisory.resolve(
            model_a_crop=res_a.crop.name,
            model_a_diag=res_a.diagnosis.name,
            model_a_conf=res_a.primary_model.raw_confidence or 0.0,
            model_a_accepted=res_a.primary_model.accepted,
            canonical_disease_probs=canonical_probs,
            crop_top1=top1_crop,
            crop_top2=top2_crop,
            crop_conf=crop_out.confidence,
            crop_margin=crop_out.margin,
            crop_entropy=crop_out.entropy,
            crop_probs=crop_dist
        )
        records_b.append({
            "crop_pred": res_b.final_crop,
            "diag_pred": res_b.final_disease,
            "conf": res_b.final_confidence,
            "accepted": res_b.final_accepted,
            "crop_true": row["crop_true"],
            "disease_true": row["disease_true"],
            "state": res_b.telemetry.resolver_state
        })

        # Arm 3: Candidate C (Mode B Resolver)
        res_c = resolver_candidate.resolve(
            model_a_crop=res_a.crop.name,
            model_a_diag=res_a.diagnosis.name,
            model_a_conf=res_a.primary_model.raw_confidence or 0.0,
            model_a_accepted=res_a.primary_model.accepted,
            canonical_disease_probs=canonical_probs,
            crop_top1=top1_crop,
            crop_top2=top2_crop,
            crop_conf=crop_out.confidence,
            crop_margin=crop_out.margin,
            crop_entropy=crop_out.entropy,
            crop_probs=crop_dist
        )
        records_c.append({
            "crop_pred": res_c.final_crop,
            "diag_pred": res_c.final_disease,
            "conf": res_c.final_confidence,
            "accepted": res_c.final_accepted,
            "crop_true": row["crop_true"],
            "disease_true": row["disease_true"],
            "state": res_c.telemetry.resolver_state
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_eval):
            print(f"  Processed {idx + 1}/{len(df_eval)} samples in {time.time() - t0:.1f}s...")

    def eval_arm(records: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        n = len(records)
        n_crop_corr = 0
        n_full_diag_corr = 0
        n_cond_diag_corr = 0
        n_cross_crop = 0
        n_abstained = 0
        confs = []
        full_corrs = []

        for r in records:
            c_true = r["crop_true"]
            d_true = r["disease_true"]
            c_pred = r["crop_pred"]
            d_pred = r["diag_pred"]
            accepted = r["accepted"]
            conf = r["conf"]

            c_corr = resolver_candidate.are_crops_compatible(c_pred, c_true)
            diag_corr = match_disease(c_true, d_true, d_pred)

            if not accepted:
                n_abstained += 1

            if c_corr:
                n_crop_corr += 1
                if diag_corr and accepted:
                    n_cond_diag_corr += 1
            else:
                if accepted and c_pred != "unknown":
                    n_cross_crop += 1

            is_full = (c_corr and diag_corr and accepted)
            if is_full:
                n_full_diag_corr += 1

            confs.append(conf)
            full_corrs.append(1.0 if is_full else 0.0)

        crop_acc = round(n_crop_corr / n * 100, 2)
        full_diag_acc = round(n_full_diag_corr / n * 100, 2)
        cond_diag_acc = round(n_cond_diag_corr / max(n_crop_corr, 1) * 100, 2)
        cross_crop_rate = round(n_cross_crop / n * 100, 2)
        abstain_rate = round(n_abstained / n * 100, 2)
        ece = round(compute_ece(confs, full_corrs, 15), 4)

        return {
            "name": name,
            "sample_count": n,
            "crop_accuracy": crop_acc,
            "full_diagnosis_accuracy": full_diag_acc,
            "conditional_diagnosis_accuracy": cond_diag_acc,
            "cross_crop_errors": n_cross_crop,
            "cross_crop_rate": cross_crop_rate,
            "abstention_rate": abstain_rate,
            "ece": ece
        }

    res_arm_a = eval_arm(records_a, "Control A: Model A Baseline")
    res_arm_b = eval_arm(records_b, "Control B: Model A + Crop Expert Advisory")
    res_arm_c = eval_arm(records_c, "Candidate C: Model A + Crop Expert + Mode B Resolver (Calibrated)")

    print("\n" + "=" * 80)
    print("   TIER 1 THREE-ARM CONTROL ABLATION RESULTS")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Control A (Model A)':<20} | {'Control B (Advisory)':<20} | {'Candidate C (Resolver)':<22}")
    print("-" * 100)
    print(f"{'Top-1 Crop Accuracy':<32} | {res_arm_a['crop_accuracy']:>18.2f}% | {res_arm_b['crop_accuracy']:>18.2f}% | {res_arm_c['crop_accuracy']:>20.2f}%")
    print(f"{'Full Diagnosis Accuracy':<32} | {res_arm_a['full_diagnosis_accuracy']:>18.2f}% | {res_arm_b['full_diagnosis_accuracy']:>18.2f}% | {res_arm_c['full_diagnosis_accuracy']:>20.2f}%")
    print(f"{'Conditional Diagnosis Acc':<32} | {res_arm_a['conditional_diagnosis_accuracy']:>18.2f}% | {res_arm_b['conditional_diagnosis_accuracy']:>18.2f}% | {res_arm_c['conditional_diagnosis_accuracy']:>20.2f}%")
    print(f"{'Cross-Crop Errors':<32} | {res_arm_a['cross_crop_errors']:>15} ({res_arm_a['cross_crop_rate']}%) | {res_arm_b['cross_crop_errors']:>15} ({res_arm_b['cross_crop_rate']}%) | {res_arm_c['cross_crop_errors']:>17} ({res_arm_c['cross_crop_rate']}%)")
    print(f"{'Abstention / Refusal Rate':<32} | {res_arm_a['abstention_rate']:>18.2f}% | {res_arm_b['abstention_rate']:>18.2f}% | {res_arm_c['abstention_rate']:>20.2f}%")
    print(f"{'Expected Calibration Error (ECE)':<32} | {res_arm_a['ece']:>18.4f}  | {res_arm_b['ece']:>18.4f}  | {res_arm_c['ece']:>20.4f} ")
    print("=" * 100)

    ablation_out = {
        "control_a": res_arm_a,
        "control_b": res_arm_b,
        "candidate_c": res_arm_c,
        "config_sha256": cfg["status"]
    }

    out_file = "validation/tier1_control_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_out, f, indent=2)
    print(f"Saved ablation summary to {out_file}")

if __name__ == "__main__":
    main()
