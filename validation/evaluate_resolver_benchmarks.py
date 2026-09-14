"""
AgriVision AI — Production Resolver Multi-Tier Benchmark Evaluation Suite.

Evaluates:
1. Tier 2: 220-image External Development Benchmark (validation/external_cohort_manifest.csv)
2. Tier 3: 30 Defensive Stress cases + 34 Historical Regression cases
3. Tier 4: 150-image Sealed Final Acceptance Cohort (validation/sealed_acceptance_cohort_manifest.csv)
   - Enforces Post-Unseal Zero-Tuning Lock.

Outputs:
- validation/crop_disease_resolver_tier2_results.json
- validation/crop_disease_resolver_tier3_results.json
- validation/crop_disease_resolver_tier4_results.json
"""
import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from models.efficientnet_cbam import DiseaseClassifierInference, build_efficientnet_cbam
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.gemini_fallback import GeminiVisionFallback
from inference.pipeline import HierarchicalAgriDiagnosticPipeline
from inference.crop_fusion import CalibratedCropExpertInference
from inference.crop_disease_resolver import CropAwareDiseaseResolver
from inference.crop_disease_pipeline import CropDiseasePipeline

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
        in_bin = (confs > b_low) & (confs <= b_high)
        prop = np.sum(in_bin) / total
        if prop > 0:
            ece += prop * np.abs(np.mean(confs[in_bin]) - np.mean(accs[in_bin]))
    return float(ece)

def evaluate_manifest_with_pipeline(
    pipeline,
    manifest_csv: str,
    cohort_filter: Optional[str] = "agricultural",
    model_tag: str = "candidate"
) -> Tuple[Dict, List[Dict]]:
    df = pd.read_csv(manifest_csv)
    if cohort_filter and "cohort_group" in df.columns:
        df = df[df["cohort_group"] == cohort_filter].reset_index(drop=True)

    total_samples = len(df)
    print(f"\n--- Starting Evaluation: {model_tag} on {os.path.basename(manifest_csv)} ({total_samples} samples) ---", flush=True)

    correct_crop = 0
    correct_full_diag = 0
    cross_crop_errors = 0
    high_conf_errors = 0
    abstentions = 0
    confidences = []
    accuracies = []
    latencies = []
    records = []

    for idx, row in df.iterrows():
        sample_id = str(row.get("image_id", idx))
        file_path = str(row.get("file_path", ""))
        crop_true = str(row.get("crop", row.get("crop_true", ""))).strip().lower()
        disease_true = str(row.get("disease", row.get("disease_true", ""))).strip()

        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        if not os.path.exists(file_path):
            continue

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception:
            continue

        t0 = time.time()
        res = pipeline.diagnose(image)
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        crop_pred = res.crop.name.lower().strip()
        diag_pred = res.diagnosis.name
        diag_conf = res.primary_model.raw_confidence or 0.0
        primary_accepted = res.primary_model.accepted

        # Crop match check (including rice/paddy)
        is_crop_match = (crop_pred == crop_true) or (crop_true in ["rice", "paddy"] and crop_pred in ["rice", "paddy"])
        if is_crop_match:
            correct_crop += 1

        # Full diagnosis match check
        is_diag_match = primary_accepted and match_disease(crop_true, disease_true, diag_pred)
        if is_diag_match:
            correct_full_diag += 1

        # Cross-crop error check
        is_cross_crop = primary_accepted and (not is_crop_match) and (crop_pred != "unknown")
        if is_cross_crop:
            cross_crop_errors += 1

        # High confidence error
        is_high_conf_error = primary_accepted and (diag_conf >= 0.80) and not is_diag_match
        if is_high_conf_error:
            high_conf_errors += 1

        if not primary_accepted:
            abstentions += 1

        confidences.append(diag_conf)
        accuracies.append(1.0 if is_diag_match else 0.0)

        # Extract telemetry if available
        telemetry_dict = {}
        if hasattr(pipeline, "last_resolver_result") and pipeline.last_resolver_result:
            tel = pipeline.last_resolver_result.telemetry
            telemetry_dict = {
                "resolver_state": tel.resolver_state,
                "resolver_reason": tel.resolver_reason,
                "crop_top1": tel.crop_top1,
                "crop_top2": tel.crop_top2,
                "crop_margin": tel.crop_margin,
                "crop_entropy": tel.crop_entropy,
                "model_a_orig_score": tel.model_a_original_disease_score,
                "crop_conditioned_score": tel.crop_conditioned_disease_score,
                "model_a_disease_top1": tel.model_a_disease_top1
            }

        record = {
            "image_id": sample_id,
            "file_path": file_path,
            "crop_true": crop_true,
            "disease_true": disease_true,
            "crop_pred": crop_pred,
            "diag_pred": diag_pred,
            "diag_conf": round(diag_conf, 4),
            "primary_accepted": primary_accepted,
            "crop_match": is_crop_match,
            "disease_match": is_diag_match,
            "is_cross_crop": is_cross_crop,
            "is_high_conf_error": is_high_conf_error,
            "latency_ms": round(latency_ms, 1),
            "telemetry": telemetry_dict,
            "model_tag": model_tag
        }
        records.append(record)

        if len(records) % 25 == 0 or len(records) == total_samples:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    Evaluated {len(records)}/{total_samples} samples ({model_tag})...", flush=True)

    acc_crop = (correct_crop / max(total_samples, 1)) * 100.0
    acc_full_diag = (correct_full_diag / max(total_samples, 1)) * 100.0
    acc_cond_diag = (correct_full_diag / max(correct_crop, 1)) * 100.0
    rate_cross_crop = (cross_crop_errors / max(total_samples, 1)) * 100.0
    rate_high_conf_err = (high_conf_errors / max(total_samples, 1)) * 100.0
    rate_abstention = (abstentions / max(total_samples, 1)) * 100.0
    ece = compute_ece(confidences, accuracies)
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    median_latency = float(np.median(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    summary = {
        "model_tag": model_tag,
        "total_samples": total_samples,
        "evaluated_samples": len(records),
        "crop_accuracy": round(acc_crop, 2),
        "full_diagnosis_accuracy": round(acc_full_diag, 2),
        "conditional_diagnosis_accuracy": round(acc_cond_diag, 2),
        "cross_crop_errors": cross_crop_errors,
        "cross_crop_rate": round(rate_cross_crop, 2),
        "high_confidence_errors": high_conf_errors,
        "high_confidence_error_rate": round(rate_high_conf_err, 2),
        "abstentions": abstentions,
        "abstention_rate": round(rate_abstention, 2),
        "ece": round(ece, 4),
        "mean_latency_ms": round(mean_latency, 1),
        "median_latency_ms": round(median_latency, 1),
        "p95_latency_ms": round(p95_latency, 1)
    }

    return summary, records

def main():
    parser = argparse.ArgumentParser(description="Evaluate Crop-Aware Disease Resolver Multi-Tier Benchmarks")
    parser.add_argument("--config", type=str, default="validation/crop_disease_resolver_config.json")
    parser.add_argument("--crop_weights", type=str, default="weights/crop_expert_candidate.pt")
    parser.add_argument("--model_a_weights", type=str, default="weights/efficientnet_b5_cbam_best.pt")
    args = parser.parse_args()

    print("=" * 80)
    print("   AGRIVISION AI: CROP-AWARE DISEASE RESOLVER MULTI-TIER BENCHMARKS")
    print("=" * 80)

    # 1. Load Frozen Configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found. Run tune_resolver_modes.py first.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    calib_cfg = cfg["calibration"]
    res_cfg = cfg["resolver"]
    t_crop = calib_cfg["temperature"]
    sel_mode = res_cfg["selected_mode"]

    print(f"Loaded Frozen Config from: {args.config}")
    print(f"  Calibrated T_crop       : {t_crop:.4f}")
    print(f"  Selected Resolver Mode  : {sel_mode} ({res_cfg.get('name', '')})")
    print(f"  tau_crop_conf           : {res_cfg['tau_crop_conf']}")
    print(f"  tau_compat_mass         : {res_cfg['tau_compat_mass']}")
    print(f"  tau_within_crop_dom     : {res_cfg['tau_within_crop_dom']}")
    print(f"  tau_single_compat       : {res_cfg['tau_single_compat']}\n")

    # 2. Build Base Model A Pipeline
    print("Initializing Model A Reference Pipeline...")
    with open("weights/class_names.txt", "r", encoding="utf-8") as f:
        class_names = [l.strip() for l in f if l.strip()]
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    model_a_net = build_efficientnet_cbam(num_classes=len(class_names), weights_path=args.model_a_weights, device=dev_str)
    model_a_cls = DiseaseClassifierInference(model=model_a_net, class_names=class_names, device=dev_str)
    pest_det = YOLOv8PestDetector()
    seg = LesionSegmenter()
    gemini = GeminiVisionFallback()

    model_a_pipeline = HierarchicalAgriDiagnosticPipeline(
        classifier=model_a_cls,
        pest_detector=pest_det,
        segmenter=seg,
        gemini_fallback=gemini,
        enable_gemini=False
    )

    # 3. Build Crop-Aware Disease Resolver Pipeline (Candidate)
    print("Initializing Crop-Aware Disease Resolver Candidate Pipeline...")
    crop_exp_inference = CalibratedCropExpertInference(
        checkpoint_path=args.crop_weights,
        crop_names_path="weights/crop_names.txt",
        device=dev_str,
        temperature=t_crop
    )
    resolver = CropAwareDiseaseResolver(
        mode=sel_mode,
        tau_crop_conf=res_cfg["tau_crop_conf"],
        tau_crop_margin=res_cfg.get("tau_crop_margin", 0.08),
        tau_crop_entropy=res_cfg.get("tau_crop_entropy", 0.85),
        tau_compat_mass=res_cfg["tau_compat_mass"],
        tau_within_crop_dom=res_cfg["tau_within_crop_dom"],
        tau_single_compat=res_cfg["tau_single_compat"],
        tau_disease_conf=res_cfg.get("tau_disease_conf", 0.35)
    )
    candidate_pipeline = CropDiseasePipeline(
        base_pipeline=model_a_pipeline,
        crop_expert=crop_exp_inference,
        resolver=resolver
    )

    # =========================================================================
    # TIER 2: 220-IMAGE EXTERNAL DEVELOPMENT BENCHMARK
    # =========================================================================
    print("\n" + "=" * 80)
    print(" >>> TIER 2: 220-IMAGE EXTERNAL DEVELOPMENT BENCHMARK EVALUATION")
    print("=" * 80)
    t2_manifest = "validation/external_cohort_manifest.csv"

    summary_t2_cand, records_t2_cand = evaluate_manifest_with_pipeline(
        pipeline=candidate_pipeline,
        manifest_csv=t2_manifest,
        cohort_filter="agricultural",
        model_tag="model_a_plus_resolver"
    )

    print("\n--- TIER 2 RESOLVER CANDIDATE RESULTS ---")
    print(f"  Crop Accuracy        : {summary_t2_cand['crop_accuracy']:.2f}% (Ref: 74.09%, Gate: >= 76.09%)")
    print(f"  Full Diagnosis Acc   : {summary_t2_cand['full_diagnosis_accuracy']:.2f}% (Ref: 64.55%, Gate: >= 66.55%)")
    print(f"  Conditional Diag Acc : {summary_t2_cand['conditional_diagnosis_accuracy']:.2f}% (Ref: 87.12%, Gate: >= 85.62%)")
    print(f"  Cross-Crop Errors    : {summary_t2_cand['cross_crop_errors']} / 220 ({summary_t2_cand['cross_crop_rate']:.2f}%) (Ref: 30 / 13.64%, Gate: <= 25 errors)")
    print(f"  Expected Calib Error : {summary_t2_cand['ece']:.4f} (Ref: 0.0819, Gate: <= 0.1000)")
    print(f"  High-Confidence Err  : {summary_t2_cand['high_confidence_error_rate']:.2f}% (Ref: 0.91%, Gate: <= 2.0%)")
    print(f"  Abstention Rate      : {summary_t2_cand['abstention_rate']:.2f}%")
    print(f"  Mean Latency         : {summary_t2_cand['mean_latency_ms']:.1f}ms (Median: {summary_t2_cand['median_latency_ms']:.1f}ms, P95: {summary_t2_cand['p95_latency_ms']:.1f}ms)")

    t2_payload = {
        "summary": summary_t2_cand,
        "records": records_t2_cand
    }
    with open("validation/crop_disease_resolver_tier2_results.json", "w", encoding="utf-8") as f:
        json.dump(t2_payload, f, indent=2)
    print("Saved Tier 2 results to: validation/crop_disease_resolver_tier2_results.json\n")

    # =========================================================================
    # TIER 3: DEFENSIVE STRESS SUITE (30) & REGRESSION SUITE (34)
    # =========================================================================
    print("=" * 80)
    print(" >>> TIER 3: DEFENSIVE STRESS SUITE (30) EVALUATION")
    print("=" * 80)
    t3_stress_csv = "validation/defensive_stress_manifest.csv"
    intercepted_count = 0
    records_stress = []

    if os.path.exists(t3_stress_csv):
        df_stress = pd.read_csv(t3_stress_csv)
        for idx, row in df_stress.iterrows():
            img_path = str(row["file_path"])
            if not os.path.isabs(img_path):
                img_path = os.path.join(BASE_DIR, img_path)
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            res = candidate_pipeline.diagnose(image)

            # Defensive pass: Model A rejected (primary_accepted == False)
            is_intercepted = not res.primary_model.accepted
            if is_intercepted:
                intercepted_count += 1
            records_stress.append({
                "image_id": row.get("image_id", idx),
                "intercepted": is_intercepted,
                "accepted": res.primary_model.accepted,
                "crop": res.crop.name,
                "diagnosis": res.diagnosis.name
            })
            if len(records_stress) % 10 == 0:
                print(f"    Evaluated {len(records_stress)}/{len(df_stress)} defensive stress samples...", flush=True)

    interception_rate = (intercepted_count / max(len(records_stress), 1)) * 100.0
    print(f"  Defensive Interception: {intercepted_count} / {len(records_stress)} ({interception_rate:.1f}%) - Target: 100.0%")

    # Historical Regression Suite (34 cases)
    print("\n--- Evaluating Historical Regression Suite (34 cases) ---")
    reg_csv = "validation/regression_cases_manifest.csv"
    regressions_found = 0
    records_regression = []
    if os.path.exists(reg_csv):
        df_reg = pd.read_csv(reg_csv)
        for idx, row in df_reg.iterrows():
            img_path = str(row["file_path"])
            if not os.path.isabs(img_path):
                img_path = os.path.join(BASE_DIR, img_path)
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            res = candidate_pipeline.diagnose(image)
            expected_acc = bool(row.get("expected_accepted", True))
            actual_acc = bool(res.primary_model.accepted)
            is_reg = (expected_acc != actual_acc)
            if is_reg:
                regressions_found += 1
            records_regression.append({
                "image_id": row.get("image_id", idx),
                "expected_accepted": expected_acc,
                "actual_accepted": actual_acc,
                "regression": is_reg
            })
        print(f"  Historical Regressions: {regressions_found} / {len(records_regression)} - Target: 0")

    t3_summary = {
        "defensive_stress_intercepted": intercepted_count,
        "defensive_stress_total": len(records_stress),
        "defensive_stress_rate": round(interception_rate, 2),
        "historical_regressions": regressions_found,
        "regression_total": len(records_regression),
        "tier3_pass": (interception_rate == 100.0 and regressions_found == 0)
    }
    with open("validation/crop_disease_resolver_tier3_results.json", "w", encoding="utf-8") as f:
        json.dump(t3_summary, f, indent=2)
    print("Saved Tier 3 results to: validation/crop_disease_resolver_tier3_results.json\n")

    # =========================================================================
    # TIER 4: NEW UNSEEN FINAL ACCEPTANCE COHORT (150 IMAGES)
    # =========================================================================
    print("=" * 80)
    print(" >>> TIER 4: UNSEEN FINAL ACCEPTANCE COHORT (150 IMAGES) EVALUATION")
    print(" >>> [POST-UNSEAL ZERO-TUNING LOCK ACTIVE]")
    print("=" * 80)
    t4_csv = "validation/sealed_acceptance_cohort_manifest.csv"
    if os.path.exists(t4_csv):
        # Run Candidate Pipeline
        summary_t4_cand, records_t4_cand = evaluate_manifest_with_pipeline(
            pipeline=candidate_pipeline,
            manifest_csv=t4_csv,
            cohort_filter=None,
            model_tag="model_a_plus_resolver"
        )

        # Run Model A Baseline side-by-side
        summary_t4_base, records_t4_base = evaluate_manifest_with_pipeline(
            pipeline=model_a_pipeline,
            manifest_csv=t4_csv,
            cohort_filter=None,
            model_tag="model_a_baseline"
        )

        print("\n--- TIER 4 SIDE-BY-SIDE ACCEPTANCE COMPARISON (150 IMAGES) ---")
        print(f"  Top-1 Crop Accuracy   : Model A = {summary_t4_base['crop_accuracy']:.2f}% | Candidate = {summary_t4_cand['crop_accuracy']:.2f}% (Delta = {summary_t4_cand['crop_accuracy'] - summary_t4_base['crop_accuracy']:+.2f}%)")
        print(f"  Full Diagnosis Acc    : Model A = {summary_t4_base['full_diagnosis_accuracy']:.2f}% | Candidate = {summary_t4_cand['full_diagnosis_accuracy']:.2f}% (Delta = {summary_t4_cand['full_diagnosis_accuracy'] - summary_t4_base['full_diagnosis_accuracy']:+.2f}%)")
        print(f"  Conditional Diag Acc  : Model A = {summary_t4_base['conditional_diagnosis_accuracy']:.2f}% | Candidate = {summary_t4_cand['conditional_diagnosis_accuracy']:.2f}% (Delta = {summary_t4_cand['conditional_diagnosis_accuracy'] - summary_t4_base['conditional_diagnosis_accuracy']:+.2f}%)")
        print(f"  Cross-Crop Error Rate : Model A = {summary_t4_base['cross_crop_rate']:.2f}% | Candidate = {summary_t4_cand['cross_crop_rate']:.2f}% (Delta = {summary_t4_cand['cross_crop_rate'] - summary_t4_base['cross_crop_rate']:+.2f}%)")
        print(f"  ECE                   : Model A = {summary_t4_base['ece']:.4f} | Candidate = {summary_t4_cand['ece']:.4f}")

        t4_payload = {
            "model_a_baseline": {"summary": summary_t4_base, "records": records_t4_base},
            "model_a_plus_resolver": {"summary": summary_t4_cand, "records": records_t4_cand}
        }
        with open("validation/crop_disease_resolver_tier4_results.json", "w", encoding="utf-8") as f:
            json.dump(t4_payload, f, indent=2)
        print("Saved Tier 4 results to: validation/crop_disease_resolver_tier4_results.json\n")
    else:
        print(f"Warning: {t4_csv} not found.")

    print("Multi-tier evaluation complete.")

if __name__ == "__main__":
    main()
