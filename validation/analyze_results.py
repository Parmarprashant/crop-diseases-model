"""
AgriVision AI: Statistical Analysis & Forensics Engine for External Validation.
Computes 95% Wilson Score Confidence Intervals, ECE, Confusion Matrices,
Defensive Rejection Metrics, Failure Categorization, and Target Pair Analyses.
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "validation/external_validation_results.json"
OUTPUT_ANALYSIS_PATH = sys.argv[2] if len(sys.argv) > 2 else "validation/external_validation_analysis.json"

def wilson_ci(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054 # 95% normal quantile
    denominator = 1 + z**2 / n
    center = (p + (z**2) / (2 * n)) / denominator
    spread = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denominator
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(p, 4), round(lower, 4), round(upper, 4)

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
        
    true_tokens = [w for w in d_true_norm.split() if len(w) > 3 and w not in ["crop", crop_true.lower()]]
    pred_tokens = [w for w in d_pred_norm.split() if len(w) > 3]
    return any(t in pred_tokens for t in true_tokens) if true_tokens else False

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: {RESULTS_PATH} not found.")
        sys.exit(1)
        
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    agri_records = [r for r in records if r.get("cohort_group") == "agricultural"]
    def_records = [r for r in records if r.get("cohort_group") == "defensive"]
    
    print(f"Loaded {len(records)} total records: {len(agri_records)} agricultural, {len(def_records)} defensive.")
    
    # Process agricultural cohort
    for r in agri_records:
        c_true = r["crop_true"].lower()
        c_pred = r["crop_pred"].lower()
        
        crop_match = (c_pred == c_true) or (c_true in ["rice", "paddy"] and c_pred in ["rice", "paddy"])
        disease_match = match_disease(c_true, r["disease_true"], r["diag_pred"])
        
        r["crop_match"] = crop_match
        r["disease_match"] = disease_match
        
        if crop_match and disease_match:
            r["verdict"] = "PASS"
        elif crop_match and not disease_match:
            r["verdict"] = "PARTIAL"
        elif not crop_match and c_pred == "unknown":
            r["verdict"] = "REFUSAL"
        else:
            r["verdict"] = "CROSS_CROP"
            
    # Process defensive cohort
    for r in def_records:
        is_defended = (
            r["crop_pred"] == "unknown" or 
            not r["primary_accepted"] or 
            r["diag_status"] in ["INSUFFICIENT_EVIDENCE", "UNKNOWN"] or
            r["diag_pred"] in ["Unreadable Image", "Unable to determine the disease reliably"]
        )
        r["defended"] = is_defended
        r["verdict"] = "DEFENDED_PASS" if is_defended else "DEFENDED_FAIL"

    n_agri = len(agri_records)
    
    # 1. Primary Metrics + 95% Wilson CI
    crop_correct = sum(1 for r in agri_records if r["crop_match"])
    crop_acc, crop_ci_low, crop_ci_high = wilson_ci(crop_correct, n_agri)
    
    diag_correct = sum(1 for r in agri_records if r["verdict"] == "PASS")
    diag_pass, diag_ci_low, diag_ci_high = wilson_ci(diag_correct, n_agri)
    
    partial_count = sum(1 for r in agri_records if r["verdict"] == "PARTIAL")
    partial_rate, partial_ci_low, partial_ci_high = wilson_ci(partial_count, n_agri)
    
    cross_crop_count = sum(1 for r in agri_records if r["verdict"] == "CROSS_CROP")
    cross_crop_rate, cross_ci_low, cross_ci_high = wilson_ci(cross_crop_count, n_agri)
    
    refusal_count = sum(1 for r in agri_records if r["verdict"] == "REFUSAL")
    refusal_rate, refusal_ci_low, refusal_ci_high = wilson_ci(refusal_count, n_agri)
    
    # Conditional Diagnosis Accuracy (given correct crop)
    cond_diag_pass, cond_ci_low, cond_ci_high = wilson_ci(diag_correct, crop_correct) if crop_correct > 0 else (0,0,0)
    
    # 2. Per-Crop Breakdown
    crops = sorted(list(set(r["crop_true"] for r in agri_records)))
    per_crop_stats = {}
    for c in crops:
        c_recs = [r for r in agri_records if r["crop_true"] == c]
        c_n = len(c_recs)
        c_crop_corr = sum(1 for r in c_recs if r["crop_match"])
        c_diag_corr = sum(1 for r in c_recs if r["verdict"] == "PASS")
        c_part = sum(1 for r in c_recs if r["verdict"] == "PARTIAL")
        c_cross = sum(1 for r in c_recs if r["verdict"] == "CROSS_CROP")
        c_refuse = sum(1 for r in c_recs if r["verdict"] == "REFUSAL")
        
        per_crop_stats[c] = {
            "total": c_n,
            "crop_accuracy": round(c_crop_corr / c_n * 100, 1),
            "full_diag_rate": round(c_diag_corr / c_n * 100, 1),
            "partial_rate": round(c_part / c_n * 100, 1),
            "cross_crop_errors": c_cross,
            "refusals": c_refuse
        }
        
    macro_crop_acc = np.mean([stats["crop_accuracy"] for stats in per_crop_stats.values()])
    macro_diag_pass = np.mean([stats["full_diag_rate"] for stats in per_crop_stats.values()])
    
    # 3. Full Crop Confusion Matrix
    unique_pred_crops = sorted(list(set([r["crop_true"] for r in agri_records] + [r["crop_pred"] for r in agri_records])))
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for r in agri_records:
        true_c = r["crop_true"]
        pred_c = r["crop_pred"]
        if true_c in ["rice", "paddy"] and pred_c in ["rice", "paddy"]:
            pred_c = true_c
        confusion_matrix[true_c][pred_c] += 1
        
    # 4. Expected Calibration Error (ECE) & Reliability Diagram
    # Evaluate calibration on accepted predictions
    ece_bins = 10
    bin_counts = [0] * ece_bins
    bin_corrects = [0] * ece_bins
    bin_confs = [0.0] * ece_bins
    
    all_confs = []
    correct_confs = []
    incorrect_confs = []
    high_conf_errors = []
    low_conf_corrects = []
    
    for r in agri_records:
        conf = float(r["primary_confidence"])
        is_corr = (r["verdict"] == "PASS")
        all_confs.append(conf)
        
        if is_corr:
            correct_confs.append(conf)
            if conf < 0.40:
                low_conf_corrects.append(r)
        else:
            incorrect_confs.append(conf)
            if conf >= 0.90:
                high_conf_errors.append(r)
                
        bin_idx = min(int(conf * ece_bins), ece_bins - 1)
        bin_counts[bin_idx] += 1
        bin_confs[bin_idx] += conf
        if is_corr:
            bin_corrects[bin_idx] += 1
            
    ece = 0.0
    reliability_diagram = []
    for m in range(ece_bins):
        count = bin_counts[m]
        lower_bound = m / ece_bins
        upper_bound = (m + 1) / ece_bins
        if count > 0:
            avg_acc = bin_corrects[m] / count
            avg_conf = bin_confs[m] / count
            ece += (count / n_agri) * abs(avg_acc - avg_conf)
        else:
            avg_acc = 0.0
            avg_conf = (lower_bound + upper_bound) / 2.0
            
        reliability_diagram.append({
            "bin": f"[{lower_bound:.1f}, {upper_bound:.1f})",
            "samples": count,
            "accuracy": round(avg_acc * 100, 1),
            "confidence": round(avg_conf * 100, 1),
            "gap": round(abs(avg_acc - avg_conf) * 100, 1)
        })
        
    # 5. Defensive Evaluation
    n_def = len(def_records)
    def_correct = sum(1 for r in def_records if r["defended"])
    def_rate, def_ci_low, def_ci_high = wilson_ci(def_correct, n_def)
    
    # Breakdown by defensive stress type
    stress_types = sorted(list(set(r["image_type"] for r in def_records)))
    stress_breakdown = {}
    for st in stress_types:
        st_recs = [r for r in def_records if r["image_type"] == st]
        st_corr = sum(1 for r in st_recs if r["defended"])
        stress_breakdown[st] = {
            "total": len(st_recs),
            "defended": st_corr,
            "pass_rate": round(st_corr / len(st_recs) * 100, 1)
        }
        
    # Latency Stats
    all_latencies = [r["latency_ms"] for r in records]
    agri_latencies = [r["latency_ms"] for r in agri_records]
    def_latencies = [r["latency_ms"] for r in def_records]
    
    latency_stats = {
        "all_mean_ms": round(float(np.mean(all_latencies)), 1),
        "all_median_ms": round(float(np.median(all_latencies)), 1),
        "all_p95_ms": round(float(np.percentile(all_latencies, 95)), 1),
        "all_min_ms": round(float(np.min(all_latencies)), 1),
        "all_max_ms": round(float(np.max(all_latencies)), 1),
        "agri_mean_ms": round(float(np.mean(agri_latencies)), 1),
        "def_mean_ms": round(float(np.mean(def_latencies)), 1)
    }
    
    # 6. Failure Taxonomy & Classification
    failures = [r for r in agri_records if r["verdict"] != "PASS"]
    categorized_failures = defaultdict(list)
    
    for f in failures:
        verdict = f["verdict"]
        c_true = f["crop_true"]
        c_pred = f["crop_pred"]
        d_true = f["disease_true"]
        d_pred = f["diag_pred"]
        conf = f["primary_confidence"]
        
        # Determine taxonomy category
        if "healthy" in d_pred.lower() and "healthy" not in d_true.lower():
            cat = "10. Healthy-vs-disease representation failure"
        elif "healthy" in d_true.lower() and "healthy" not in d_pred.lower():
            cat = "10. Healthy-vs-disease representation failure"
        elif verdict == "CROSS_CROP":
            if (c_true == "apple" and c_pred == "peach") or (c_true == "peach" and c_pred == "apple"):
                cat = "8. Cross-crop visual confusion (Rosaceae leaf morphology)"
            elif (c_true == "soybean" and c_pred == "blackgram") or (c_true == "blackgram" and c_pred == "soybean"):
                cat = "8. Cross-crop visual confusion (Legume foliar overlap)"
            elif (c_true == "grape" and c_pred == "tomato") or (c_true == "tomato" and c_pred == "grape"):
                cat = "8. Cross-crop visual confusion (Symptom resemblance)"
            elif (c_true in ["chilli", "bell pepper"] and c_pred in ["chilli", "bell pepper"]):
                cat = "8. Cross-crop visual confusion (Capsicum inter-species)"
            else:
                cat = "8. Cross-crop visual confusion"
        elif verdict == "PARTIAL":
            cat = "9. Intra-crop disease confusion"
        elif verdict == "REFUSAL":
            if f["is_ood"]:
                cat = "6. OOD/gating issue (Excessive rejection of in-distribution leaf)"
            elif f["diag_status"] in ["INSUFFICIENT_EVIDENCE", "UNKNOWN"]:
                cat = "3. Crop detector ambiguity / statistical refusal"
            else:
                cat = "11. Genuine model representation limitation"
        else:
            cat = "11. Genuine model representation limitation"
            
        categorized_failures[cat].append({
            "image_id": f["image_id"],
            "crop_true": c_true,
            "crop_pred": c_pred,
            "disease_true": d_true,
            "diag_pred": d_pred,
            "confidence": conf,
            "source": f["source_dataset"]
        })
        
    # 7. Focus Area Analysis
    focus_analysis = {
        "apple_peach": {
            "apple_total": len([r for r in agri_records if r["crop_true"] == "apple"]),
            "apple_pred_peach": len([r for r in agri_records if r["crop_true"] == "apple" and r["crop_pred"] == "peach"]),
            "peach_total": len([r for r in agri_records if r["crop_true"] == "peach"]),
            "peach_pred_apple": len([r for r in agri_records if r["crop_true"] == "peach" and r["crop_pred"] == "apple"])
        },
        "soybean_blackgram": {
            "soybean_total": len([r for r in agri_records if r["crop_true"] == "soybean"]),
            "soybean_pred_blackgram": len([r for r in agri_records if r["crop_true"] == "soybean" and r["crop_pred"] == "blackgram"]),
            "blackgram_total": len([r for r in agri_records if r["crop_true"] == "blackgram"]),
            "blackgram_pred_soybean": len([r for r in agri_records if r["crop_true"] == "blackgram" and r["crop_pred"] == "soybean"])
        },
        "grape_tomato": {
            "grape_total": len([r for r in agri_records if r["crop_true"] == "grape"]),
            "grape_pred_tomato": len([r for r in agri_records if r["crop_true"] == "grape" and r["crop_pred"] == "tomato"]),
            "tomato_total": len([r for r in agri_records if r["crop_true"] == "tomato"]),
            "tomato_pred_grape": len([r for r in agri_records if r["crop_true"] == "tomato" and r["crop_pred"] == "grape"])
        },
        "strawberry_scorch_healthy": {
            "strawberry_scorch_total": len([r for r in agri_records if r["crop_true"] == "strawberry" and "scorch" in r["disease_true"].lower()]),
            "scorch_pred_healthy": len([r for r in agri_records if r["crop_true"] == "strawberry" and "scorch" in r["disease_true"].lower() and "healthy" in r["diag_pred"].lower()]),
            "strawberry_healthy_total": len([r for r in agri_records if r["crop_true"] == "strawberry" and "healthy" in r["disease_true"].lower()]),
            "healthy_pred_healthy": len([r for r in agri_records if r["crop_true"] == "strawberry" and "healthy" in r["disease_true"].lower() and "healthy" in r["diag_pred"].lower()])
        },
        "wheat_ambiguity": {
            "wheat_total": len([r for r in agri_records if r["crop_true"] == "wheat"]),
            "wheat_crop_acc": sum(1 for r in agri_records if r["crop_true"] == "wheat" and r["crop_match"]),
            "wheat_diag_pass": sum(1 for r in agri_records if r["crop_true"] == "wheat" and r["verdict"] == "PASS"),
            "wheat_partial": sum(1 for r in agri_records if r["crop_true"] == "wheat" and r["verdict"] == "PARTIAL")
        }
    }
    
    # Compile output analysis dictionary
    analysis_results = {
        "cohort_summary": {
            "total_samples": len(records),
            "agricultural_samples": n_agri,
            "defensive_samples": n_def,
            "crops_evaluated": len(crops)
        },
        "primary_metrics": {
            "top1_crop_accuracy": {
                "correct": crop_correct, "total": n_agri,
                "percentage": round(crop_acc * 100, 2),
                "ci_95": [round(crop_ci_low * 100, 2), round(crop_ci_high * 100, 2)]
            },
            "full_diagnosis_pass_rate": {
                "correct": diag_correct, "total": n_agri,
                "percentage": round(diag_pass * 100, 2),
                "ci_95": [round(diag_ci_low * 100, 2), round(diag_ci_high * 100, 2)]
            },
            "conditional_diagnosis_accuracy": {
                "correct": diag_correct, "given_crop_correct": crop_correct,
                "percentage": round(cond_diag_pass * 100, 2),
                "ci_95": [round(cond_ci_low * 100, 2), round(cond_ci_high * 100, 2)]
            },
            "partial_crop_pass_rate": {
                "count": partial_count, "total": n_agri,
                "percentage": round(partial_rate * 100, 2),
                "ci_95": [round(partial_ci_low * 100, 2), round(partial_ci_high * 100, 2)]
            },
            "cross_crop_error_rate": {
                "count": cross_crop_count, "total": n_agri,
                "percentage": round(cross_crop_rate * 100, 2),
                "ci_95": [round(cross_ci_low * 100, 2), round(cross_ci_high * 100, 2)]
            },
            "refusal_rate": {
                "count": refusal_count, "total": n_agri,
                "percentage": round(refusal_rate * 100, 2),
                "ci_95": [round(refusal_ci_low * 100, 2), round(refusal_ci_high * 100, 2)]
            },
            "macro_average_crop_accuracy": round(macro_crop_acc, 2),
            "macro_average_diagnosis_accuracy": round(macro_diag_pass, 2)
        },
        "per_crop_performance": per_crop_stats,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "calibration": {
            "expected_calibration_error": round(ece, 4),
            "mean_confidence_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else 0.0,
            "mean_confidence_incorrect": round(float(np.mean(incorrect_confs)), 4) if incorrect_confs else 0.0,
            "high_confidence_errors_count": len(high_conf_errors),
            "high_confidence_errors": [
                {
                    "image_id": e["image_id"],
                    "crop_true": e["crop_true"], "crop_pred": e["crop_pred"],
                    "disease_true": e["disease_true"], "diag_pred": e["diag_pred"],
                    "confidence": e["primary_confidence"], "source": e["source_dataset"],
                    "verdict": e["verdict"]
                }
                for e in high_conf_errors
            ],
            "low_confidence_correct_count": len(low_conf_corrects),
            "reliability_diagram": reliability_diagram
        },
        "defensive_system": {
            "total_stress_samples": n_def,
            "correct_rejections": def_correct,
            "rejection_rate": round(def_rate * 100, 2),
            "ci_95": [round(def_ci_low * 100, 2), round(def_ci_high * 100, 2)],
            "stress_type_breakdown": stress_breakdown
        },
        "latency_profile": latency_stats,
        "failure_taxonomy": {k: len(v) for k, v in categorized_failures.items()},
        "failure_details": {k: v for k, v in categorized_failures.items()},
        "target_pair_deep_dive": focus_analysis
    }
    
    with open(OUTPUT_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)
        
    print(f"\nStatistical analysis successfully written to: {OUTPUT_ANALYSIS_PATH}")
    
    # Print high level console report
    print("\n" + "="*80)
    print("                    EXTERNAL VALIDATION: SUMMARY REPORT                         ")
    print("="*80)
    print(f"Cohort Size              : {n_agri} Unseen Agricultural Images (from test_split.csv)")
    print(f"Top-1 Crop Accuracy      : {crop_acc*100:.1f}% ({crop_correct}/{n_agri}) [95% CI: {crop_ci_low*100:.1f}% - {crop_ci_high*100:.1f}%]")
    print(f"Macro Crop Accuracy      : {macro_crop_acc:.1f}%")
    print(f"Full Diagnosis Pass Rate : {diag_pass*100:.1f}% ({diag_correct}/{n_agri}) [95% CI: {diag_ci_low*100:.1f}% - {diag_ci_high*100:.1f}%]")
    print(f"Conditional Diag Accuracy: {cond_diag_pass*100:.1f}% ({diag_correct}/{crop_correct}) [95% CI: {cond_ci_low*100:.1f}% - {cond_ci_high*100:.1f}%]")
    print(f"Partial Crop Pass Rate   : {partial_rate*100:.1f}% ({partial_count}/{n_agri}) [95% CI: {partial_ci_low*100:.1f}% - {partial_ci_high*100:.1f}%]")
    print(f"Cross-Crop Error Rate    : {cross_crop_rate*100:.1f}% ({cross_crop_count}/{n_agri}) [95% CI: {cross_ci_low*100:.1f}% - {cross_ci_high*100:.1f}%]")
    print(f"Defensive Refusal Rate   : {refusal_rate*100:.1f}% ({refusal_count}/{n_agri})")
    print(f"Defensive Stress Security: {def_rate*100:.1f}% ({def_correct}/{n_def}) [95% CI: {def_ci_low*100:.1f}% - {def_ci_high*100:.1f}%]")
    print(f"Expected Calibration Err : {ece:.4f} (ECE)")
    print(f"High-Confidence Errors   : {len(high_conf_errors)} cases (>90% conf wrong)")
    print(f"Average Pipeline Latency : {latency_stats['all_mean_ms']} ms/image (Median: {latency_stats['all_median_ms']} ms)")
    print("="*80)

if __name__ == "__main__":
    main()
