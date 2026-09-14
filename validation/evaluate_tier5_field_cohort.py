"""
AgriVision AI — Tier 5 Independent Field Cohort Evaluation.

Executes side-by-side evaluation of Model A Baseline vs. Candidate C
(Calibrated Crop Expert + Model A + Mode B Resolver + Final Calibrator)
on the 400-image genuinely unseen real-world field cohort.

Adheres strictly to User Directives:
1. Full diagnosis = correct disease + correct crop handling according to evaluator.
2. Conditional diagnosis = correct disease given correct crop.
3. Separate reporting for:
   - VERIFIED-only (240 images)
   - VERIFIED + LIKELY (400 images)
4. Domain-stratified breakdown across all 6 field domains to detect systematic collapse.
5. Selective risk curves (100%, 95%, 90%, 80% coverage).
6. Evaluated strictly once under Post-Unseal Zero-Tuning Lock.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Any, Tuple

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

def compute_selective_risk(confs: List[float], correct_flags: List[float]) -> Dict[str, float]:
    """
    Computes diagnostic error rate at various coverage percentiles (100%, 95%, 90%, 80%).
    """
    if not confs:
        return {}
    paired = sorted(zip(confs, correct_flags), key=lambda x: x[0], reverse=True)
    n = len(paired)
    results = {}
    for cov_pct in [100, 95, 90, 80]:
        k = max(1, int(n * (cov_pct / 100.0)))
        top_k = paired[:k]
        acc_k = np.mean([x[1] for x in top_k])
        risk_k = 1.0 - acc_k
        results[f"risk_at_{cov_pct}pct_coverage"] = round(float(risk_k) * 100, 2)
    return results

def calculate_metrics(
    records: List[Dict[str, Any]],
    resolver_helper: CropAwareDiseaseResolver
) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    n_crop_corr = 0
    n_full_diag_corr = 0
    n_cond_diag_corr = 0
    n_cross_crop = 0
    n_accepted = 0
    confs = []
    full_correct_flags = []

    for r in records:
        c_true = r["crop_true"]
        d_true = r["disease_true"]
        c_pred = r["crop_pred"]
        d_pred = r["diag_pred"]
        accepted = r["accepted"]
        conf = r["conf"]

        c_corr = resolver_helper.are_crops_compatible(c_pred, c_true)
        d_corr = match_disease(c_true, d_true, d_pred)

        if accepted:
            n_accepted += 1

        # Explicit definitions per User Directive #2:
        # Full diagnosis = correct disease + correct crop handling according to evaluator
        is_full = (c_corr and d_corr and accepted)
        if is_full:
            n_full_diag_corr += 1

        if c_corr:
            n_crop_corr += 1
            # Conditional diagnosis = correct disease given correct crop
            if d_corr and accepted:
                n_cond_diag_corr += 1
        else:
            if accepted and c_pred != "unknown":
                n_cross_crop += 1

        confs.append(conf)
        full_correct_flags.append(1.0 if is_full else 0.0)

    crop_acc = round(n_crop_corr / n * 100, 2)
    full_diag_acc = round(n_full_diag_corr / n * 100, 2)
    cond_diag_acc = round(n_cond_diag_corr / max(n_crop_corr, 1) * 100, 2)
    cross_crop_rate = round(n_cross_crop / n * 100, 2)
    abstention_rate = round((n - n_accepted) / n * 100, 2)
    ece = round(compute_ece(confs, full_correct_flags, 15), 4)
    sel_risk = compute_selective_risk(confs, full_correct_flags)

    return {
        "sample_count": n,
        "crop_accuracy": crop_acc,
        "full_diagnosis_accuracy": full_diag_acc,
        "conditional_diagnosis_accuracy": cond_diag_acc,
        "cross_crop_errors": n_cross_crop,
        "cross_crop_error_rate": cross_crop_rate,
        "abstention_rate": abstention_rate,
        "ece": ece,
        "selective_risk": sel_risk
    }

def main():
    print("=" * 80)
    print("   AGRIVISION AI: TIER 5 INDEPENDENT FIELD COHORT BENCHMARK")
    print("=" * 80)

    manifest_path = "validation/tier5_field_cohort_manifest.csv"
    frozen_cfg_path = "validation/final_candidate_frozen_config.json"

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}. Run build_tier5_field_manifest.py first.")

    with open(frozen_cfg_path, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)

    print(f"Candidate Architecture: {frozen_cfg['candidate_architecture']}")
    print(f"Zero-Tuning Lock Status: {frozen_cfg['status']}")
    t_crop = frozen_cfg["crop_expert"]["calibration_temperature"]
    t_res = frozen_cfg["confidence_calibrator"]["temperature"]

    df_manifest = pd.read_csv(manifest_path)
    print(f"Loaded {len(df_manifest)} field images from manifest.")

    # Initialize Base Model A & Crop Expert
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute device: {dev_str}")

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

    calibrator = FinalConfidenceCalibrator(temperature=t_res)

    resolver_candidate = CropAwareDiseaseResolver(
        mode="mode_b",
        tau_crop_conf=frozen_cfg["resolver"]["tau_crop_conf"],
        tau_compat_mass=frozen_cfg["resolver"]["tau_compat_mass"],
        tau_within_crop_dom=frozen_cfg["resolver"]["tau_within_crop_dom"],
        tau_single_compat=frozen_cfg["resolver"]["tau_single_compat"],
        tau_crop_margin=frozen_cfg["resolver"]["tau_crop_margin"],
        tau_crop_entropy=frozen_cfg["resolver"]["tau_crop_entropy"],
        tau_disease_conf=frozen_cfg["resolver"]["tau_disease_conf"],
        calibrator=calibrator
    )

    print("\nRunning single-pass evaluation on Tier 5 Field Cohort...")
    records_model_a = []
    records_candidate = []
    detailed_samples = []

    t0 = time.time()

    for idx, row in df_manifest.iterrows():
        img_path = row["file_path"]
        img = Image.open(img_path).convert("RGB")

        # 1. Run Base Model A
        res_a = base_pipeline.diagnose(img)

        # 2. Run Calibrated Crop Expert
        crop_out = crop_expert.predict(img)
        crop_dist = crop_out.distribution
        sorted_crops = sorted(crop_dist.items(), key=lambda x: x[1], reverse=True)
        top1_crop = crop_out.crop
        top2_crop = sorted_crops[1][0] if len(sorted_crops) > 1 else "unknown"

        canonical_probs = getattr(base_pipeline, "last_canonical_probs", {})
        if not canonical_probs and res_a.primary_model.top_candidates:
            canonical_probs = {c["class_name"]: c["confidence"] for c in res_a.primary_model.top_candidates}

        # 3. Run Candidate Resolver
        res_cand = resolver_candidate.resolve(
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

        meta = {
            "image_id": row["image_id"],
            "source": row["source"],
            "source_domain": row["source_domain"],
            "crop_true": row["crop_true"],
            "disease_true": row["disease_true"],
            "verification_status": row["verification_status"]
        }

        # Model A record
        rec_a = {
            **meta,
            "crop_pred": res_a.crop.name,
            "diag_pred": res_a.diagnosis.name,
            "conf": res_a.primary_model.raw_confidence or 0.0,
            "accepted": res_a.primary_model.accepted
        }
        records_model_a.append(rec_a)

        # Candidate record
        rec_c = {
            **meta,
            "crop_pred": res_cand.final_crop,
            "diag_pred": res_cand.final_disease,
            "conf": res_cand.final_confidence,
            "accepted": res_cand.final_accepted,
            "resolver_state": res_cand.telemetry.resolver_state,
            "resolver_reason": res_cand.telemetry.resolver_reason
        }
        records_candidate.append(rec_c)

        detailed_samples.append({
            "image_id": row["image_id"],
            "domain": row["source_domain"],
            "verification_status": row["verification_status"],
            "crop_true": row["crop_true"],
            "disease_true": row["disease_true"],
            "model_a": {
                "crop": res_a.crop.name,
                "disease": res_a.diagnosis.name,
                "confidence": round(res_a.primary_model.raw_confidence or 0.0, 4),
                "accepted": res_a.primary_model.accepted
            },
            "candidate": {
                "crop": res_cand.final_crop,
                "disease": res_cand.final_disease,
                "confidence": round(res_cand.final_confidence, 4),
                "accepted": res_cand.final_accepted,
                "state": res_cand.telemetry.resolver_state
            }
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_manifest):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            rem = (len(df_manifest) - (idx + 1)) / rate
            print(f"  Processed {idx + 1}/{len(df_manifest)} samples ({elapsed:.1f}s elapsed, ~{rem:.0f}s remaining)...")

    # =========================================================================
    # STRATIFIED REPORTING (Directives #4 & #5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("   TIER 5 FIELD COHORT EVALUATION RESULTS")
    print("=" * 80)

    # 1. Overall (VERIFIED + LIKELY, N=400)
    overall_a = calculate_metrics(records_model_a, resolver_candidate)
    overall_c = calculate_metrics(records_candidate, resolver_candidate)

    # 2. VERIFIED-only subset (N=240)
    verified_records_a = [r for r in records_model_a if r["verification_status"] == "VERIFIED"]
    verified_records_c = [r for r in records_candidate if r["verification_status"] == "VERIFIED"]
    verified_a = calculate_metrics(verified_records_a, resolver_candidate)
    verified_c = calculate_metrics(verified_records_c, resolver_candidate)

    # 3. LIKELY subset (N=160)
    likely_records_a = [r for r in records_model_a if r["verification_status"] == "LIKELY"]
    likely_records_c = [r for r in records_candidate if r["verification_status"] == "LIKELY"]
    likely_a = calculate_metrics(likely_records_a, resolver_candidate)
    likely_c = calculate_metrics(likely_records_c, resolver_candidate)

    def print_comparison_table(title: str, m_a: Dict[str, Any], m_c: Dict[str, Any]):
        print(f"\n--- {title} ---")
        print(f"{'Metric':<34} | {'Model A Baseline':<18} | {'Candidate (Resolver)':<20} | {'Delta / Verdict'}")
        print("-" * 90)
        c_delta = m_c['crop_accuracy'] - m_a['crop_accuracy']
        f_delta = m_c['full_diagnosis_accuracy'] - m_a['full_diagnosis_accuracy']
        cd_delta = m_c['conditional_diagnosis_accuracy'] - m_a['conditional_diagnosis_accuracy']
        cc_red = m_a['cross_crop_errors'] - m_c['cross_crop_errors']
        
        print(f"{'Top-1 Crop Accuracy':<34} | {m_a['crop_accuracy']:>16.2f}% | {m_c['crop_accuracy']:>18.2f}% | {c_delta:>+6.2f}%")
        print(f"{'Full Diagnosis Accuracy':<34} | {m_a['full_diagnosis_accuracy']:>16.2f}% | {m_c['full_diagnosis_accuracy']:>18.2f}% | {f_delta:>+6.2f}%")
        print(f"{'Conditional Diagnosis Acc':<34} | {m_a['conditional_diagnosis_accuracy']:>16.2f}% | {m_c['conditional_diagnosis_accuracy']:>18.2f}% | {cd_delta:>+6.2f}%")
        print(f"{'Cross-Crop Errors':<34} | {m_a['cross_crop_errors']:>13} ({m_a['cross_crop_error_rate']}%) | {m_c['cross_crop_errors']:>15} ({m_c['cross_crop_error_rate']}%) | {cc_red:>+3} errors")
        print(f"{'Abstention / Refusal Rate':<34} | {m_a['abstention_rate']:>16.2f}% | {m_c['abstention_rate']:>18.2f}% | {m_c['abstention_rate'] - m_a['abstention_rate']:>+6.2f}%")
        print(f"{'Calibrated ECE':<34} | {m_a['ece']:>16.4f}  | {m_c['ece']:>18.4f}  | {m_c['ece'] - m_a['ece']:>+6.4f}")

    print_comparison_table(f"OVERALL COHORT: VERIFIED + LIKELY (N={len(df_manifest)})", overall_a, overall_c)
    print_comparison_table(f"VERIFIED-ONLY COHORT (N={len(verified_records_a)})", verified_a, verified_c)
    print_comparison_table(f"LIKELY COHORT (N={len(likely_records_a)})", likely_a, likely_c)

    # 4. Domain-stratified breakdown (Directive #5: Check for systematic collapse)
    print("\n" + "=" * 80)
    print("DOMAIN-STRATIFIED BREAKDOWN (Testing for Systematic Domain Collapse)")
    print("=" * 80)
    domains = sorted(df_manifest["source_domain"].unique())
    domain_results = {}
    systematic_collapse_flag = False

    for dom in domains:
        dom_a_recs = [r for r in records_model_a if r["source_domain"] == dom]
        dom_c_recs = [r for r in records_candidate if r["source_domain"] == dom]
        m_dom_a = calculate_metrics(dom_a_recs, resolver_candidate)
        m_dom_c = calculate_metrics(dom_c_recs, resolver_candidate)
        
        diag_delta = m_dom_c["full_diagnosis_accuracy"] - m_dom_a["full_diagnosis_accuracy"]
        crop_delta = m_dom_c["crop_accuracy"] - m_dom_a["crop_accuracy"]

        # Systematic collapse defined as: severe regression (>15% drop in full diag or crop)
        has_collapsed = (diag_delta < -15.0) or (crop_delta < -15.0)
        if has_collapsed:
            systematic_collapse_flag = True

        status_str = "[COLLAPSE WARNING]" if has_collapsed else "[PASS]"

        print(f"Domain: {dom:<18} (N={m_dom_a['sample_count']}) {status_str}")
        print(f"  Crop Acc:      Model A {m_dom_a['crop_accuracy']:>5.1f}%  ->  Candidate {m_dom_c['crop_accuracy']:>5.1f}% ({crop_delta:>+5.1f}%)")
        print(f"  Full Diag Acc: Model A {m_dom_a['full_diagnosis_accuracy']:>5.1f}%  ->  Candidate {m_dom_c['full_diagnosis_accuracy']:>5.1f}% ({diag_delta:>+5.1f}%)")
        print(f"  Cross-Crop:    Model A {m_dom_a['cross_crop_errors']:>2} errs  ->  Candidate {m_dom_c['cross_crop_errors']:>2} errs")
        print(f"  ECE:           Model A {m_dom_a['ece']:.4f}  ->  Candidate {m_dom_c['ece']:.4f}")

        domain_results[dom] = {
            "model_a": m_dom_a,
            "candidate": m_dom_c,
            "crop_delta": crop_delta,
            "full_diagnosis_delta": diag_delta,
            "systematic_collapse": has_collapsed
        }

    print("\n" + "=" * 80)
    print(f"SYSTEMATIC DOMAIN COLLAPSE AUDIT: {'FAILED (Collapse Detected)' if systematic_collapse_flag else 'PASSED (No Systematic Domain Collapse)'}")
    print("=" * 80)

    # State distribution for candidate
    states_count = {}
    for r in records_candidate:
        s = r.get("resolver_state", "UNKNOWN")
        states_count[s] = states_count.get(s, 0) + 1
    print("\nCandidate Resolver State Distribution:")
    for s, c in sorted(states_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s:<32}: {c:>4} ({c/len(records_candidate)*100:.1f}%)")

    # Save comprehensive results
    output_payload = {
        "manifest_file": manifest_path,
        "sample_count": len(df_manifest),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics": {
            "verified_plus_likely_overall": {
                "model_a": overall_a,
                "candidate": overall_c
            },
            "verified_only": {
                "model_a": verified_a,
                "candidate": verified_c
            },
            "likely_only": {
                "model_a": likely_a,
                "candidate": likely_c
            }
        },
        "domain_stratified": domain_results,
        "systematic_collapse_detected": systematic_collapse_flag,
        "candidate_states_distribution": states_count,
        "detailed_samples": detailed_samples
    }

    out_json = "validation/tier5_field_cohort_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[OK] Saved Tier 5 benchmark results to: {out_json}")

if __name__ == "__main__":
    main()
