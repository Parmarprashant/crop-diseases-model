"""
AgriVision AI: Final External Validation Suite Runner.
Pure, passive HTTP client that evaluates the 250-sample unseen cohort
(220 agricultural strictly from test_split.csv + 30 defensive/stress samples)
against the live production FastAPI server without implementing any inference logic.
"""
import os
import sys
import time
import json
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/api/v1/diagnose"
MANIFEST_PATH = "validation/external_cohort_manifest.csv"
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VALIDATION_OUTPUT", "validation/external_validation_results.json")

def run_external_validation():
    print("=" * 80)
    print("      AGRIVISION PRODUCTION PIPELINE: 250-SAMPLE EXTERNAL VALIDATION")
    print("=" * 80)
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: {MANIFEST_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_csv(MANIFEST_PATH)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples from {MANIFEST_PATH}.")
    print(f"  - Agricultural unseen samples: {len(df[df['cohort_group'] == 'agricultural'])}")
    print(f"  - Defensive stress samples   : {len(df[df['cohort_group'] == 'defensive'])}\n")
    
    results = []
    
    for idx, row in df.iterrows():
        sample_id = str(row["image_id"])
        file_path = str(row["file_path"])
        cohort_group = str(row["cohort_group"])
        crop_true = str(row["crop"])
        disease_true = str(row["disease"])
        source_dataset = str(row["source"])
        image_type = str(row["image_type"])
        
        t0 = time.time()
        try:
            with open(file_path, "rb") as f:
                res = requests.post(API_URL, files={"file": (os.path.basename(file_path), f, "image/jpeg")}, timeout=30)
            elapsed_ms = (time.time() - t0) * 1000.0
            
            if res.status_code != 200:
                print(f"[{idx+1:3d}/{total_samples}] HTTP {res.status_code} ERROR on {sample_id} ({elapsed_ms:.1f}ms)")
                results.append({
                    "image_id": sample_id,
                    "file_path": file_path,
                    "cohort_group": cohort_group,
                    "crop_true": crop_true,
                    "disease_true": disease_true,
                    "source_dataset": source_dataset,
                    "image_type": image_type,
                    "http_status": res.status_code,
                    "error": res.text,
                    "latency_ms": round(elapsed_ms, 1)
                })
                continue
                
            data = res.json()
            
            # Extract raw API responses
            crop_info = data.get("crop") or {}
            crop_pred = crop_info.get("name", "unknown")
            crop_status = crop_info.get("status", "UNKNOWN")
            crop_conf = crop_info.get("confidence", 0.0)
            
            diag_info = data.get("diagnosis") or {}
            diag_pred = diag_info.get("name", "Unknown")
            diag_status = diag_info.get("status", "UNKNOWN")
            
            primary_info = data.get("primary_model") or {}
            primary_status = primary_info.get("status", "unknown")
            primary_accepted = primary_info.get("accepted", False)
            raw_top_pred = primary_info.get("raw_top_prediction", "")
            raw_conf = primary_info.get("raw_confidence", 0.0)
            primary_conf = primary_info.get("confidence") or raw_conf
            entropy = primary_info.get("entropy", 0.0)
            energy_score = primary_info.get("energy_score", 0.0)
            is_ood = primary_info.get("ood", False)
            rejection_reasons = primary_info.get("rejection_reasons", [])
            
            focus_info = data.get("focus_region") or {}
            is_focused = focus_info.get("is_focused", False)
            focus_region = focus_info.get("box_normalized", [0.0, 0.0, 1.0, 1.0])
            
            record = {
                "image_id": sample_id,
                "file_path": file_path,
                "cohort_group": cohort_group,
                "crop_true": crop_true,
                "disease_true": disease_true,
                "source_dataset": source_dataset,
                "image_type": image_type,
                "crop_pred": crop_pred,
                "crop_status": crop_status,
                "crop_conf": round(crop_conf, 4),
                "diag_pred": diag_pred,
                "diag_status": diag_status,
                "primary_status": primary_status,
                "primary_accepted": primary_accepted,
                "raw_top_prediction": raw_top_pred,
                "raw_confidence": round(raw_conf, 4),
                "primary_confidence": round(primary_conf, 4),
                "entropy": round(entropy, 4),
                "energy_score": round(energy_score, 4),
                "is_ood": is_ood,
                "is_focused": is_focused,
                "focus_region": focus_region,
                "rejection_reasons": rejection_reasons,
                "latency_ms": round(elapsed_ms, 1),
                "http_status": 200
            }
            results.append(record)
            
            tag = "[AUTO-FOCUSED]" if is_focused else "[FULL-FRAME]"
            if cohort_group == "defensive":
                print(f"[{idx+1:3d}/{total_samples}] DEFENSIVE | True: {disease_true:<40} | Crop: {crop_pred:<10} | Diag: {diag_pred:<28} | {tag} ({elapsed_ms:4.0f}ms)")
            else:
                crop_ok = "OK" if (crop_pred.lower() == crop_true.lower() or (crop_true.lower() in ["rice", "paddy"] and crop_pred.lower() in ["rice", "paddy"])) else "MISMATCH"
                print(f"[{idx+1:3d}/{total_samples}] AGRI [{crop_ok:<8}] | {crop_true:<10} -> {crop_pred:<10} | {disease_true:<35} -> {diag_pred:<25} | Conf: {primary_conf*100:5.1f}% | {tag} ({elapsed_ms:4.0f}ms)")
                
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000.0
            print(f"[{idx+1:3d}/{total_samples}] EXCEPTION on {sample_id}: {e}")
            results.append({
                "image_id": sample_id,
                "file_path": file_path,
                "cohort_group": cohort_group,
                "crop_true": crop_true,
                "disease_true": disease_true,
                "source_dataset": source_dataset,
                "image_type": image_type,
                "http_status": 500,
                "error": str(e),
                "latency_ms": round(elapsed_ms, 1)
            })
            
    # Save complete results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCompleted evaluation of {len(results)} samples.")
    print(f"Raw results successfully saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_external_validation()
