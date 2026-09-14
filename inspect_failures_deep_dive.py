"""
Deep-Dive Failure Diagnostic Inspector for AgriVision AI.
Analyzes the 6 prioritized failure/ambiguity cases with microscopic precision across:
- Raw logits & raw class probabilities
- Canonical aggregated class probabilities
- Crop Detector internal evidence (affinity, runner-up margin, status)
- Energy score, Entropy, and OOD gating values
- Auto-Leaf Focus proposals, candidate logits, and Step 7.5 recovery decisions
"""
import os
import sys
import json
import torch
import numpy as np
from PIL import Image

from models.efficientnet_cbam import build_efficientnet_cbam, DiseaseClassifierInference
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.gemini_fallback import GeminiVisionFallback
from inference.crop_detector import CropDetector
from inference.ood_detector import OODDetector
from inference.leaf_focus import LeafFocusDetector
from inference.pipeline import HierarchicalAgriDiagnosticPipeline, AGGREGATOR, CANONICAL_CLASS_MAP

CASES = [
    {
        "id": "img_00003854",
        "crop_true": "apple",
        "disease_true": "Apple - Scab",
        "path": "Data/master_images/master_images/images/img_00003854.jpg",
        "label": "Case 1: Apple Scab -> Peach Leaf Curl"
    },
    {
        "id": "img_00038868",
        "crop_true": "grape",
        "disease_true": "Grape - grape downy mildew",
        "path": "Data/master_images/master_images/images/img_00038868.jpg",
        "label": "Case 2: Grape Downy Mildew -> Tomato Late Blight"
    },
    {
        "id": "img_00040433",
        "crop_true": "soybean",
        "disease_true": "Soybean - soybean frog eye leaf spot Bing",
        "path": "Data/master_images/master_images/images/img_00040433.jpg",
        "label": "Case 3: Soybean Frog Eye Spot -> Blackgram Anthracnose"
    },
    {
        "id": "img_00041701",
        "crop_true": "wheat",
        "disease_true": "Wheat - wheat bacterial leaf streak (black chaff) Baidu",
        "path": "Data/master_images/master_images/images/img_00041701.jpg",
        "label": "Case 4: Wheat Bacterial Leaf Streak -> Powdery Mildew"
    },
    {
        "id": "img_00057650",
        "crop_true": "chilli",
        "disease_true": "Chilli - Leafcurl",
        "path": "Data/master_images/master_images/images/img_00057650.jpg",
        "label": "Case 5: Chilli Leafcurl -> Leaf Curl"
    },
    {
        "id": "img_00040900",
        "crop_true": "strawberry",
        "disease_true": "Strawberry - strawberry leaf scorch",
        "path": "Data/master_images/master_images/images/img_00040900.jpg",
        "label": "Case 6: Strawberry Leaf Scorch -> Healthy"
    }
]

def run_deep_dive():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = [line.strip() for line in open("weights/class_names.txt") if line.strip()]
    weights_path = "weights/efficientnet_b5_cbam_best.pt"
    thresholds_path = "weights/calibration_thresholds.json"
    
    print(f"Loading preserved checkpoint: {weights_path} on {device}...")
    classifier_model = build_efficientnet_cbam(len(class_names), weights_path, device)
    classifier_inf = DiseaseClassifierInference(classifier_model, class_names, device=device, img_size=256)
    pest_detector = YOLOv8PestDetector(device="cpu")
    segmenter = LesionSegmenter(device=device)
    gemini_fallback = GeminiVisionFallback(api_key="")
    
    pipeline = HierarchicalAgriDiagnosticPipeline(
        classifier=classifier_inf,
        pest_detector=pest_detector,
        segmenter=segmenter,
        gemini_fallback=gemini_fallback,
        thresholds_path=thresholds_path
    )
    
    crop_detector = pipeline.crop_detector
    ood_detector = pipeline.ood_detector
    focus_detector = pipeline.leaf_focus_detector
    
    diagnostics = {}
    
    for c in CASES:
        print(f"\n" + "="*80)
        print(f"DIAGNOSTIC DEEP-DIVE: {c['label']} ({c['id']})")
        print(f"True Crop: {c['crop_true']} | True Condition: {c['disease_true']}")
        print(f"="*80)
        
        im = Image.open(c["path"]).convert("RGB")
        w, h = im.size
        print(f"Image Resolution: {w}x{h}")
        
        # 1. Full-frame Classifier Prediction
        raw_res = classifier_inf.predict(im, top_k=10)
        raw_logits = raw_res["raw_logits"]
        raw_probs = raw_res["all_probabilities"]
        
        # Top 5 Raw Classes
        top_raw_indices = np.argsort(raw_probs)[::-1][:5]
        print("\n--- [A] RAW CLASSIFIER OUTPUT (Top 5 / 281 Classes) ---")
        for rank, idx in enumerate(top_raw_indices, 1):
            name = class_names[idx]
            p = float(raw_probs[idx])
            z = float(raw_logits[idx])
            print(f"  {rank}. {name:<45} | Prob: {p*100:6.2f}% | Logit: {z:6.3f}")
            
        # 2. Canonical Probability Aggregation
        canon_probs, canon_names, top_canon = AGGREGATOR.aggregate_numpy(raw_probs)
        print("\n--- [B] CANONICAL AGGREGATION (Top 5 / 167 Canonical Classes) ---")
        for rank, item in enumerate(top_canon[:5], 1):
            print(f"  {rank}. {item['class_name']:<45} | Aggregated Prob: {item['confidence']*100:6.2f}%")
            
        # 3. Crop Detector Internal Reasoning
        canon_probs_dict = {p["class_name"]: p["confidence"] for p in top_canon}
        crop_res = crop_detector.detect_from_predictions(canon_probs_dict)
        print("\n--- [C] CROP DETECTOR DECISION ---")
        print(f"  Detected Crop : {crop_res.crop}")
        print(f"  Crop Status   : {crop_res.status}")
        print(f"  Confidence    : {crop_res.confidence*100:.2f}%")
        print(f"  Peak Margin   : {crop_res.margin*100:.2f}% (Top vs Runner-Up)")
        print(f"  Explanation   : {crop_res.explanation}")
        
        # 4. Energy & OOD Metrics
        ood_res = ood_detector.evaluate(raw_logits, raw_probs)
        print("\n--- [D] OOD & CALIBRATION GATING ---")
        print(f"  Free Energy E(x; T) : {ood_res.energy_score:7.3f} (Threshold: {ood_detector.max_energy_threshold})")
        print(f"  Normalized Entropy  : {ood_res.normalized_entropy:7.3f} (Threshold: {ood_detector.max_entropy_threshold})")
        print(f"  Max Softmax Prob    : {ood_res.max_softmax_probability:7.3f} (Threshold: {ood_detector.min_msp_threshold})")
        print(f"  OOD Flag (is_ood)   : {ood_res.is_ood}")
        print(f"  Rejection Reasons   : {ood_res.rejection_reasons if ood_res.rejection_reasons else 'None (ACCEPTED)'}")
        
        # 5. Auto-Leaf Focus Inspection
        focus_candidates = focus_detector.detect_focus_candidates(im, max_candidates=3)
        print(f"\n--- [E] AUTO-LEAF FOCUS PROPOSALS ({len(focus_candidates)} candidates) ---")
        for f_idx, cand in enumerate(focus_candidates, 1):
            print(f"  Candidate {f_idx}: is_focused={cand.is_focused} | score={cand.focus_score:.4f} | box_pixels={cand.box_pixels} | box_norm={[round(x, 3) for x in cand.box_normalized]} | msg='{cand.message}'")
            if cand.is_focused:
                f_raw = classifier_inf.predict(cand.cropped_image, top_k=5)
                _, _, f_top_c = AGGREGATOR.aggregate_numpy(f_raw["all_probabilities"])
                f_top_canon = f_top_c[0]
                f_crop_res = crop_detector.detect_from_predictions({p["class_name"]: p["confidence"] for p in f_top_c})
                f_ood = ood_detector.evaluate(f_raw["raw_logits"], f_raw["all_probabilities"])
                print(f"    -> Focused Top Class : {f_top_canon['class_name']} ({f_top_canon['confidence']*100:.1f}%)")
                print(f"    -> Focused Crop      : {f_crop_res.crop} (conf: {f_crop_res.confidence*100:.1f}%, status: {f_crop_res.status})")
                print(f"    -> Focused OOD       : is_ood={f_ood.is_ood}, energy={f_ood.energy_score:.3f}, entropy={f_ood.normalized_entropy:.3f}")
        
        # 6. Complete End-to-End Pipeline Output (Unhinted vs Hinted)
        pipe_res = pipeline.diagnose(im)
        pipe_hinted = pipeline.diagnose(im, user_crop_hint=c["crop_true"])
        print("\n--- [F] FINAL END-TO-END PIPELINE OUTCOME ---")
        print(f"  [Unhinted] Crop  : {pipe_res.crop.name} (Status: {pipe_res.crop.status}, Conf: {pipe_res.crop.confidence*100:.1f}%)")
        print(f"  [Unhinted] Diag  : {pipe_res.diagnosis.name} (Status: {pipe_res.diagnosis.status})")
        print(f"  [Unhinted] Focus : {pipe_res.focus_region.is_focused}")
        print(f"  [With Hint] Crop : {pipe_hinted.crop.name} (Status: {pipe_hinted.crop.status}, Conf: {pipe_hinted.crop.confidence*100:.1f}%)")
        print(f"  [With Hint] Diag : {pipe_hinted.diagnosis.name} (Status: {pipe_hinted.diagnosis.status})")
        
        diagnostics[c["id"]] = {
            "case": c["label"],
            "true_crop": c["crop_true"],
            "true_disease": c["disease_true"],
            "top_raw": [{"class": class_names[idx], "prob": float(raw_probs[idx]), "logit": float(raw_logits[idx])} for idx in top_raw_indices],
            "top_canonical": top_canon[:5],
            "crop_detector": {
                "crop": crop_res.crop,
                "status": crop_res.status,
                "confidence": crop_res.confidence,
                "margin": crop_res.margin
            },
            "ood_metrics": {
                "energy": ood_res.energy_score,
                "entropy": ood_res.normalized_entropy,
                "msp": ood_res.max_softmax_probability,
                "is_ood": ood_res.is_ood,
                "reasons": ood_res.rejection_reasons
            },
            "pipeline_outcome": {
                "crop": pipe_res.crop.name,
                "crop_status": pipe_res.crop.status,
                "diagnosis": pipe_res.diagnosis.name,
                "diagnosis_status": pipe_res.diagnosis.status,
                "is_focused": pipe_res.focus_region.is_focused
            }
        }
        
    out_file = "weights/failure_deep_dive_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nComprehensive diagnostic data saved to: {out_file}")

if __name__ == "__main__":
    run_deep_dive()
