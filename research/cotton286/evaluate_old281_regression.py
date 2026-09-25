import os
import sys
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM


def evaluate_old281_regression(student_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_path = "weights/efficientnet_b5_cbam_best.pt"
    
    print(f"Comparing Teacher ({teacher_path}) vs Student ({student_path}) on {device}")
    
    # Load Teacher (281 classes)
    teacher = EfficientNetB5_CBAM(num_classes=281, pretrained=False)
    t_sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(t_sd)
    teacher.to(device)
    teacher.eval()
    
    # Load Student (286 classes)
    student = EfficientNetB5_CBAM(num_classes=286, pretrained=False)
    s_sd = torch.load(student_path, map_location="cpu", weights_only=False)
    student.load_state_dict(s_sd)
    student.to(device)
    student.eval()
    
    with open("weights/class_names.txt", "r", encoding="utf-8") as f:
        orig_classes = [l.strip() for l in f if l.strip()]
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Sourced evaluation images (field_test_cohort + curated_ginger)
    eval_sources = [
        "D:/1winbackup/desktop/Ganpat University/Model/validation/field_test_cohort",
        "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger"
    ]
    
    test_files = []
    for d in eval_sources:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_files.append(os.path.join(d, f))
                    
    print(f"Sourced {len(test_files)} evaluation images across {len(eval_sources)} cohorts.")
    
    agreements = 0
    total = len(test_files)
    logit_diffs = []
    conf_diffs = []
    class_switches = []
    
    with torch.no_grad():
        for fpath in test_files:
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
            except Exception:
                continue
                
            t_logits = teacher(tensor)           # [1, 281]
            s_logits = student(tensor)           # [1, 286]
            s_logits_281 = s_logits[:, :281]     # Sliced 281 logits
            
            t_probs = F.softmax(t_logits, dim=1).squeeze(0)
            s_probs_full = F.softmax(s_logits, dim=1).squeeze(0)
            s_probs_281 = F.softmax(s_logits_281, dim=1).squeeze(0)
            
            t_conf, t_pred = torch.max(t_probs, dim=0)
            s_conf_full, s_pred_full = torch.max(s_probs_full, dim=0)
            s_conf_281, s_pred_281 = torch.max(s_probs_281, dim=0)
            
            # Logit drift
            diff = torch.abs(t_logits - s_logits_281).mean().item()
            logit_diffs.append(diff)
            
            # Confidence drift
            c_diff = abs(t_conf.item() - s_conf_full.item())
            conf_diffs.append(c_diff)
            
            # Prediction agreement
            t_idx = t_pred.item()
            s_idx = s_pred_full.item()
            
            if t_idx == s_idx:
                agreements += 1
            else:
                t_cname = orig_classes[t_idx] if t_idx < len(orig_classes) else f"Class_{t_idx}"
                s_cname = orig_classes[s_idx] if s_idx < len(orig_classes) else f"Cotton_Class_{s_idx}"
                class_switches.append({
                    "file": os.path.basename(fpath),
                    "teacher_pred": t_cname,
                    "student_pred": s_cname,
                    "teacher_conf": round(t_conf.item(), 4),
                    "student_conf": round(s_conf_full.item(), 4)
                })
                
    preservation_rate = (agreements / max(1, total)) * 100.0
    mean_logit_drift = float(np.mean(logit_diffs)) if logit_diffs else 0.0
    mean_conf_drift = float(np.mean(conf_diffs)) if conf_diffs else 0.0
    
    distill_results = {
        "teacher_checkpoint": teacher_path,
        "student_checkpoint": student_path,
        "total_eval_samples": total,
        "agreements": agreements,
        "class_switches_count": len(class_switches),
        "old_class_preservation_rate": round(preservation_rate, 2),
        "mean_logit_drift": round(mean_logit_drift, 4),
        "mean_confidence_drift": round(mean_conf_drift, 4),
        "class_switches_sample": class_switches[:10]
    }
    
    out_distill = "research/cotton286/distillation_metrics.json"
    with open(out_distill, "w", encoding="utf-8") as f:
        json.dump(distill_results, f, indent=2)
        
    regress_results = {
        "status": "PASS" if preservation_rate >= 96.0 else "REVIEW_REQUIRED",
        "benchmark_samples": total,
        "preservation_rate": round(preservation_rate, 2),
        "tolerance_threshold": 96.0,
        "regression_delta_percent": round(100.0 - preservation_rate, 2),
        "mean_logit_drift": round(mean_logit_drift, 4),
        "distillation_metrics_file": out_distill
    }
    
    out_regress = "research/cotton286/old281_regression.json"
    with open(out_regress, "w", encoding="utf-8") as f:
        json.dump(regress_results, f, indent=2)
        
    print(f"Old 281 Regression Check Complete:")
    print(f"  Preservation Rate:   {preservation_rate:.2f}% (Threshold: >= 96.0%)")
    print(f"  Mean Logit Drift:    {mean_logit_drift:.4f}")
    print(f"  Mean Conf Drift:     {mean_conf_drift:.4f}")
    print(f"Saved {out_distill} and {out_regress}")


if __name__ == "__main__":
    ckpt = "weights/research/efficientnet_b5_cbam_cotton286_v1.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    evaluate_old281_regression(ckpt)
