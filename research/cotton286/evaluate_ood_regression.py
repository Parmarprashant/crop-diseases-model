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


def evaluate_ood_regression(student_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_path = "weights/efficientnet_b5_cbam_best.pt"
    
    print(f"Evaluating OOD Regression on {student_path} vs {teacher_path} on {device}")
    
    # Load Teacher (281)
    teacher = EfficientNetB5_CBAM(num_classes=281, pretrained=False)
    t_sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(t_sd)
    teacher.to(device)
    teacher.eval()
    
    # Load Student (286)
    student = EfficientNetB5_CBAM(num_classes=286, pretrained=False)
    s_sd = torch.load(student_path, map_location="cpu", weights_only=False)
    student.load_state_dict(s_sd)
    student.to(device)
    student.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # OOD cohort (open_set_v2_cohort)
    ood_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/open_set_v2_cohort"
    ood_files = []
    if os.path.exists(ood_dir):
        for f in sorted(os.listdir(ood_dir))[:100]:  # Evaluate 100 OOD samples for speed
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                ood_files.append(os.path.join(ood_dir, f))
                
    t_energies = []
    s_energies = []
    t_msps = []
    s_msps = []
    
    with torch.no_grad():
        for fpath in ood_files:
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
            except Exception:
                continue
                
            t_logits = teacher(tensor)
            s_logits = student(tensor)
            
            # Energy calculation: -T * log(sum(exp(logits / T))) with T=1.0
            t_energy = -float(torch.logsumexp(t_logits, dim=1).item())
            s_energy = -float(torch.logsumexp(s_logits, dim=1).item())
            
            t_msp = float(F.softmax(t_logits, dim=1).max(dim=1)[0].item())
            s_msp = float(F.softmax(s_logits, dim=1).max(dim=1)[0].item())
            
            t_energies.append(t_energy)
            s_energies.append(s_energy)
            t_msps.append(t_msp)
            s_msps.append(s_msp)
            
    t_mean_energy = float(np.mean(t_energies))
    s_mean_energy = float(np.mean(s_energies))
    t_mean_msp = float(np.mean(t_msps))
    s_mean_msp = float(np.mean(s_msps))
    
    # Energy delta
    energy_drift = abs(s_mean_energy - t_mean_energy)
    msp_drift = abs(s_mean_msp - t_mean_msp)
    
    results = {
        "status": "PASS" if msp_drift < 0.05 else "REVIEW_REQUIRED",
        "evaluated_ood_samples": len(ood_files),
        "teacher_mean_energy": round(t_mean_energy, 4),
        "student_mean_energy": round(s_mean_energy, 4),
        "energy_drift": round(energy_drift, 4),
        "teacher_mean_msp": round(t_mean_msp, 4),
        "student_mean_msp": round(s_mean_msp, 4),
        "msp_drift": round(msp_drift, 4),
        "ood_behavior_preserved": msp_drift < 0.05
    }
    
    out_path = "research/cotton286/ood_regression.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"OOD Regression Evaluation Complete:")
    print(f"  Teacher MSP: {t_mean_msp:.4f} | Student MSP: {s_mean_msp:.4f} (Drift: {msp_drift:.4f})")
    print(f"  Teacher Energy: {t_mean_energy:.4f} | Student Energy: {s_mean_energy:.4f} (Drift: {energy_drift:.4f})")
    print(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    ckpt = "weights/research/efficientnet_b5_cbam_cotton286_v1.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    evaluate_ood_regression(ckpt)
