import os
import sys
import json
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM


def evaluate_cotton_negatives(student_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Cotton Negative Specificity Test on {student_path} using {device}")
    
    # Load 286 model
    model = EfficientNetB5_CBAM(num_classes=286, pretrained=False)
    sd = torch.load(student_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    
    with open("weights/class_names.txt", "r", encoding="utf-8") as f:
        orig_classes = [l.strip() for l in f if l.strip()]
        
    cotton_classes = [
        "Cotton - Aphids",
        "Cotton - Bacterial Blight",
        "Cotton - Healthy",
        "Cotton - Powdery Mildew",
        "Cotton - Target Spot"
    ]
    all_classes = orig_classes + cotton_classes
    cotton_indices = set(range(281, 286))
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Negative cohorts
    negative_sources = [
        ("field_test_cohort", "D:/1winbackup/desktop/Ganpat University/Model/validation/field_test_cohort"),
        ("sunflower_holdout", "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/sunflower_holdout"),
        ("rose_leaf_holdout", "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/rose_leaf_holdout"),
        ("curated_ginger", "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger")
    ]
    
    total_negatives = 0
    false_cotton_count = 0
    per_cohort_results = {}
    false_positives = []
    
    with torch.no_grad():
        for cohort_name, cohort_dir in negative_sources:
            if not os.path.exists(cohort_dir):
                continue
                
            c_total = 0
            c_false_cotton = 0
            
            for fname in sorted(os.listdir(cohort_dir)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                fpath = os.path.join(cohort_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        img = img.convert("RGB")
                        tensor = transform(img).unsqueeze(0).to(device)
                except Exception:
                    continue
                    
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
                conf, pred_idx = torch.max(probs, dim=0)
                
                p_idx = pred_idx.item()
                p_conf = conf.item()
                
                c_total += 1
                total_negatives += 1
                
                if p_idx in cotton_indices:
                    c_false_cotton += 1
                    false_cotton_count += 1
                    false_positives.append({
                        "file": fname,
                        "cohort": cohort_name,
                        "predicted_cotton_class": all_classes[p_idx],
                        "confidence": round(p_conf, 4)
                    })
                    
            per_cohort_results[cohort_name] = {
                "total_samples": c_total,
                "false_cotton_count": c_false_cotton,
                "false_cotton_rate": round((c_false_cotton / max(1, c_total)) * 100.0, 2)
            }
            
    false_acceptance_rate = (false_cotton_count / max(1, total_negatives)) * 100.0
    
    results = {
        "status": "PASS" if false_acceptance_rate <= 5.0 else "FAIL",
        "total_negative_samples": total_negatives,
        "false_cotton_count": false_cotton_count,
        "false_cotton_acceptance_rate": round(false_acceptance_rate, 2),
        "tolerance_threshold": 5.0,
        "per_cohort_results": per_cohort_results,
        "false_positives": false_positives[:10]
    }
    
    out_path = "research/cotton286/cotton_negative_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Cotton Negative Specificity Test Complete:")
    print(f"  Total Negatives:              {total_negatives}")
    print(f"  False Cotton Acceptance Rate: {false_acceptance_rate:.2f}% (Tolerance: <= 5.0%)")
    print(f"  Status:                       {results['status']}")
    print(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    ckpt = "weights/research/efficientnet_b5_cbam_cotton286_v1.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    evaluate_cotton_negatives(ckpt)
