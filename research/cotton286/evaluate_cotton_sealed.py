import os
import sys
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM


def evaluate_sealed_test(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Sealed Test on: {checkpoint_path} using {device}")
    
    with open("research/cotton286/cotton_split_manifest.json", "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    sealed_items = split_manifest["splits"]["sealed_test"]
    
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
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    cotton_indices = [class_to_idx[c] for c in cotton_classes]
    
    # Load model
    model = EfficientNetB5_CBAM(num_classes=286, pretrained=False)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    y_true = []
    y_pred = []
    confidences = []
    
    # Confusion matrix tracker: [5, 5]
    conf_mat = np.zeros((5, 5), dtype=int)
    class_to_c_idx = {c: i for i, c in enumerate(cotton_classes)}
    
    with torch.no_grad():
        for item in sealed_items:
            fpath = item["file_path"]
            cname = item["canonical_class"]
            true_idx = class_to_idx[cname]
            true_c_idx = class_to_c_idx[cname]
            
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
            top_prob, pred_idx = torch.topk(probs, k=1)
            p_idx = pred_idx.item()
            p_conf = top_prob.item()
            
            y_true.append(true_idx)
            y_pred.append(p_idx)
            confidences.append(p_conf)
            
            # If predicted class is one of the cotton classes
            if p_idx in cotton_indices:
                pred_c_idx = cotton_indices.index(p_idx)
                conf_mat[true_c_idx, pred_c_idx] += 1
            else:
                # Predicted an old class instead of cotton
                pass
                
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total = len(y_true)
    correct = (y_true == y_pred).sum()
    overall_acc = (correct / max(1, total)) * 100.0
    
    per_class_metrics = {}
    f1_list = []
    rec_list = []
    prec_list = []
    
    for i, cname in enumerate(cotton_classes):
        c_idx = class_to_idx[cname]
        tp = np.sum((y_true == c_idx) & (y_pred == c_idx))
        fp = np.sum((y_true != c_idx) & (y_pred == c_idx))
        fn = np.sum((y_true == c_idx) & (y_pred != c_idx))
        tn = np.sum((y_true != c_idx) & (y_pred != c_idx))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class_metrics[cname] = {
            "total_sealed_test": int(np.sum(y_true == c_idx)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "f1_score": round(f1 * 100.0, 2)
        }
        f1_list.append(f1)
        rec_list.append(rec)
        prec_list.append(prec)
        
    macro_f1 = float(np.mean(f1_list)) * 100.0
    balanced_acc = float(np.mean(rec_list)) * 100.0
    mean_conf = float(np.mean(confidences))
    
    results = {
        "checkpoint": checkpoint_path,
        "total_sealed_samples": total,
        "overall_accuracy": round(overall_acc, 2),
        "balanced_accuracy": round(balanced_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "mean_confidence": round(mean_conf, 4),
        "statistical_limitations": {
            "sample_size": total,
            "note": "Per-class sample count is ~6 images in sealed test due to dataset size (~200 images total)."
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "classes": cotton_classes,
            "matrix": conf_mat.tolist()
        }
    }
    
    out_path = "research/cotton286/cotton_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Sealed Test Evaluation Complete:")
    print(f"  Overall Accuracy:  {overall_acc:.2f}%")
    print(f"  Balanced Accuracy: {balanced_acc:.2f}%")
    print(f"  Macro F1:          {macro_f1:.2f}%")
    print(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    ckpt = "weights/research/efficientnet_b5_cbam_cotton286_v1.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    evaluate_sealed_test(ckpt)
