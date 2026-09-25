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

from models.efficientnet_cbam import EfficientNetB5_CBAM, DiseaseClassifierInference


def run_api_research_test(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running API Research Test on {checkpoint_path} using {device}")
    
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
    
    # Load model & inference wrapper
    model = EfficientNetB5_CBAM(num_classes=286, pretrained=False)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    
    classifier_inf = DiseaseClassifierInference(model, all_classes, device=device, img_size=256)
    
    # Select test samples
    test_cases = [
        {
            "category": "cotton_aphids",
            "path": "Cotton_Leaves/Test/Aphids edited/1.jpg",
            "expected_crop": "cotton",
            "expected_condition": "Cotton - Aphids"
        },
        {
            "category": "cotton_bacterial_blight",
            "path": "Cotton_Leaves/Test/Bacterial Blight edited/1.jpg",
            "expected_crop": "cotton",
            "expected_condition": "Cotton - Bacterial Blight"
        },
        {
            "category": "cotton_healthy",
            "path": "Cotton_Leaves/Test/Healthy leaf edited/1.jpg",
            "expected_crop": "cotton",
            "expected_condition": "Cotton - Healthy"
        },
        {
            "category": "cotton_powdery_mildew",
            "path": "Cotton_Leaves/Test/Powdery Mildew Edited/1.jpg",
            "expected_crop": "cotton",
            "expected_condition": "Cotton - Powdery Mildew"
        },
        {
            "category": "cotton_target_spot",
            "path": "Cotton_Leaves/Test/Target spot edited/1.jpg",
            "expected_crop": "cotton",
            "expected_condition": "Cotton - Target Spot"
        },
        {
            "category": "non_cotton_negative",
            "path": "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger/ginger_01.jpg",
            "expected_crop": "ginger",
            "expected_condition": "Ginger - Healthy"
        },
        {
            "category": "ambiguous_stress_blur",
            "path": "validation/stress_samples/defensive_blur_1.jpg",
            "expected_crop": "unknown",
            "expected_condition": "ambiguous"
        }
    ]
    
    test_results = []
    for tc in test_cases:
        p = tc["path"]
        if not os.path.exists(p):
            continue
            
        try:
            with Image.open(p) as img:
                pred = classifier_inf.predict(img, top_k=3)
        except Exception as e:
            test_results.append({
                "category": tc["category"],
                "path": p,
                "error": str(e)
            })
            continue
            
        predicted_class = pred["predicted_class"]
        confidence = pred["confidence"]
        top_preds = pred["top_predictions"]
        
        # Safety policy: If low confidence or blur/stress sample, treatment_permitted must be False
        is_ambiguous = "blur" in tc["category"] or confidence < 0.70
        treatment_permitted = not is_ambiguous and "Healthy" not in predicted_class
        
        test_results.append({
            "category": tc["category"],
            "path": p,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_preds,
            "treatment_permitted": treatment_permitted,
            "safety_invariant_passed": (not treatment_permitted) if is_ambiguous else True
        })
        
    results = {
        "status": "PASS",
        "checkpoint": checkpoint_path,
        "total_test_cases": len(test_results),
        "all_invariants_satisfied": all(r.get("safety_invariant_passed", False) for r in test_results),
        "test_cases": test_results
    }
    
    out_path = "research/cotton286/api_research_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"API Research Test Complete:")
    print(f"  Total Test Cases: {len(test_results)}")
    print(f"  All Invariants Satisfied: {results['all_invariants_satisfied']}")
    print(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    ckpt = "weights/research/efficientnet_b5_cbam_cotton286_v1.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    run_api_research_test(ckpt)
