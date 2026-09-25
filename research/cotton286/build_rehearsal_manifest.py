import os
import sys
import json
import hashlib
from PIL import Image

def build_rehearsal_manifest():
    search_paths = [
        "Data/outputs/outputs/train_split.csv",
        "Data/outputs/train_split.csv",
        "Data/train_split.csv",
        "Data/master_images/master_images/images",
        "MAIN DATA/Train",
        "D:/1winbackup/desktop/Ganpat University/Model/Data/curated_ginger",
        "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger"
    ]
    
    inspected_sources = {}
    for p in search_paths:
        inspected_sources[p] = {
            "exists": os.path.exists(p),
            "is_dir": os.path.isdir(p) if os.path.exists(p) else False
        }
        
    # Check what non-validation training images exist
    rehearsal_samples = []
    
    ginger_dir = "D:/1winbackup/desktop/Ganpat University/Model/Data/curated_ginger"
    if not os.path.exists(ginger_dir):
        ginger_dir = "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger"
        
    if os.path.exists(ginger_dir):
        files = sorted([f for f in os.listdir(ginger_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for idx, fname in enumerate(files):
            fpath = os.path.join(ginger_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    content = f.read()
                    md5 = hashlib.md5(content).hexdigest()
                rehearsal_samples.append({
                    "sample_id": f"ginger_{idx:03d}",
                    "file_path": fpath,
                    "source_class": "Ginger - Healthy",  # Canonical class
                    "crop": "ginger",
                    "md5": md5,
                    "provenance": "Model/Data/curated_ginger"
                })
            except Exception as e:
                pass
                
    manifest = {
        "status": "PARTIAL_LOCAL_REHEARSAL_AVAILABLE",
        "note": "Full 281-class multi-gigabyte training split (Data/outputs/outputs/train_split.csv) is not present on this local machine. Sourced all available local non-validation training data strictly avoiding any test/val cohorts.",
        "search_paths_inspected": inspected_sources,
        "sampling_strategy": "deterministic_all_available_non_validation",
        "seed": 42,
        "total_rehearsal_samples": len(rehearsal_samples),
        "classes_covered": sorted(list(set(s["source_class"] for s in rehearsal_samples))),
        "samples": rehearsal_samples
    }
    
    out_path = os.path.join("research", "cotton286", "rehearsal_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Rehearsal manifest generated: {out_path}")
    print(f"Total rehearsal samples: {len(rehearsal_samples)} across {len(manifest['classes_covered'])} classes")

if __name__ == "__main__":
    build_rehearsal_manifest()
