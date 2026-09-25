import os
import sys
import json
import hashlib
from PIL import Image

def audit_dataset():
    dataset_root = "Cotton_Leaves"
    test_dir = os.path.join(dataset_root, "Test")
    
    expected_classes = {
        "Aphids edited": "Cotton - Aphids",
        "Bacterial Blight edited": "Cotton - Bacterial Blight",
        "Healthy leaf edited": "Cotton - Healthy",
        "Powdery Mildew Edited": "Cotton - Powdery Mildew",
        "Target spot edited": "Cotton - Target Spot"
    }
    
    audit_data = {
        "dataset_root": os.path.abspath(dataset_root),
        "source_folder": os.path.abspath(test_dir),
        "total_files": 0,
        "valid_images": 0,
        "corrupt_files": [],
        "duplicates": [],
        "per_class_counts": {},
        "canonical_class_mapping": expected_classes,
        "image_dimensions": {},
        "file_manifest": []
    }
    
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} does not exist!")
        sys.exit(1)
        
    md5_to_files = {}
    
    for folder_name in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        canonical_name = expected_classes.get(folder_name, folder_name)
        class_files = 0
        dim_list = []
        
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
                
            audit_data["total_files"] += 1
            class_files += 1
            
            # Read file bytes for hash
            try:
                with open(fpath, "rb") as f:
                    content = f.read()
                    file_hash = hashlib.md5(content).hexdigest()
                    sha256_hash = hashlib.sha256(content).hexdigest()
            except Exception as e:
                audit_data["corrupt_files"].append({"path": fpath, "error": f"Read error: {e}"})
                continue
                
            # Check duplicate
            if file_hash in md5_to_files:
                audit_data["duplicates"].append({
                    "original": md5_to_files[file_hash],
                    "duplicate": fpath,
                    "md5": file_hash
                })
            else:
                md5_to_files[file_hash] = fpath
                
            # Check image integrity & dimensions
            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode
                    dim_list.append((w, h))
                    audit_data["valid_images"] += 1
                    
                    audit_data["file_manifest"].append({
                        "file_path": fpath,
                        "relative_path": os.path.relpath(fpath, dataset_root),
                        "folder_name": folder_name,
                        "canonical_class": canonical_name,
                        "width": w,
                        "height": h,
                        "mode": mode,
                        "md5": file_hash,
                        "sha256": sha256_hash
                    })
            except Exception as e:
                audit_data["corrupt_files"].append({"path": fpath, "error": f"Corrupt image: {e}"})
                
        audit_data["per_class_counts"][canonical_name] = {
            "source_folder": folder_name,
            "total_files": class_files,
            "min_dimension": min(dim_list) if dim_list else None,
            "max_dimension": max(dim_list) if dim_list else None
        }
        
    out_path = os.path.join("research", "cotton286", "dataset_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"Audit completed: {audit_data['valid_images']}/{audit_data['total_files']} valid images.")
    print(f"Corrupt files: {len(audit_data['corrupt_files'])}, Duplicates: {len(audit_data['duplicates'])}")
    for c, stats in audit_data["per_class_counts"].items():
        print(f"  {c}: {stats['total_files']} images (min: {stats['min_dimension']}, max: {stats['max_dimension']})")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    audit_dataset()
