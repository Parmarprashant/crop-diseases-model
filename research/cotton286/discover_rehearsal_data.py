import os
import sys
import json
import glob

def discover_data():
    root_dirs = [
        "D:/1winbackup/desktop/Ganpat University",
        "D:/1winbackup/desktop/Ganpat University/crop-diseases-model",
        "D:/1winbackup/desktop/Ganpat University/Model",
        "D:/1winbackup/desktop/Ganpat University/kisandost"
    ]
    
    target_filenames = [
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
        "class_id_map.csv",
        "train.csv",
        "dataset.yaml",
        "master_images"
    ]
    
    archive_extensions = ('.zip', '.tar', '.tar.gz', '.tgz', '.7z', '.rar', '.parquet')
    
    findings = {
        "searched_roots": root_dirs,
        "target_files_found": [],
        "archives_found": [],
        "image_directories": [],
        "dataset_folders": []
    }
    
    print("Searching for training files, split manifests, and archives across workspace...")
    
    for base in root_dirs:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            # Skip git and cache internals to avoid noise, but note if relevant
            if ".git" in root or "node_modules" in root or ".next" in root:
                continue
                
            # Check target filenames
            for f in files:
                f_lower = f.lower()
                if any(target in f_lower for target in target_filenames):
                    fpath = os.path.join(root, f)
                    findings["target_files_found"].append({
                        "file": f,
                        "path": fpath,
                        "size_bytes": os.path.getsize(fpath)
                    })
                if f_lower.endswith(archive_extensions):
                    fpath = os.path.join(root, f)
                    findings["archives_found"].append({
                        "file": f,
                        "path": fpath,
                        "size_bytes": os.path.getsize(fpath)
                    })
                    
            # Check image directories
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            if len(images) > 0:
                findings["image_directories"].append({
                    "directory": root,
                    "image_count": len(images),
                    "sample_files": images[:3]
                })
                
    out_path = "research/cotton286/data_discovery_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
        
    print(f"Data discovery complete. Found:")
    print(f"  Target files: {len(findings['target_files_found'])}")
    for tf in findings['target_files_found']:
        print(f"    - {tf['file']} at {tf['path']} ({tf['size_bytes']} bytes)")
    print(f"  Archives: {len(findings['archives_found'])}")
    for ar in findings['archives_found']:
        print(f"    - {ar['file']} at {ar['path']} ({ar['size_bytes'] / (1024*1024):.2f} MB)")
    print(f"  Image directories: {len(findings['image_directories'])}")
    for idir in sorted(findings['image_directories'], key=lambda x: x['image_count'], reverse=True)[:15]:
        print(f"    - {idir['image_count']} images in {idir['directory']}")

if __name__ == "__main__":
    discover_data()
