import os
import sys
import json
import random
from collections import defaultdict

def create_split():
    audit_path = os.path.join("research", "cotton286", "dataset_audit.json")
    if not os.path.exists(audit_path):
        print(f"Error: {audit_path} not found.")
        sys.exit(1)
        
    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
        
    file_manifest = audit_data["file_manifest"]
    duplicates = audit_data["duplicates"]
    
    # Map each duplicate file to its cluster representative
    dup_map = {}
    for d in duplicates:
        orig = d["original"]
        dup = d["duplicate"]
        # canonicalize path
        orig_norm = os.path.normpath(orig)
        dup_norm = os.path.normpath(dup)
        dup_map[dup_norm] = orig_norm
        
    # Group files by canonical class
    class_groups = defaultdict(list)
    for item in file_manifest:
        norm_p = os.path.normpath(item["file_path"])
        item["norm_path"] = norm_p
        class_groups[item["canonical_class"]].append(item)
        
    # Seed for determinism
    rng = random.Random(42)
    
    split_manifest = {
        "seed": 42,
        "target_ratios": {"train": 0.70, "dev": 0.15, "sealed_test": 0.15},
        "total_images": len(file_manifest),
        "split_counts": {"train": 0, "dev": 0, "sealed_test": 0},
        "per_class_split": {},
        "splits": {
            "train": [],
            "dev": [],
            "sealed_test": []
        }
    }
    
    for cname in sorted(class_groups.keys()):
        items = class_groups[cname]
        
        # Identify duplicate clusters so they stay in the same split
        clusters = []
        seen = set()
        for item in items:
            p = item["norm_path"]
            if p in seen:
                continue
            rep = dup_map.get(p, p)
            # Find all items that share this representative
            cluster = [x for x in items if dup_map.get(x["norm_path"], x["norm_path"]) == rep]
            for x in cluster:
                seen.add(x["norm_path"])
            clusters.append(cluster)
            
        rng.shuffle(clusters)
        
        # Determine split allocations for clusters
        n_clusters = len(clusters)
        n_test = max(1, round(n_clusters * 0.15))
        n_dev = max(1, round(n_clusters * 0.15))
        n_train = n_clusters - n_test - n_dev
        
        test_clusters = clusters[:n_test]
        dev_clusters = clusters[n_test:n_test + n_dev]
        train_clusters = clusters[n_test + n_dev:]
        
        c_train_items = [x for cl in train_clusters for x in cl]
        c_dev_items = [x for cl in dev_clusters for x in cl]
        c_test_items = [x for cl in test_clusters for x in cl]
        
        split_manifest["per_class_split"][cname] = {
            "total": len(items),
            "train": len(c_train_items),
            "dev": len(c_dev_items),
            "sealed_test": len(c_test_items)
        }
        
        for item in c_train_items:
            split_manifest["splits"]["train"].append(item)
        for item in c_dev_items:
            split_manifest["splits"]["dev"].append(item)
        for item in c_test_items:
            split_manifest["splits"]["sealed_test"].append(item)
            
    split_manifest["split_counts"]["train"] = len(split_manifest["splits"]["train"])
    split_manifest["split_counts"]["dev"] = len(split_manifest["splits"]["dev"])
    split_manifest["split_counts"]["sealed_test"] = len(split_manifest["splits"]["sealed_test"])
    
    out_path = os.path.join("research", "cotton286", "cotton_split_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
        
    print(f"Created split manifest: {out_path}")
    print(f"Total: {split_manifest['total_images']} | Train: {split_manifest['split_counts']['train']} | Dev: {split_manifest['split_counts']['dev']} | Sealed Test: {split_manifest['split_counts']['sealed_test']}")
    for c, counts in split_manifest["per_class_split"].items():
        print(f"  {c}: Train={counts['train']}, Dev={counts['dev']}, Sealed Test={counts['sealed_test']} (Total={counts['total']})")

if __name__ == "__main__":
    create_split()
