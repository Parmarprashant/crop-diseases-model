"""
Create PlantDoc Class Mapping Manifest.
Maps all 28 PlantDoc classes to AgriVision AI's 167 canonical classes
and 281 raw model classes.
"""
import os
import json

def create_mapping():
    with open("weights/class_names.txt", "r") as f:
        raw_classes = [line.strip() for line in f if line.strip()]

    with open("weights/canonical_class_map.json", "r") as f:
        cmap = json.load(f)

    # Map each canonical class to its primary raw index
    canon_to_raw_idx = {}
    for idx, r in enumerate(raw_classes):
        c = cmap.get(r, r)
        if c not in canon_to_raw_idx:
            canon_to_raw_idx[c] = idx

    train_dir = "plantDoc/train"
    test_dir = "plantDoc/test"

    # Botanical definitions for all 28 PlantDoc classes
    plantdoc_definitions = {
        "Apple_Scab_Leaf": {
            "canonical_label": "Apple - Scab",
            "crop_family": "apple",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Venturia inaequalis olive-brown lesions on apple foliage"
        },
        "Apple_leaf": {
            "canonical_label": "Apple - Healthy",
            "crop_family": "apple",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy apple foliage"
        },
        "Apple_rust_leaf": {
            "canonical_label": "Apple - Rust",
            "crop_family": "apple",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Gymnosporangium juniperi-virginianae cedar apple rust"
        },
        "Bell_pepper_leaf": {
            "canonical_label": "Bell Pepper - Healthy",
            "crop_family": "bell pepper",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy bell pepper foliage"
        },
        "Bell_pepper_leaf_spot": {
            "canonical_label": "Bell Pepper - Bacterial Spot",
            "crop_family": "bell pepper",
            "mapping_status": "VALID_SYNONYM",
            "mapping_confidence": 0.95,
            "notes": "Xanthomonas campestris pv. vesicatoria bacterial leaf spot on Capsicum"
        },
        "Blueberry_leaf": {
            "canonical_label": "Blueberry - Healthy",
            "crop_family": "blueberry",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy blueberry foliage"
        },
        "Cherry_leaf": {
            "canonical_label": "Cherry - Healthy",
            "crop_family": "cherry",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy cherry foliage"
        },
        "Corn_Gray_leaf_spot": {
            "canonical_label": "Corn - Gray Leaf Spot",
            "crop_family": "corn",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Cercospora zeae-maydis on maize foliage"
        },
        "Corn_leaf_blight": {
            "canonical_label": "Corn - Northern Leaf Blight",
            "crop_family": "corn",
            "mapping_status": "VALID_SYNONYM",
            "mapping_confidence": 0.95,
            "notes": "Helminthosporium turcicum / Exserohilum turcicum leaf blight on maize"
        },
        "Corn_rust_leaf": {
            "canonical_label": "Corn - Rust",
            "crop_family": "corn",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Puccinia sorghi common rust on maize"
        },
        "Peach_leaf": {
            "canonical_label": "Peach - Healthy",
            "crop_family": "peach",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy peach foliage"
        },
        "Potato_leaf_early_blight": {
            "canonical_label": "Potato - Early Blight",
            "crop_family": "potato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Alternaria solani concentric ring foliar lesions on potato"
        },
        "Potato_leaf_late_blight": {
            "canonical_label": "Potato - Late Blight",
            "crop_family": "potato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Phytophthora infestans water-soaked necrosis on potato"
        },
        "Raspberry_leaf": {
            "canonical_label": "Raspberry - Healthy",
            "crop_family": "raspberry",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy raspberry foliage"
        },
        "Soyabean_leaf": {
            "canonical_label": "Soybean - Healthy",
            "crop_family": "soybean",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy soybean foliage"
        },
        "Squash_Powdery_mildew_leaf": {
            "canonical_label": "Squash - Powdery Mildew",
            "crop_family": "squash",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Podosphaera xanthii white epiphytic fungal mycelium on cucurbit foliage"
        },
        "Strawberry_leaf": {
            "canonical_label": "Strawberry - Healthy",
            "crop_family": "strawberry",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy strawberry foliage"
        },
        "Tomato_Early_blight_leaf": {
            "canonical_label": "Tomato - Early Blight",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Alternaria solani foliar target lesions on tomato"
        },
        "Tomato_Septoria_leaf_spot": {
            "canonical_label": "Tomato - Septoria Leaf Spot",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Septoria lycopersici circular lesions on tomato"
        },
        "Tomato_leaf": {
            "canonical_label": "Tomato - Healthy",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy tomato foliage"
        },
        "Tomato_leaf_bacterial_spot": {
            "canonical_label": "Tomato - Bacterial Spot",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Xanthomonas campestris pv. vesicatoria on tomato"
        },
        "Tomato_leaf_late_blight": {
            "canonical_label": "Tomato - Late Blight",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Phytophthora infestans foliar necrosis on tomato"
        },
        "Tomato_leaf_mosaic_virus": {
            "canonical_label": "Tomato - Mosaic Virus",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Tomato mosaic tobamovirus mottling and distortion"
        },
        "Tomato_leaf_yellow_virus": {
            "canonical_label": "Tomato - Yellow Leaf Curl Virus",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Tomato yellow leaf curl begomovirus stunting and chlorosis"
        },
        "Tomato_mold_leaf": {
            "canonical_label": "Tomato - Leaf Mold",
            "crop_family": "tomato",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Passalora fulva olive-green velvety mold on lower leaf surface"
        },
        "Tomato_two_spotted_spider_mites_leaf": {
            "canonical_label": "Tomato - Spider Mites",
            "crop_family": "tomato",
            "mapping_status": "VALID_SYNONYM",
            "mapping_confidence": 0.95,
            "notes": "Tetranychus urticae stippling and webbing on tomato foliage"
        },
        "grape_leaf": {
            "canonical_label": "Grape - Healthy",
            "crop_family": "grape",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Asymptomatic healthy grape foliage"
        },
        "grape_leaf_black_rot": {
            "canonical_label": "Grape - Black Rot",
            "crop_family": "grape",
            "mapping_status": "EXACT",
            "mapping_confidence": 1.0,
            "notes": "Guignardia bidwellii necrotic foliar lesions with pycnidia on grape"
        }
    }

    manifest = {}
    for plantdoc_cls, info in plantdoc_definitions.items():
        c_label = info["canonical_label"]
        raw_idx = canon_to_raw_idx.get(c_label)
        if raw_idx is None:
            print(f"ERROR: Canonical label '{c_label}' not found in raw classes!")
            continue

        train_p = os.path.join(train_dir, plantdoc_cls)
        test_p = os.path.join(test_dir, plantdoc_cls)

        train_cnt = len([f for f in os.listdir(train_p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(train_p) else 0
        test_cnt = len([f for f in os.listdir(test_p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(test_p) else 0

        manifest[plantdoc_cls] = {
            "plantdoc_label": plantdoc_cls,
            "canonical_label": c_label,
            "crop_family": info["crop_family"],
            "raw_class_index": raw_idx,
            "raw_class_name": raw_classes[raw_idx],
            "mapping_status": info["mapping_status"],
            "mapping_confidence": info["mapping_confidence"],
            "notes": info["notes"],
            "train_image_count": train_cnt,
            "test_image_count": test_cnt
        }

    out_path = "validation/plantdoc_class_mapping.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {out_path} with {len(manifest)} mapped classes.")
    total_train = sum(m["train_image_count"] for m in manifest.values())
    total_test = sum(m["test_image_count"] for m in manifest.values())
    print(f"Total usable PlantDoc images: {total_train} Train, {total_test} Test.")

if __name__ == "__main__":
    create_mapping()
