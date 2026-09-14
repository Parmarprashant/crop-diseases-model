import os
import re
import json

CATEGORY_A_EXPLICIT_CONSOLIDATIONS = {
    # Wheat Black Chaff & Septoria redundant scrape tokens
    "Wheat - Bacterial Leaf Streak (Black Chaff) Black Chaff": "Wheat - Bacterial Leaf Streak (Black Chaff)",
    "Wheat - Bacterial Leaf Streak (Black Chaff)Black Chaff": "Wheat - Bacterial Leaf Streak (Black Chaff)",
    "Wheat - Septoria Blotch Blotch": "Wheat - Septoria Blotch",
    "Wheat - Septoria Blotchseptoria": "Wheat - Septoria Blotch",
    # Banana synonyms & sigatoka complex
    "Banana - Cordana": "Banana - Cordana Leaf Spot",
    "Banana - Panama": "Banana - Panama Disease",
    "Banana - Black Leaf Streak Banana Black Sigatoka": "Banana - Black Sigatoka",
    "Banana - Black Leaf Streak": "Banana - Black Sigatoka",
    "Banana - Sigatoka": "Banana - Yellow Sigatoka",
    "Banana - Yb Sigatoka": "Banana - Yellow Sigatoka",
    # Tomato & Cauliflower
    "Tomato - Bacterial Leaf Spot": "Tomato - Bacterial Spot",
    "Cauliflower - Bacterial  Spot  Rot": "Cauliflower - Bacterial Soft Rot",
    "Cauliflower - Bacterial Spot Rot": "Cauliflower - Bacterial Soft Rot",
    "Cauliflower - Blackrot": "Cauliflower - Black Rot",
    # Corn blight
    "Corn - Leaf Blight": "Corn - Northern Leaf Blight",
    # Cleanups & Typos
    "Chilli - Leafspot": "Chilli - Leaf Spot",
    "Chilli - Leafcurl": "Chilli - Leaf Curl",
    "Radish - Downey Mildew": "Radish - Downy Mildew",
    "Groundnut - Early Rust": "Groundnut - Rust",
}


def normalize_class_name(raw_name: str) -> str:
    name = raw_name.strip()
    
    # Remove known scrape source suffixes and trailing artifacts
    source_suffixes = [
        r"\s+Bing$", r"\s+Baidu$", r"\s+Google$",
        r"\s+plantvillage$", r"\s+PlantVillage$",
        r"\s+\(train\)$", r"\s+\(val\)$",
        r"\s+dataset$", r"\s+Dataset$",
        r"\s+\(\)$",
    ]
    for pattern in source_suffixes:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

    # Handle PlantVillage triple-underscore format: "Tomato___Early_blight"
    if "___" in name:
        parts = name.split("___")
        crop = parts[0].strip().replace("_", " ").title()
        disease = parts[1].strip().replace("_", " ").title() if len(parts) > 1 else "Healthy"
        disease = re.sub(r"^" + re.escape(crop) + r"\s+", "", disease, flags=re.IGNORECASE)
        name = f"{crop} - {disease}"

    # Handle "Crop - crop disease name" redundancy
    # e.g. "Tomato - tomato early blight" → "Tomato - Early Blight"
    if " - " in name:
        parts = name.split(" - ", 1)
        crop_part = parts[0].strip().title()
        disease_part = parts[1].strip()
        # Remove leading crop name from disease description if present
        disease_part = re.sub(
            r"^" + re.escape(crop_part.lower()) + r"\s+",
            "", disease_part, flags=re.IGNORECASE
        )
        disease_part = disease_part.strip().title()
        # Standardize "Healthy" variants
        if disease_part.lower() in ["healthy", "health", "no disease", "normal"]:
            disease_part = "Healthy"
        name = f"{crop_part} - {disease_part}"
    elif " in " in name:
        # e.g. "bacterial_blight in Cotton" -> "Cotton - Bacterial Blight"
        parts = name.split(" in ", 1)
        disease_part = parts[0].replace("_", " ").strip().title()
        crop_part = parts[1].replace("_", " ").strip().title()
        name = f"{crop_part} - {disease_part}"
    elif " on " in name:
        # e.g. "Anthracnose on Cotton" -> "Cotton - Anthracnose"
        parts = name.split(" on ", 1)
        disease_part = parts[0].replace("_", " ").strip().title()
        crop_part = parts[1].replace("_", " ").strip().title()
        name = f"{crop_part} - {disease_part}"

    # Fix multiple whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # Apply Category A explicit consolidations
    if name in CATEGORY_A_EXPLICIT_CONSOLIDATIONS:
        name = CATEGORY_A_EXPLICIT_CONSOLIDATIONS[name]

    return name.strip()


def build_maps():
    class_names_path = "weights/class_names.txt"
    with open(class_names_path, "r", encoding="utf-8") as f:
        raw_classes = [line.strip() for line in f if line.strip()]

    print(f"Total raw classes: {len(raw_classes)}")

    canonical_map = {}
    canonical_to_raw = {}

    for raw_name in raw_classes:
        canonical = normalize_class_name(raw_name)
        canonical_map[raw_name] = canonical
        
        if canonical not in canonical_to_raw:
            canonical_to_raw[canonical] = []
        canonical_to_raw[canonical].append(raw_name)

    # Ensure weights directory exists
    os.makedirs("weights", exist_ok=True)

    # Save forward map (raw -> canonical)
    forward_path = "weights/canonical_class_map.json"
    with open(forward_path, "w", encoding="utf-8") as f:
        json.dump(canonical_map, f, indent=2, ensure_ascii=False)
    print(f"Saved forward map to {forward_path}")

    # Save reverse map (canonical -> list of raw)
    reverse_path = "weights/canonical_reverse_map.json"
    with open(reverse_path, "w", encoding="utf-8") as f:
        json.dump(canonical_to_raw, f, indent=2, ensure_ascii=False)
    print(f"Saved reverse map to {reverse_path}")

    print(f"Raw classes count      : {len(raw_classes)}")
    print(f"Canonical classes count: {len(canonical_to_raw)}")

    # Show example merged groups
    merged_groups = {k: v for k, v in canonical_to_raw.items() if len(v) > 1}
    print(f"Number of collapsed/merged disease groups: {len(merged_groups)}")
    print("\nSample collapsed groups:")
    for canonical, raws in list(merged_groups.items())[:8]:
        print(f"  '{canonical}':")
        for r in raws:
            print(f"    <- '{r}'")


if __name__ == "__main__":
    build_maps()
