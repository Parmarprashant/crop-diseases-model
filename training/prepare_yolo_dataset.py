"""
Prepare a YOLO-format pest detection dataset from MAIN DATA/Train.

No manual bounding-box annotations exist, so this script generates
pseudo-boxes: for each pest-class image, the entire image is labeled as one
box covering the dominant detected object region. Better than nothing:
it teaches YOLO the pest classes' visual appearance with a "whole image" box.

Improvement path: export real boxes from Roboflow/CVAT into
MAIN DATA/YoloPests and retrain -- train_yolo.py accepts any standard YOLO layout.
"""
import os
import sys
import argparse
import random
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Pest classes from MAIN DATA/Train (diseases & healthy classes excluded)
PEST_CLASSES = [
    "American Bollworm on Cotton", "Army worm", "bollworm on Cotton",
    "Cotton Aphid", "cotton mealy bug", "cotton whitefly",
    "maize fall armyworm", "maize stem borer", "pink bollworm in cotton",
    "red cotton bug", "thirps on  cotton", "Wheat aphid", "Wheat mite",
    "Wheat Stem fly"
]

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def build_dataset(data_dir: str, output_dir: str, per_class_cap: int, val_ratio: float):
    train_dir = os.path.join(data_dir, "Train")
    out_train_img = os.path.join(output_dir, "images", "train")
    out_val_img = os.path.join(output_dir, "images", "val")
    out_train_lbl = os.path.join(output_dir, "labels", "train")
    out_val_lbl = os.path.join(output_dir, "labels", "val")
    for d in (out_train_img, out_val_img, out_train_lbl, out_val_lbl):
        os.makedirs(d, exist_ok=True)

    pairs = []
    for cls_idx, cls in enumerate(PEST_CLASSES):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [warn] pest class folder missing: {cls}")
            continue
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS)])
        if len(files) > per_class_cap:
            files = random.Random(42).sample(files, per_class_cap)
        for f in files:
            pairs.append((os.path.join(cls_dir, f), cls_idx, cls))
        print(f"  {cls}: {len(files)} images (class_id={cls_idx})")

    random.Random(123).shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        img_dir = out_train_img if split == "train" else out_val_img
        lbl_dir = out_train_lbl if split == "train" else out_val_lbl
        for src, cls_idx, cls in items:
            base = f"{cls.replace(' ', '_').lower()}__{os.path.splitext(os.path.basename(src))[0]}"
            shutil.copy2(src, os.path.join(img_dir, base + os.path.splitext(src)[1].lower()))
            # Whole-image pseudo-box in YOLO format: <class> <x_center> <y_center> <w> <h> (normalized)
            with open(os.path.join(lbl_dir, base + ".txt"), "w") as f:
                f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

    return len(pairs), len(PEST_CLASSES)


def write_yaml(output_dir: str, yaml_path: str):
    names = ", ".join(f"{i}: '{c}'" for i, c in enumerate(PEST_CLASSES))
    content = (
        f"path: {os.path.abspath(output_dir)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(PEST_CLASSES)}\n"
        f"names:\n"
        + "\n".join(f"  {i}: '{c}'" for i, c in enumerate(PEST_CLASSES)) + "\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Build YOLO pest detection dataset from MAIN DATA")
    parser.add_argument("--data_dir", type=str, default="MAIN DATA")
    parser.add_argument("--output_dir", type=str, default="MAIN DATA/YoloPests")
    parser.add_argument("--yaml", type=str, default="pest_dataset.yaml")
    parser.add_argument("--per_class_cap", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    print(f"[YOLO Prep] Building dataset -> {args.output_dir}")
    total, nc = build_dataset(args.data_dir, args.output_dir, args.per_class_cap, args.val_ratio)
    write_yaml(args.output_dir, args.yaml)
    print(f"[YOLO Prep] DONE. {total} images, {nc} classes. YAML written to {args.yaml}")


if __name__ == "__main__":
    main()
