"""
Pseudo-mask generation for U-Net lesion segmentation training.

Since no human-annotated segmentation masks exist, this script generates
pseudo-labels via HSV color thresholding:
  - Leaf mask: green-dominant vegetation pixels
  - Lesion mask: necrotic brown / yellow / dark-spot pixels inside the leaf

Masks are written as PNGs to <output_dir>/masks mirroring the Train class structure.
Only disease classes (not Healthy) are processed.
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MASK_SIZE = 256

def generate_leaf_lesion_masks(rgb: np.ndarray) -> tuple:
    """Return (leaf_mask, lesion_mask) uint8 arrays for an RGB image."""
    img = cv2.resize(rgb, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)

    # --- Leaf mask: green vegetation pixels (broad HSV range for field photos) ---
    leaf = (
        ((h >= 20) & (h <= 95) & (s > 30) & (v > 30)) |
        ((g > r * 0.9) & (g > b * 1.05) & (g > 50))
    ).astype(np.uint8)

    # Clean leaf mask: close holes, drop specks
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # --- Lesion mask: necrotic pixels inside leaf ---
    diff_rg = np.abs(r - g)
    diff_gb = np.abs(g - b)
    brown_necrotic = (r > 100) & (r >= g) & (g >= b * 0.8) & (diff_rg > 20)
    yellowing = (r > 130) & (g > 120) & (b < 110) & (diff_rg < 60) & (diff_gb > 30)
    dark_spot = (r < 75) & (g < 75) & (b < 75) & (s > 0)
    lesion = (brown_necrotic | yellowing | dark_spot).astype(np.uint8)

    # Lesions must lie inside the leaf
    lesion = lesion & leaf

    # Clean lesion mask
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Fill leaf interior holes (lesion can sit inside leaf holes legitimately, so
    # fill holes on a copy, then re-apply lesion)
    leaf_filled = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    leaf = leaf_filled | lesion
    return leaf.astype(np.uint8), lesion.astype(np.uint8)


def process_split(train_dir: str, output_root: str, per_class_cap: int, healthy_cap: int):
    """Walk Train classes; write masks to output_root mirroring structure."""
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    total = 0
    for cls in classes:
        src = os.path.join(train_dir, cls)
        files = sorted([f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        dst = os.path.join(output_root, cls)
        os.makedirs(dst, exist_ok=True)

        is_healthy = "healthy" in cls.lower()
        cap = healthy_cap if is_healthy else per_class_cap
        if len(files) > cap:
            files = random.Random(42).sample(files, cap)

        n_cls = 0
        for fname in files:
            try:
                img = np.array(Image.open(os.path.join(src, fname)).convert("RGB"))
                leaf, lesion = generate_leaf_lesion_masks(img)
                # Skip degenerate masks (no leaf visible)
                if leaf.sum() < 200:
                    continue
                # Combine into single 2-class PNG: 0=background, 1=leaf, 2=lesion
                combined = np.maximum(leaf * 1, lesion * 2).astype(np.uint8)
                out_name = os.path.splitext(fname)[0] + "_mask.png"
                Image.fromarray(combined).save(os.path.join(dst, out_name))
                n_cls += 1
            except Exception as e:
                print(f"  [skip] {cls}/{fname}: {e}")
        total += n_cls
        print(f"  {cls}: {n_cls} masks")
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo segmentation masks for U-Net training")
    parser.add_argument("--data_dir", type=str, default="MAIN DATA")
    parser.add_argument("--output_dir", type=str, default="MAIN DATA/SegMasks")
    parser.add_argument("--per_class_cap", type=int, default=400, help="Max images per disease class")
    parser.add_argument("--healthy_cap", type=int, default=150, help="Max images per healthy class")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "Train")
    if not os.path.exists(train_dir):
        print(f"[Error] Train directory not found: {train_dir}")
        sys.exit(1)

    print(f"[PseudoMask] Generating masks from: {train_dir}")
    print(f"[PseudoMask] Output: {args.output_dir}")
    total = process_split(train_dir, args.output_dir, args.per_class_cap, args.healthy_cap)
    print(f"[PseudoMask] DONE. {total} masks written to {args.output_dir}")


if __name__ == "__main__":
    main()
