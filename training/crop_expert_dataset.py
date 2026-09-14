"""
AgriVision AI — Dedicated Crop Expert Dataset & DataLoader with Hard-Negative Mining.

Loads images from Data/master_images/master_images/images/ and maps each sample
to a verified canonical crop family index (0 to 40, matching weights/crop_names.txt).

Enforces:
1. Strict pre-training taxonomy assertion (must match weights/crop_names.txt exactly).
2. Inverse square-root class frequency sampling to resolve 280:1 class imbalance.
3. Hard-negative boost for dominant confusing crop pairs identified in forensic audit.
4. Field-robust data augmentation pipeline.
"""
import os
import json
import pandas as pd
from typing import Tuple, List, Optional, Dict
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Dominant confusing crop pairs from forensic audit (validation/crop_confusion_matrix.json)
FORENSIC_HARD_NEGATIVE_PAIRS = [
    ("raspberry", "strawberry"),
    ("apple", "banana"),
    ("grape", "squash"),
    ("corn", "wheat"),
    ("chilli", "tomato"),
    ("apple", "citrus"),
    ("citrus", "peach"),
    ("apple", "peach"),
    ("bean", "soybean"),
    ("paddy", "wheat"),
    ("rice", "wheat"),
    ("potato", "tomato"),
    ("blackgram", "soybean"),
    ("broccoli", "cabbage")
]

class CropExpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        master_images_dir: str = "Data/master_images/master_images/images",
        crop_names_path: str = "weights/crop_names.txt",
        transform = None
    ):
        self.csv_path = csv_path
        self.master_images_dir = master_images_dir
        self.transform = transform

        # Load canonical crop list
        with open(crop_names_path, "r", encoding="utf-8") as f:
            self.crop_names = [l.strip().lower() for l in f if l.strip()]
        self.num_crop_classes = len(self.crop_names)
        self.crop_to_idx = {c: i for i, c in enumerate(self.crop_names)}

        # Strict Pre-Training Taxonomy Assertion
        df = pd.read_csv(csv_path)
        csv_crops = set(df["crop"].str.lower().str.strip().unique())
        tax_crops = set(self.crop_names)

        diff_csv_tax = csv_crops - tax_crops
        if diff_csv_tax:
            raise ValueError(
                f"[PRE-TRAINING TAXONOMY ASSERTION FAILED] The following crops in {csv_path} "
                f"are not present in {crop_names_path}: {diff_csv_tax}"
            )

        # Index master directory case-insensitively
        files = os.listdir(master_images_dir)
        self.available_files = {f.lower(): f for f in files}

        self.samples = []
        self.crop_indices = []
        self.crop_strings = []

        for idx, row in df.iterrows():
            img_id = str(row["image_id"]).lower()
            crop_name = str(row["crop"]).lower().strip()

            crop_idx = self.crop_to_idx[crop_name]

            # Resolve physical path
            phys_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = f"{img_id}{ext}"
                if cand in self.available_files:
                    phys_file = os.path.join(master_images_dir, self.available_files[cand])
                    break

            if phys_file and os.path.exists(phys_file):
                self.samples.append((phys_file, crop_idx))
                self.crop_indices.append(crop_idx)
                self.crop_strings.append(crop_name)

        print(
            f"[CropExpertDataset] Loaded {len(self.samples):,} valid samples from {csv_path} "
            f"across {self.num_crop_classes} canonical crops."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, crop_idx = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (256, 256), color=0)

        if self.transform:
            image = self.transform(image)

        return image, crop_idx

    def get_sampling_weights(self, hard_negative_boost: float = 1.5) -> torch.Tensor:
        """
        Computes sampling weights combining:
        1. Inverse square-root frequency weighting: w = 1 / sqrt(count)
        2. Hard-negative oversampling boost for confusing crop pairs.
        """
        from collections import Counter
        counts = Counter(self.crop_indices)

        # Identify crops involved in forensic hard-negative pairs
        hard_crops = set()
        for c1, c2 in FORENSIC_HARD_NEGATIVE_PAIRS:
            hard_crops.add(c1)
            hard_crops.add(c2)

        weights = []
        for c_idx, c_str in zip(self.crop_indices, self.crop_strings):
            # Base inverse square-root frequency weight
            w = 1.0 / (counts[c_idx] ** 0.5)

            # Apply hard-negative pair boost
            if c_str in hard_crops:
                w *= hard_negative_boost

            weights.append(w)

        return torch.tensor(weights, dtype=torch.float)

def get_crop_expert_transforms():
    """Builds field-robust train and validation transformations."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256, padding=12, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=20, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_crop_expert_loaders(
    train_csv: str = "Data/outputs/outputs/train_split.csv",
    val_csv: str = "Data/outputs/outputs/val_split.csv",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, CropExpertDataset, CropExpertDataset]:
    """Instantiates balanced train and validation DataLoaders for Crop Expert."""
    train_transform, val_transform = get_crop_expert_transforms()

    train_dataset = CropExpertDataset(csv_path=train_csv, transform=train_transform)
    val_dataset = CropExpertDataset(csv_path=val_csv, transform=val_transform)

    # Class-balanced + hard-negative sampler for training
    train_weights = train_dataset.get_sampling_weights(hard_negative_boost=1.5)
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, train_dataset, val_dataset
