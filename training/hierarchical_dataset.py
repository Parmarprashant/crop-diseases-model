"""
AgriVision AI — Hierarchical Multi-Task Dataset & Balanced DataLoader.
Loads images from Data/master_images/master_images/images/ and maps each sample
to both:
1. Disease label index (0 to 280, matching weights/class_names.txt)
2. Crop family label index (0 to 40, matching weights/crop_names.txt)
Implements crop-aware inverse-frequency sampling to ensure minority crop representation.
"""
import os
import pandas as pd
from typing import Tuple, List, Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class HierarchicalCropDiseaseDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        master_images_dir: str = "Data/master_images/master_images/images",
        class_names_path: str = "weights/class_names.txt",
        crop_names_path: str = "weights/crop_names.txt",
        transform = None
    ):
        self.csv_path = csv_path
        self.master_images_dir = master_images_dir
        self.transform = transform
        
        # Load mappings
        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = [l.strip() for l in f if l.strip()]
        self.num_disease_classes = len(self.class_names)
        
        with open(crop_names_path, "r", encoding="utf-8") as f:
            self.crop_names = [l.strip() for l in f if l.strip()]
        self.crop_to_idx = {c: i for i, c in enumerate(self.crop_names)}
        self.num_crop_classes = len(self.crop_names)

        # Index master directory case-insensitively
        files = os.listdir(master_images_dir)
        self.available_files = {f.lower(): f for f in files}

        df = pd.read_csv(csv_path)
        self.samples = []
        self.crop_indices = []

        for idx, row in df.iterrows():
            img_id = str(row["image_id"]).lower()
            disease_idx = int(row["class_id"])
            crop_name = str(row["crop"]).lower().strip()
            
            # Map crop
            crop_idx = self.crop_to_idx.get(crop_name)
            if crop_idx is None:
                continue

            # Resolve physical path
            phys_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = f"{img_id}{ext}"
                if cand in self.available_files:
                    phys_file = os.path.join(master_images_dir, self.available_files[cand])
                    break
            
            if phys_file and os.path.exists(phys_file):
                self.samples.append((phys_file, disease_idx, crop_idx))
                self.crop_indices.append(crop_idx)

        print(f"[HierarchicalDataset] Loaded {len(self.samples):,} valid physical samples from {csv_path} across {self.num_crop_classes} crops.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path, disease_idx, crop_idx = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (256, 256), color=0)

        if self.transform:
            image = self.transform(image)

        return image, disease_idx, crop_idx

    def get_crop_weights(self) -> torch.Tensor:
        """Calculate inverse square-root frequency weights for crop-balanced sampling."""
        from collections import Counter
        counts = Counter(self.crop_indices)
        weights = []
        for c_idx in self.crop_indices:
            w = 1.0 / (counts[c_idx] ** 0.5)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float)

def get_hierarchical_loaders(
    train_csv: str = "Data/outputs/outputs/train_split.csv",
    val_csv: str = "Data/outputs/outputs/val_split.csv",
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = 256
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = HierarchicalCropDiseaseDataset(csv_path=train_csv, transform=train_transform)
    val_ds = HierarchicalCropDiseaseDataset(csv_path=val_csv, transform=val_transform)

    # Crop-balanced sampler for training
    sample_weights = train_ds.get_crop_weights()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
