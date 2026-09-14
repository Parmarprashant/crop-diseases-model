"""
PlantDoc Dataset Loader with Field-Oriented Augmentations.
Loads images from plantDoc/train and plantDoc/test and maps each sample
to the 281-dimensional raw class space of the AgriVision model.
"""
import os
import json
from typing import Tuple, List, Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PlantDocDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "plantDoc",
        split: str = "train",
        mapping_file: str = "validation/plantdoc_class_mapping.json",
        transform = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        with open(mapping_file, "r") as f:
            self.mapping = json.load(f)
            
        self.samples: List[Tuple[str, int, str]] = [] # (path, raw_class_idx, canonical_label)
        
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise RuntimeError(f"Split directory {split_dir} does not exist!")
            
        for folder_name in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            info = self.mapping.get(folder_name)
            if not info:
                continue
                
            raw_idx = info["raw_class_index"]
            c_label = info["canonical_label"]
            
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_p = os.path.join(folder_path, fname)
                    self.samples.append((full_p, raw_idx, c_label))
                    
        print(f"[PlantDocDataset] Loaded {len(self.samples)} images from {split_dir} across {len(self.mapping)} mapped classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, raw_idx, _ = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupted
            image = Image.new("RGB", (256, 256), color=0)
            
        if self.transform:
            image = self.transform(image)
            
        return image, raw_idx

def get_plantdoc_loaders(
    root_dir: str = "plantDoc",
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

    train_ds = PlantDocDataset(root_dir=root_dir, split="train", transform=train_transform)
    test_ds = PlantDocDataset(root_dir=root_dir, split="test", transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
