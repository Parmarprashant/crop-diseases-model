import os
from typing import Tuple, List, Optional
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms, datasets
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class MainDataCropDataset(Dataset):
    """
    Custom PyTorch Dataset sourcing imagery strictly from the local 'MAIN DATA' directory.
    Supports:
    1. Directory classification mode (MAIN DATA/Train & MAIN DATA/Validation)
    2. CSV mode (MAIN DATA/train.csv & MAIN DATA/train_images)
    """
    def __init__(
        self,
        root_dir: str = "MAIN DATA",
        subset: str = "Train",
        transform = None,
        classes: Optional[List[str]] = None
    ):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = classes or []
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)} if classes else {}
        # Blacklist of corrupt/unreadable files discovered at runtime
        self._bad_files = set()

        self._load_dataset()

    def _load_dataset(self):
        target_dir = os.path.join(self.root_dir, self.subset)
        if os.path.exists(target_dir):
            # Load from folder structure
            if not self.classes:
                dir_classes = sorted([
                    d for d in os.listdir(target_dir)
                    if os.path.isdir(os.path.join(target_dir, d))
                ])
                self.classes = dir_classes
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            norm_to_official = {c.lower().replace("_", " ").strip(): c for c in self.classes}

            for d in os.listdir(target_dir):
                sub_path = os.path.join(target_dir, d)
                if os.path.isdir(sub_path):
                    norm_d = d.lower().replace("_", " ").strip()
                    official_name = norm_to_official.get(norm_d, d)
                    if official_name in self.class_to_idx:
                        cls_idx = self.class_to_idx[official_name]
                        for fname in os.listdir(sub_path):
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                self.samples.append((os.path.join(sub_path, fname), cls_idx))
        else:
            # Fallback to train.csv if directory not matched
            csv_path = os.path.join(self.root_dir, "train.csv")
            img_dir = os.path.join(self.root_dir, "train_images")
            if os.path.exists(csv_path) and os.path.exists(img_dir):
                df = pd.read_csv(csv_path)
                if not self.classes:
                    self.classes = sorted(df['label'].unique().tolist())
                    self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                for _, row in df.iterrows():
                    fpath = os.path.join(img_dir, str(row['image_id']))
                    if os.path.exists(fpath) and row['label'] in self.class_to_idx:
                        self.samples.append((fpath, self.class_to_idx[row['label']]))

    def get_class_counts(self) -> dict:
        """Return per-class sample counts (for balancing & reporting)."""
        counts = {c: 0 for c in self.classes}
        for _, idx in self.samples:
            counts[self.classes[idx]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def _get_random_replacement(self, failed_path: str, failed_target: int, attempt: int = 0):
        """Pick a random readable sample from the same class. Give up after N tries."""
        import random
        if attempt >= 5:
            raise RuntimeError(f"Too many corrupt files in class '{self.classes[failed_target]}'")
        candidates = [s for s in self.samples if s[1] == failed_target and s[0] not in self._bad_files and s[0] != failed_path]
        if not candidates:
            raise RuntimeError(f"No readable samples remain in class '{self.classes[failed_target]}'")
        path, target = random.choice(candidates)
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, target
        except Exception:
            self._bad_files.add(path)
            return self._get_random_replacement(path, target, attempt + 1)

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        if img_path in self._bad_files:
            return self._get_random_replacement(img_path, target)
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception:
            self._bad_files.add(img_path)
            return self._get_random_replacement(img_path, target)


def get_data_loaders(
    root_dir: str = "MAIN DATA",
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 456,
    balance: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Construct high-throughput DataLoaders with agritech augmentations.
    When balance=True, oversamples minority classes via WeightedRandomSampler
    so rare diseases (e.g. 'bollrot on Cotton' with 2 images) get equal exposure.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MainDataCropDataset(root_dir=root_dir, subset="Train", transform=train_transforms)
    classes = train_dataset.classes
    val_dataset = MainDataCropDataset(root_dir=root_dir, subset="Validation", transform=val_transforms, classes=classes)

    train_sampler = None
    shuffle = True
    if balance and TORCH_AVAILABLE:
        class_counts = train_dataset.get_class_counts()
        count_per_class = [class_counts[c] for c in classes]
        # Inverse-frequency weights: rare classes sampled proportionally more often
        weights_per_sample = [1.0 / count_per_class[t] for _, t in train_dataset.samples]
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights_per_sample),
            num_samples=len(train_dataset.samples),
            replacement=True
        )
        shuffle = False
        minority = sorted(class_counts.items(), key=lambda kv: kv[1])[:5]
        print(f"[Balancing] WeightedRandomSampler active. Smallest classes: "
              f"{'; '.join(f'{c}={n}' for c, n in minority)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, classes
