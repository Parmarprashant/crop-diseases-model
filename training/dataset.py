import os
import csv
from typing import Tuple, List, Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class MainDataCropDataset(Dataset):
    """
    Custom PyTorch Dataset sourcing imagery from:
    1. CSV split mode (Data/outputs/outputs/{train,val,test}_split.csv & Data/master_images/master_images/images)
    2. Directory classification mode (MAIN DATA/Train & MAIN DATA/Validation)
    3. Legacy CSV mode (MAIN DATA/train.csv & MAIN DATA/train_images)
    """
    def __init__(
        self,
        root_dir: str = "Data",
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
        # 1. First priority: Check for split CSV files (Data/outputs/outputs/{train,val,test}_split.csv)
        split_name = "train_split.csv"
        sub_lower = self.subset.lower()
        if "val" in sub_lower:
            split_name = "val_split.csv"
        elif "test" in sub_lower:
            split_name = "test_split.csv"

        split_candidates = [
            os.path.join(self.root_dir, "outputs", "outputs", split_name),
            os.path.join(self.root_dir, "outputs", split_name),
            os.path.join(self.root_dir, split_name),
            os.path.join(os.path.dirname(os.path.abspath(self.root_dir)), "Data", "outputs", "outputs", split_name),
        ]

        found_split = None
        for cand in split_candidates:
            if os.path.exists(cand):
                found_split = cand
                break

        if found_split:
            # Find images directory
            split_root = os.path.dirname(os.path.dirname(os.path.dirname(found_split))) if "outputs" in found_split else self.root_dir
            img_dir_candidates = [
                os.path.join(self.root_dir, "master_images", "master_images", "images"),
                os.path.join(self.root_dir, "master_images", "images"),
                os.path.join(self.root_dir, "images"),
                os.path.join(split_root, "master_images", "master_images", "images"),
                os.path.join(split_root, "master_images", "images"),
                os.path.join(split_root, "images"),
            ]
            img_dir = None
            for cand in img_dir_candidates:
                if os.path.isdir(cand):
                    img_dir = cand
                    break

            # Find class_id_map.csv if classes not provided
            if not self.classes:
                map_candidates = [
                    os.path.join(self.root_dir, "metadata", "metadata", "class_id_map.csv"),
                    os.path.join(self.root_dir, "metadata", "class_id_map.csv"),
                    os.path.join(self.root_dir, "class_id_map.csv"),
                    os.path.join(split_root, "metadata", "metadata", "class_id_map.csv"),
                ]
                for cand in map_candidates:
                    if os.path.exists(cand):
                        try:
                            class_map = {}
                            with open(cand, "r", encoding="utf-8", errors="replace") as f:
                                for r in csv.DictReader(f):
                                    class_map[int(r["class_id"])] = r["canonical_class"]
                            self.classes = [class_map[i] for i in sorted(class_map.keys())]
                            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                            break
                        except Exception as e:
                            print(f"[Dataset] Notice: Could not parse class_id_map {cand}: {e}")

            with open(found_split, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not self.classes:
                raw_classes = sorted(list({r.get("canonical_class", "") for r in rows if r.get("canonical_class")}))
                self.classes = raw_classes
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

            for r in rows:
                cname = r.get("canonical_class", "")
                cid = self.class_to_idx.get(cname)
                if cid is None and "class_id" in r:
                    try:
                        parsed_cid = int(r["class_id"])
                        if parsed_cid < len(self.classes):
                            cid = parsed_cid
                    except ValueError:
                        pass
                if cid is None:
                    continue

                raw_path = r.get("image_path", "")
                fname = os.path.basename(raw_path)
                full_path = os.path.join(img_dir, fname) if img_dir else os.path.join(self.root_dir, raw_path)
                self.samples.append((full_path, cid))

            print(f"[Dataset] Sourced {len(self.samples)} images across {len(self.classes)} classes from {os.path.basename(found_split)}")
            return

        # 2. Folder structure mode
        target_dir = os.path.join(self.root_dir, self.subset)
        if os.path.exists(target_dir):
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
            return

        # 3. Fallback to train.csv if directory not matched
        csv_path = os.path.join(self.root_dir, "train.csv")
        img_dir = os.path.join(self.root_dir, "train_images")
        if os.path.exists(csv_path) and os.path.exists(img_dir):
            import pandas as pd
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
    root_dir: str = "Data",
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 256,
    balance: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Construct high-throughput DataLoaders with agritech augmentations.
    When balance=True, oversamples minority classes via WeightedRandomSampler
    so rare diseases get balanced exposure.
    """
    if not os.path.exists(root_dir):
        if os.path.exists("Data"):
            root_dir = "Data"
        elif os.path.exists("MAIN DATA"):
            root_dir = "MAIN DATA"

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
        count_per_class = [class_counts.get(c, 0) for c in classes]
        # Inverse-frequency weights with safe floor to avoid div-by-zero
        weights_per_sample = [1.0 / max(count_per_class[t], 1) for _, t in train_dataset.samples]
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
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )

    return train_loader, val_loader, classes
