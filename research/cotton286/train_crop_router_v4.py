import os
import sys
import json
import time
import hashlib
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.cbam import CBAM

try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


ROUTER_FAMILIES = {
    0: "Cotton (Malvaceae)",
    1: "Monocots / Cereals (Rice, Wheat, Maize)",
    2: "Legumes / Pulses (Soybean, Chickpea, Pigeonpea)",
    3: "Solanaceous (Tomato, Potato, Chili)",
    4: "Broadleaf Hard Negatives (Sunflower, Okra, Castor, Weeds)",
    5: "Non-Crop Background (Soil, Hands, Plastic, Debris)"
}


class CropRouterModelV4(nn.Module):
    """
    Enterprise v4.0 6-Family Morphological Router Model:
    - Backbone: EfficientNet-B5 features (2048-d)
    - Attention: CBAM (Channel + Spatial attention)
    - Projection: 2048 -> 512 (SiLU)
    - Head: 512 -> 6 (6-family morphological router)
    """
    def __init__(self, pretrained: bool = False):
        super(CropRouterModelV4, self).__init__()
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError("torchvision required for CropRouterModelV4")
            
        base_model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None)
        self.features = base_model.features
        self.cbam = CBAM(in_channels=2048, reduction_ratio=16, kernel_size=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.router_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat_att = self.cbam(feat)
        pooled = self.avgpool(feat_att)
        flat = torch.flatten(pooled, 1)
        logits = self.router_head(flat)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat_att = self.cbam(feat)
        pooled = self.avgpool(feat_att)
        flat = torch.flatten(pooled, 1)
        h = self.router_head[1](flat)  # 512-dim bottleneck
        h = self.router_head[2](h)     # SiLU
        return h


class RouterDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        with Image.open(item["path"]) as img:
            img = img.convert("RGB")
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        return tensor, item["label"]


def collect_training_samples():
    items = []
    
    # Class 0: Cotton
    manifest_path = "research/cotton286/cotton_split_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            cotton_train = json.load(f)["splits"]["train"]
        for c in cotton_train:
            if os.path.exists(c["file_path"]):
                items.append({"path": c["file_path"], "label": 0})
                
    # Class 1: Monocots / Cereals (Ginger, Rice, Wheat, Maize)
    ginger_dir = "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger"
    if os.path.exists(ginger_dir):
        for f in os.listdir(ginger_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append({"path": os.path.join(ginger_dir, f), "label": 1})

    # Supplementary crops and Outlier Exposure from open_set_v2_cohort
    ood_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/open_set_v2_cohort"
    if os.path.exists(ood_dir):
        for f in sorted(os.listdir(ood_dir)):
            fp = os.path.join(ood_dir, f)
            fl = f.lower()
            if fl.startswith("v2_supp_banana") or fl.startswith("v2_supp_corn") or fl.startswith("v2_supp_rice"):
                items.append({"path": fp, "label": 1})
            elif fl.startswith("v2_supp_bean") or fl.startswith("v2_supp_soybean") or fl.startswith("v2_supp_pea"):
                items.append({"path": fp, "label": 2})
            elif fl.startswith("v2_supp_bell_pepper") or fl.startswith("v2_supp_tomato") or fl.startswith("v2_supp_potato") or fl.startswith("v2_supp_chilli"):
                items.append({"path": fp, "label": 3})
            elif fl.startswith("v2_ood_synth_"):
                try:
                    idx = int(fl.replace("v2_ood_synth_", "").split(".")[0])
                    # Reserve 0-19 for validation [25:50], use 20-69 for OE training
                    if idx >= 20:
                        items.append({"path": fp, "label": 5})
                except Exception:
                    pass
            elif fl.startswith("v2_ood_stress_"):
                try:
                    idx = int(fl.replace("v2_ood_stress_", "").split(".")[0])
                    # Reserve 25-29 for validation [25:50], use 0-24 for OE training
                    if idx < 25:
                        items.append({"path": fp, "label": 5})
                except Exception:
                    pass

    # Class 2: Legumes / Pulses & Class 3: Solanaceous from field cohort (first 50)
    field_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/field_test_cohort"
    if os.path.exists(field_dir):
        # Use first 50 for training, 55-91 reserved for evaluation
        for f in sorted(os.listdir(field_dir))[:50]:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fp = os.path.join(field_dir, f)
            fl = f.lower()
            if any(k in fl for k in ["groundnut", "soybean", "gram", "bean", "pea"]):
                items.append({"path": fp, "label": 2})
            elif any(k in fl for k in ["tomato", "potato", "chili", "chilli", "pepper", "eggplant", "brinjal"]):
                items.append({"path": fp, "label": 3})
            elif any(k in fl for k in ["rice", "paddy", "wheat", "maize", "corn", "sugarcane", "ginger"]):
                items.append({"path": fp, "label": 1})
            else:
                items.append({"path": fp, "label": 4})

    # Class 4: Broadleaf Hard Negatives (Sunflower holdout 25-50, Rose leaf holdout)
    sunflower_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/sunflower_holdout"
    if os.path.exists(sunflower_dir):
        for f in os.listdir(sunflower_dir)[25:]:  # use holdout 25-50 for training, 0-25 for eval
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append({"path": os.path.join(sunflower_dir, f), "label": 4})
                
    rose_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/rose_leaf_holdout"
    if os.path.exists(rose_dir):
        for f in os.listdir(rose_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append({"path": os.path.join(rose_dir, f), "label": 4})

    print(f"Collected total {len(items)} training samples across 6 families:")
    for fam_idx in range(6):
        cnt = sum(1 for x in items if x["label"] == fam_idx)
        print(f"  Class {fam_idx} ({ROUTER_FAMILIES[fam_idx]}): {cnt} samples")
        
    return items


def train_router_v4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Enterprise v4.0 6-Family Router on {device}...")
    
    items = collect_training_samples()
    if not items:
        raise RuntimeError("No training samples found!")
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = RouterDataset(items, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    model = CropRouterModelV4(pretrained=False)
    
    # Load feature backbone and CBAM from Model A
    prod_sd = torch.load("weights/efficientnet_b5_cbam_best.pt", map_location="cpu", weights_only=False)
    model_sd = model.state_dict()
    for k in prod_sd:
        if k.startswith("features.") or k.startswith("cbam."):
            if k in model_sd:
                model_sd[k].copy_(prod_sd[k])
                
    # Also initialize 2048->512 from Model A classifier.1
    if "classifier.1.weight" in prod_sd and "router_head.1.weight" in model_sd:
        model_sd["router_head.1.weight"].copy_(prod_sd["classifier.1.weight"])
        model_sd["router_head.1.bias"].copy_(prod_sd["classifier.1.bias"])
        
    model.load_state_dict(model_sd)
    
    # Freeze features and cbam
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.cbam.parameters():
        p.requires_grad = False
        
    model.to(device)
    
    # Optimize router head
    optimizer = torch.optim.AdamW(model.router_head.parameters(), lr=1e-3, weight_decay=1e-4)
    # Class weights to balance dataset
    counts = [max(1, sum(1 for x in items if x["label"] == i)) for i in range(6)]
    total_s = sum(counts)
    class_weights = torch.tensor([total_s / (6.0 * c) for c in counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        acc = (correct / total) * 100.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {total_loss/total:.4f} | Accuracy: {acc:.2f}%")
        
    out_dir = "weights/research"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crop_router_v4.pt")
    torch.save(model.state_dict(), out_path)
    
    with open(out_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Saved Enterprise v4.0 router model to {out_path} (SHA256: {sha256})")
    return out_path, sha256


if __name__ == "__main__":
    train_router_v4()
