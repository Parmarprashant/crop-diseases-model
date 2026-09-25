import os
import sys
import json
import time
import hashlib
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import CropRouterModel


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


def train_router():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Dedicated Crop Router on {device}...")
    
    # 1. Prepare Fitting Data
    with open("research/cotton286/cotton_split_manifest.json", "r", encoding="utf-8") as f:
        cotton_split = json.load(f)
    cotton_train = cotton_split["splits"]["train"]  # 135 images
    
    items = []
    for item in cotton_train:
        items.append({"path": item["file_path"], "label": 1})  # 1 = Cotton
        
    # Ginger images (50)
    ginger_dir = "D:/1winbackup/desktop/Ganpat University/Model/data/curated_ginger"
    if os.path.exists(ginger_dir):
        for f in os.listdir(ginger_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                items.append({"path": os.path.join(ginger_dir, f), "label": 0})  # 0 = Non-Cotton
                
    # First 30 Field samples
    field_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/field_test_cohort"
    if os.path.exists(field_dir):
        for f in sorted(os.listdir(field_dir))[:30]:
            if f.endswith(('.jpg', '.jpeg', '.png')):
                items.append({"path": os.path.join(field_dir, f), "label": 0})  # 0 = Non-Cotton
                
    cotton_count = sum(1 for x in items if x["label"] == 1)
    non_cotton_count = sum(1 for x in items if x["label"] == 0)
    print(f"Fitting set size: {len(items)} (Cotton: {cotton_count}, Non-Cotton: {non_cotton_count})")
    
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
    
    # 2. Build model and load pre-trained features from Model A
    model = CropRouterModel(pretrained=False)
    prod_sd = torch.load("weights/efficientnet_b5_cbam_best.pt", map_location="cpu", weights_only=False)
    
    # Copy features and cbam weights from Model A
    model_sd = model.state_dict()
    for k in prod_sd:
        if k.startswith("features.") or k.startswith("cbam."):
            if k in model_sd:
                model_sd[k].copy_(prod_sd[k])
    model.load_state_dict(model_sd)
    
    # Freeze features and cbam
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.cbam.parameters():
        p.requires_grad = False
        
    model.to(device)
    
    # Optimize only router head
    optimizer = torch.optim.AdamW(model.router_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
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
    out_path = os.path.join(out_dir, "crop_router_v1.pt")
    torch.save(model.state_dict(), out_path)
    
    with open(out_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Saved router model to {out_path} (SHA256: {sha256})")
    return out_path, sha256


if __name__ == "__main__":
    train_router()
