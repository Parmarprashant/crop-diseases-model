import os
import sys
import json
import time
import hashlib
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM


class CottonRehearsalDataset(Dataset):
    """
    Blended dataset combining:
    1. Cotton training images (5 classes, mapped to indices 281..285)
    2. Rehearsal images from original classes (mapped to indices 0..280)
    """
    def __init__(self, cotton_items, rehearsal_items, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        
        # Add cotton samples
        for item in cotton_items:
            fpath = item["file_path"]
            cname = item["canonical_class"]
            idx = class_to_idx[cname]
            self.samples.append({
                "path": fpath,
                "label": idx,
                "is_cotton": True,
                "class_name": cname
            })
            
        # Add rehearsal samples
        for item in rehearsal_items:
            fpath = item["file_path"]
            cname = item["source_class"]
            idx = class_to_idx.get(cname, 0)
            self.samples.append({
                "path": fpath,
                "label": idx,
                "is_cotton": False,
                "class_name": cname
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            with Image.open(item["path"]) as img:
                img = img.convert("RGB")
        except Exception:
            # Fallback black image if corrupt
            img = Image.new("RGB", (256, 256), (0, 0, 0))
            
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
            
        return tensor, item["label"], item["is_cotton"], item["class_name"]


def get_safe_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.enabled = False
            probe = (torch.ones(1, device="cuda") + 1).cpu()
            if probe.item() == 2:
                return torch.device("cuda")
        except Exception as e:
            print(f"[Device] CUDA probe notice ({e}). Falling back to CPU.")
    return torch.device("cpu")


def train_epoch(
    student,
    teacher,
    loader,
    optimizer,
    device,
    temperature=2.0,
    lambda_rehearsal=1.0,
    lambda_distill=2.0,
    label_smoothing=0.05
):
    student.train()
    teacher.eval()
    
    # Freeze backbone features
    student.features.eval()
    student.cbam.train()
    student.classifier.train()
    
    total_loss = 0.0
    cotton_correct = 0
    cotton_total = 0
    rehearsal_correct = 0
    rehearsal_total = 0
    
    criterion_ce_smooth = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_ce_plain = nn.CrossEntropyLoss()
    
    for images, labels, is_cotton, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        is_cotton = is_cotton.to(device)
        
        optimizer.zero_grad()
        
        student_logits = student(images)  # [B, 286]
        
        cotton_mask = is_cotton
        rehearsal_mask = ~is_cotton
        
        loss = torch.tensor(0.0, device=device)
        
        # 1. Cotton loss
        if cotton_mask.sum() > 0:
            c_logits = student_logits[cotton_mask]
            c_labels = labels[cotton_mask]
            l_cotton = criterion_ce_smooth(c_logits, c_labels)
            loss = loss + l_cotton
            
            _, c_preds = torch.max(c_logits, dim=1)
            cotton_correct += (c_preds == c_labels).sum().item()
            cotton_total += cotton_mask.sum().item()
            
        # 2. Rehearsal & Distillation loss
        if rehearsal_mask.sum() > 0:
            r_images = images[rehearsal_mask]
            r_student_logits = student_logits[rehearsal_mask, :281]  # Slice 281 original classes
            r_labels = labels[rehearsal_mask]
            
            # Rehearsal CE loss
            l_rehearsal = criterion_ce_plain(r_student_logits, r_labels)
            loss = loss + lambda_rehearsal * l_rehearsal
            
            # Distillation from Teacher
            with torch.no_grad():
                teacher_logits = teacher(r_images)  # [B_r, 281]
                
            p_teacher = F.softmax(teacher_logits / temperature, dim=1)
            log_p_student = F.log_softmax(r_student_logits / temperature, dim=1)
            l_distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)
            
            loss = loss + lambda_distill * l_distill
            
            _, r_preds = torch.max(r_student_logits, dim=1)
            rehearsal_correct += (r_preds == r_labels).sum().item()
            rehearsal_total += rehearsal_mask.sum().item()
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
    n_samples = max(1, cotton_total + rehearsal_total)
    avg_loss = total_loss / n_samples
    cotton_acc = (cotton_correct / max(1, cotton_total)) * 100.0
    rehearsal_acc = (rehearsal_correct / max(1, rehearsal_total)) * 100.0
    
    return avg_loss, cotton_acc, rehearsal_acc


def evaluate_dev(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    
    with torch.no_grad():
        for images, labels, _, class_names in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            _, preds = torch.max(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for p, l, cname in zip(preds.cpu().tolist(), labels.cpu().tolist(), class_names):
                per_class_total[cname] = per_class_total.get(cname, 0) + 1
                if p == l:
                    per_class_correct[cname] = per_class_correct.get(cname, 0) + 1
                    
    acc = (correct / max(1, total)) * 100.0
    class_accs = {c: (per_class_correct.get(c, 0) / per_class_total[c]) * 100.0 for c in per_class_total}
    return acc, class_accs


def train_candidate(seed: int, epochs: int = 10):
    print(f"\n============================================================")
    print(f" STARTING TRAINING CANDIDATE (SEED {seed}) - PHASE 1")
    print(f"============================================================")
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = get_safe_device()
    print(f"Device: {device}")
    
    # Load manifests
    with open("research/cotton286/cotton_split_manifest.json", "r", encoding="utf-8") as f:
        cotton_split = json.load(f)
    with open("research/cotton286/rehearsal_manifest.json", "r", encoding="utf-8") as f:
        rehearsal_data = json.load(f)
    with open("weights/class_names.txt", "r", encoding="utf-8") as f:
        orig_classes = [l.strip() for l in f if l.strip()]
        
    # Build full 286 class list
    new_cotton_classes = [
        "Cotton - Aphids",
        "Cotton - Bacterial Blight",
        "Cotton - Healthy",
        "Cotton - Powdery Mildew",
        "Cotton - Target Spot"
    ]
    all_classes = orig_classes + new_cotton_classes
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = CottonRehearsalDataset(
        cotton_items=cotton_split["splits"]["train"],
        rehearsal_items=rehearsal_data["samples"],
        class_to_idx=class_to_idx,
        transform=train_transform
    )
    
    dev_dataset = CottonRehearsalDataset(
        cotton_items=cotton_split["splits"]["dev"],
        rehearsal_items=[],  # Dev evaluates cotton
        class_to_idx=class_to_idx,
        transform=eval_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)} (Cotton: {len(cotton_split['splits']['train'])}, Rehearsal: {len(rehearsal_data['samples'])})")
    print(f"Dev samples:   {len(dev_dataset)}")
    
    # Load Teacher (Frozen 281 model)
    teacher = EfficientNetB5_CBAM(num_classes=281, pretrained=False)
    teacher_sd = torch.load("weights/efficientnet_b5_cbam_best.pt", map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_sd)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
        
    # Load Student (Initialized 286 model)
    student = EfficientNetB5_CBAM(num_classes=286, pretrained=False)
    init_sd = torch.load("research/cotton286/init_286.pt", map_location="cpu", weights_only=False)
    student.load_state_dict(init_sd)
    student.to(device)
    
    # Freeze backbone (Blocks 0 to 8)
    for p in student.features.parameters():
        p.requires_grad = False
    for p in student.cbam.parameters():
        p.requires_grad = True
    for p in student.classifier.parameters():
        p.requires_grad = True
        
    optimizer = torch.optim.AdamW([
        {"params": student.classifier.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": student.cbam.parameters(), "lr": 5e-5, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    
    epoch_metrics = []
    best_dev_acc = 0.0
    best_sd = None
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, cotton_acc, rehearsal_acc = train_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=2.0,
            lambda_rehearsal=1.0,
            lambda_distill=2.0,
            label_smoothing=0.05
        )
        scheduler.step()
        
        dev_acc, dev_class_accs = evaluate_dev(student, dev_loader, device)
        elapsed = time.time() - t0
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_sd = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {loss:.4f} | Cotton Train Acc: {cotton_acc:.1f}% | Rehearsal Acc: {rehearsal_acc:.1f}% | Dev Acc: {dev_acc:.1f}% ({elapsed:.1f}s)")
        
        epoch_metrics.append({
            "epoch": epoch,
            "loss": round(loss, 4),
            "cotton_train_acc": round(cotton_acc, 2),
            "rehearsal_acc": round(rehearsal_acc, 2),
            "dev_acc": round(dev_acc, 2),
            "dev_per_class_acc": dev_class_accs,
            "elapsed_seconds": round(elapsed, 2)
        })
        
    # Save checkpoint
    out_pt = f"research/cotton286/cotton286_seed{seed}.pt"
    save_sd = best_sd if best_sd is not None else student.state_dict()
    torch.save(save_sd, out_pt)
    
    with open(out_pt, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Finished Seed {seed}. Best Dev Acc: {best_dev_acc:.2f}%. Saved to {out_pt} (SHA256: {sha256[:16]}...)")
    
    metrics_out = f"research/cotton286/training_metrics_seed{seed}.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "epochs": epochs,
            "best_dev_acc": round(best_dev_acc, 2),
            "checkpoint": out_pt,
            "sha256": sha256,
            "epoch_history": epoch_metrics
        }, f, indent=2)
        
    return out_pt, sha256, best_dev_acc


def main():
    config = {
        "architecture": "EfficientNet-B5 + CBAM",
        "phase": 1,
        "backbone_frozen": True,
        "trainable_components": ["cbam", "classifier"],
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "learning_rates": {
            "classifier": 1e-4,
            "cbam": 5e-5,
            "backbone": 0.0
        },
        "loss_function": {
            "formula": "L_total = L_new_class + lambda_rehearsal * L_old_class + lambda_distill * L_distill",
            "lambda_rehearsal": 1.0,
            "lambda_distill": 2.0,
            "temperature": 2.0,
            "label_smoothing": 0.05
        },
        "epochs": 10,
        "seeds": [42, 43],
        "batch_size": 16
    }
    
    with open("research/cotton286/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    checkpoints = {}
    
    # Train Seed 42
    pt42, sha42, dev42 = train_candidate(seed=42, epochs=10)
    checkpoints["seed42"] = {"path": pt42, "sha256": sha42, "best_dev_acc": dev42}
    
    # Train Seed 43
    pt43, sha43, dev43 = train_candidate(seed=43, epochs=10)
    checkpoints["seed43"] = {"path": pt43, "sha256": sha43, "best_dev_acc": dev43}
    
    # Select best candidate
    best_candidate_key = "seed42" if dev42 >= dev43 else "seed43"
    best_pt = checkpoints[best_candidate_key]["path"]
    
    # Copy to weights/research/efficientnet_b5_cbam_cotton286_v1.pt
    research_v1_path = "weights/research/efficientnet_b5_cbam_cotton286_v1.pt"
    import shutil
    shutil.copyfile(best_pt, research_v1_path)
    with open(research_v1_path, "rb") as f:
        v1_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    checkpoints["research_v1"] = {
        "path": research_v1_path,
        "source_seed": best_candidate_key,
        "sha256": v1_sha256
    }
    
    with open("research/cotton286/checkpoint_manifest.json", "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2)
        
    print(f"\n============================================================")
    print(f" TRAINING COMPLETE!")
    print(f" Seed 42 Dev Acc: {dev42:.2f}% (SHA256: {sha42[:16]}...)")
    print(f" Seed 43 Dev Acc: {dev43:.2f}% (SHA256: {sha43[:16]}...)")
    print(f" Selected Best:   {best_candidate_key} -> {research_v1_path}")
    print(f" Saved checkpoint manifest to research/cotton286/checkpoint_manifest.json")
    print(f"============================================================")


if __name__ == "__main__":
    main()
