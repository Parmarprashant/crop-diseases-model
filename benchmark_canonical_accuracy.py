import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from models.efficientnet_cbam import build_efficientnet_cbam
from training.dataset import MainDataCropDataset

def run_benchmark(max_samples: int = 1500, batch_size: int = 16):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AgriVision Validation Benchmark ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Load Classes and Canonical Map
    with open("weights/class_names.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from weights/class_names.txt")
    
    with open("weights/canonical_class_map.json", "r", encoding="utf-8") as f:
        canonical_map = json.load(f)
    print(f"Loaded canonical map with {len(canonical_map)} entries")
    
    def to_canonical(name: str) -> str:
        return canonical_map.get(name, name)
    
    # 2. Load Model Checkpoint
    checkpoint_path = "weights/efficientnet_b5_cbam_best.pt"
    print(f"Loading checkpoint: {checkpoint_path}...")
    model = build_efficientnet_cbam(num_classes=num_classes, weights_path=checkpoint_path, device=str(device))
    model.eval()
    
    # 3. Load Validation Dataset
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = MainDataCropDataset(
        root_dir="Data",
        subset="Validation",
        transform=val_transform,
        classes=class_names
    )
    total_val_available = len(val_dataset)
    print(f"Validation dataset total size: {total_val_available} images")
    
    eval_count = min(max_samples, total_val_available) if max_samples > 0 else total_val_available
    print(f"Benchmarking on {eval_count} validation images (batch size: {batch_size})...\n")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Tracking accumulators
    n_total = 0
    raw_top1_single = 0
    raw_top3_single = 0
    raw_top1_tta = 0
    raw_top3_tta = 0
    
    canon_top1_single = 0
    canon_top3_single = 0
    canon_top1_tta = 0
    canon_top3_tta = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            b_size = images.size(0)
            if eval_count and (n_total + b_size > eval_count):
                b_size = eval_count - n_total
                images = images[:b_size]
                targets = targets[:b_size]
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Single pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits_orig = model(images)
                probs_single = F.softmax(logits_orig, dim=1)
                
                # TTA pass: horizontal flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = F.softmax(logits_flip, dim=1)
                probs_tta = (probs_single + probs_flip) / 2.0
            
            # Top-3 predictions for single pass
            top3_single = torch.topk(probs_single, k=min(3, num_classes), dim=1)[1].cpu().numpy()
            # Top-3 predictions for TTA
            top3_tta = torch.topk(probs_tta, k=min(3, num_classes), dim=1)[1].cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(b_size):
                tgt_idx = int(targets_np[i])
                tgt_raw = class_names[tgt_idx] if tgt_idx < num_classes else f"Class_{tgt_idx}"
                tgt_canon = to_canonical(tgt_raw)
                
                # Single Pass evaluation
                s_pred1_raw = class_names[top3_single[i, 0]]
                s_pred3_raw = [class_names[idx] for idx in top3_single[i]]
                s_pred1_canon = to_canonical(s_pred1_raw)
                s_pred3_canon = [to_canonical(r) for r in s_pred3_raw]
                
                if s_pred1_raw == tgt_raw:
                    raw_top1_single += 1
                if tgt_raw in s_pred3_raw:
                    raw_top3_single += 1
                if s_pred1_canon == tgt_canon:
                    canon_top1_single += 1
                if tgt_canon in s_pred3_canon:
                    canon_top3_single += 1
                
                # TTA Pass evaluation
                t_pred1_raw = class_names[top3_tta[i, 0]]
                t_pred3_raw = [class_names[idx] for idx in top3_tta[i]]
                t_pred1_canon = to_canonical(t_pred1_raw)
                t_pred3_canon = [to_canonical(r) for r in t_pred3_raw]
                
                if t_pred1_raw == tgt_raw:
                    raw_top1_tta += 1
                if tgt_raw in t_pred3_raw:
                    raw_top3_tta += 1
                if t_pred1_canon == tgt_canon:
                    canon_top1_tta += 1
                if tgt_canon in t_pred3_canon:
                    canon_top3_tta += 1
            
            n_total += b_size
            if (batch_idx + 1) % 10 == 0 or n_total >= eval_count:
                print(f"Processed {n_total}/{eval_count} images... Current TTA Canonical Top-1: {canon_top1_tta/n_total*100:.2f}%")
            
            if n_total >= eval_count:
                break
    
    elapsed = time.time() - start_time
    fps = n_total / max(elapsed, 0.001)
    
    print("\n" + "="*70)
    print("                 OFFICIAL BENCHMARK EVALUATION RESULTS")
    print("="*70)
    print(f"Images Evaluated : {n_total}")
    print(f"Total Time Taken : {elapsed:.2f} seconds ({fps:.1f} images/sec)")
    print("-" * 70)
    print(f"{'Evaluation Metric':<40} | {'Top-1 Acc':<12} | {'Top-3 Acc':<12}")
    print("-" * 70)
    print(f"{'1. Standard Baseline (Raw 281 Classes)':<40} | {raw_top1_single/n_total*100:>10.2f}% | {raw_top3_single/n_total*100:>10.2f}%")
    print(f"{'2. With Horizontal Flip TTA (Raw 281)':<40} | {raw_top1_tta/n_total*100:>10.2f}% | {raw_top3_tta/n_total*100:>10.2f}%")
    print(f"{'3. Canonical Merged (168 Classes, No TTA)':<40} | {canon_top1_single/n_total*100:>10.2f}% | {canon_top3_single/n_total*100:>10.2f}%")
    print(f"{'4. FULL PIPELINE: Canonical Merged + TTA':<40} | {canon_top1_tta/n_total*100:>10.2f}% | {canon_top3_tta/n_total*100:>10.2f}%")
    print("="*70)
    
    gain_top1 = (canon_top1_tta - raw_top1_single) / n_total * 100
    gain_top3 = (canon_top3_tta - raw_top3_single) / n_total * 100
    print(f"\nNET PERFORMANCE GAIN OVER BASELINE:")
    print(f"  Top-1 Accuracy Gain : +{gain_top1:.2f}%")
    print(f"  Top-3 Accuracy Gain : +{gain_top3:.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_benchmark(max_samples=1500, batch_size=16)
