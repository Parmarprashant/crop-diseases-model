import os
import sys
import json
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM
from training.dataset import get_data_loaders


def get_safe_device() -> torch.device:
    """Use CUDA with native ATen kernels on RTX 5060 (Blackwell sm_120)."""
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.enabled = False
            probe = (torch.ones(1, device="cuda") + 1).cpu()
            if probe.item() == 2:
                return torch.device("cuda")
        except Exception as e:
            print(f"[Device] CUDA probe notice ({e}). Falling back to CPU.")
    return torch.device("cpu")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    total_epochs: int,
    grad_accum_steps: int = 2,
    max_batches: int = 0,
    log_interval: int = 50
):
    model.train()
    # Freeze stages 0 to 5 of features in eval mode
    for i in range(min(6, len(model.features))):
        model.features[i].eval()

    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader) if max_batches <= 0 else min(max_batches, len(loader))
    epoch_start = time.time()

    optimizer.zero_grad()
    for i, (images, targets) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Frozen low-level features (Blocks 0 to 5) run with no grad
        with torch.no_grad():
            for block_idx in range(6):
                images = model.features[block_idx](images)

        # High-level features (Blocks 6 to 8) and CBAM + Head compute gradients
        for block_idx in range(6, len(model.features)):
            images = model.features[block_idx](images)

        feat_attended = model.cbam(images)
        pooled = model.avgpool(feat_attended)
        flattened = torch.flatten(pooled, 1)

        outputs = model.classifier(flattened)
        loss = criterion(outputs, targets)

        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == total_batches:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % 50 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        running_loss += loss.item() * targets.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if i == 0 or (i + 1) % log_interval == 0 or (i + 1) == total_batches:
            batch_acc = (correct / total) * 100.0
            avg_loss = running_loss / total
            elapsed = time.time() - epoch_start
            pct = 100.0 * (i + 1) / total_batches
            vram_str = f" | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB" if device.type == "cuda" else ""
            print(
                f"  Epoch [{epoch:02d}/{total_epochs:02d}] "
                f"Batch [{i+1:04d}/{total_batches:04d}] ({pct:5.1f}%) - "
                f"Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) | "
                f"Acc: {batch_acc:.2f}% | Elapsed: {elapsed:.0f}s{vram_str}",
                flush=True
            )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, max_batches: int = 0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            feat = model.features(images)
            feat_attended = model.cbam(feat)
            pooled = model.avgpool(feat_attended)
            flattened = torch.flatten(pooled, 1)
            outputs = model.classifier(flattened)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()

            # Top-3 evaluation
            _, top3_preds = torch.topk(outputs, k=min(3, outputs.size(1)), dim=1)
            targets_expanded = targets.view(-1, 1).expand_as(top3_preds)
            top3_correct += (top3_preds == targets_expanded).any(dim=1).sum().item()

            total += targets.size(0)

            if (i + 1) % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    val_top3_acc = (top3_correct / max(total, 1)) * 100.0
    return val_loss, val_acc, val_top3_acc


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Deep Fine-Tuning of EfficientNet-B5 + CBAM on 281 Classes")
    parser.add_argument("--data_dir", type=str, default="Data", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for classification head")
    parser.add_argument("--lr_cbam", type=float, default=5e-5, help="Learning rate for CBAM module")
    parser.add_argument("--lr_backbone", type=float, default=2e-5, help="Learning rate for Blocks 6-8")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--output_dir", type=str, default="weights", help="Checkpoint directory")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--warm_start", type=str, default="weights/efficientnet_b5_cbam_best.pt", help="Path to best Stage 1 weights")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Max batches per epoch (0 for full)")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Max val batches (0 for full)")
    parser.add_argument("--log_interval", type=int, default=50, help="Log frequency")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_safe_device()
    effective_bs = args.batch_size * args.grad_accum

    print("=" * 80)
    print(" [Stage 2 Deep Fine-Tuning] AgriVision AI - EfficientNet-B5 + CBAM Attention")
    print(f" [Hardware]          {device.type.upper()}" + (f" - {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print(f" [Configuration]     Target Epochs: {args.epochs} | Effective Batch Size: {effective_bs} (Batch: {args.batch_size} x Accum: {args.grad_accum})")
    print(f"                     Learning Rates: Head={args.lr_head} | CBAM={args.lr_cbam} | Backbone(Blocks 6-8)={args.lr_backbone}")
    print(f" [Warm Start Weights] {args.warm_start}")
    print("=" * 80, flush=True)

    train_loader, val_loader, class_names = get_data_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        balance=False
    )

    total_train = len(train_loader.dataset)
    total_val = len(val_loader.dataset)
    print(f"[Dataset] Sourced {len(class_names)} classes | Train: {total_train:,} images | Val: {total_val:,} images\n", flush=True)

    model = EfficientNetB5_CBAM(num_classes=len(class_names), pretrained=False)

    if os.path.exists(args.warm_start):
        try:
            sd = torch.load(args.warm_start, map_location="cpu", weights_only=False)
            model.load_state_dict(sd, strict=False)
            print(f"[Warm Start] Successfully loaded Stage 1 checkpoint from {args.warm_start} (77.81% baseline)!")
            del sd
        except Exception as e:
            print(f"[Warm Start Warning] Could not load checkpoint: {e}")

    # Freeze low-level blocks 0 to 5; unfreeze blocks 6, 7, 8
    for i in range(6):
        for p in model.features[i].parameters():
            p.requires_grad = False

    backbone_trainable = []
    for i in range(6, len(model.features)):
        for p in model.features[i].parameters():
            p.requires_grad = True
            backbone_trainable.append(p)

    for p in model.cbam.parameters():
        p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW([
        {"params": model.classifier.parameters(), "lr": args.lr_head, "weight_decay": 1e-4},
        {"params": model.cbam.parameters(), "lr": args.lr_cbam, "weight_decay": 1e-4},
        {"params": backbone_trainable, "lr": args.lr_backbone, "weight_decay": 1e-4}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

    best_val_acc = 77.81  # Baseline to beat
    history_path = os.path.join(args.output_dir, "training_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            pass

    best_weights_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_best.pt")
    latest_weights_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_latest.pt")
    finetune_best_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_stage2_best.pt")

    print(f"[Fine-Tuning Active] Baseline to beat: {best_val_acc:.2f}% | Training {args.epochs} deep epochs...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            grad_accum_steps=args.grad_accum,
            max_batches=args.max_train_batches,
            log_interval=args.log_interval
        )

        val_loss, val_acc, val_top3_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches
        )
        scheduler.step()
        elapsed = time.time() - start_t

        print("-" * 80)
        print(
            f"STAGE 2 EPOCH [{epoch:02d}/{args.epochs:02d}] FINISHED | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc (Top-1): {val_acc:.2f}% | Top-3 Acc: {val_top3_acc:.2f}% | Duration: {elapsed:.1f}s",
            flush=True
        )
        print("-" * 80)

        record = {
            "stage": 2,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_top3_acc": round(val_top3_acc, 2),
            "duration_seconds": round(elapsed, 1),
            "lr_head": optimizer.param_groups[0]["lr"],
            "classes_count": len(class_names)
        }
        history.append(record)

        torch.save(model.state_dict(), latest_weights_path)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            torch.save(model.state_dict(), finetune_best_path)
            print(f" >>> [NEW ALL-TIME BEST MODEL] Val Top-1: {best_val_acc:.2f}%! Checkpoint saved to {best_weights_path}", flush=True)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"\n[Stage 2 Complete] All-Time Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[Best Checkpoint]  {best_weights_path}", flush=True)


if __name__ == "__main__":
    main()
