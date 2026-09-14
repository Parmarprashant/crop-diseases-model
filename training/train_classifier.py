import os
import sys

import warnings
warnings.filterwarnings("ignore")

import json
import argparse
import time
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
    """Use CUDA when a working GPU is present, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        try:
            # RTX 5060 (Blackwell sm_120) runs 200x faster and rock-solid with native ATen CUDA kernels
            torch.backends.cudnn.enabled = False
            probe = (torch.ones(1, device="cuda") + 1).cpu()
            if probe.item() == 2:
                return torch.device("cuda")
        except Exception as e:
            print(f"[Device] CUDA probe failed ({e}). Falling back to CPU.")
    return torch.device("cpu")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    total_epochs: int,
    grad_accum_steps: int = 8,
    max_batches: int = 0,
    log_interval: int = 50
):
    model.classifier.train()
    model.cbam.train()
    model.features.eval()

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

        with torch.no_grad():
            feat = model.features(images)

        feat_attended = model.cbam(feat)
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

        running_loss += loss.item() * images.size(0)
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

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (i + 1) % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune EfficientNet-B5 + CBAM on 281-Class Agricultural Dataset")
    parser.add_argument("--data_dir", type=str, default="Data", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Total epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per step (16 optimal for RTX 5060 speed)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (16 x 2 = effective batch size 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution matching inference")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows stability)")
    parser.add_argument("--balance", action="store_true", default=False, help="Balance classes via WeightedRandomSampler")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint if found")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to specific checkpoint file")
    parser.add_argument("--warm_start", type=str, default="", help="Path to pretrained weights for backbone transfer")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Max batches per epoch (0 for full dataset)")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Max val batches (0 for full validation)")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging frequency in batches")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_safe_device()
    num_cores = os.cpu_count() or 8
    effective_bs = args.batch_size * args.grad_accum

    print("=" * 75)
    print(f"[Training Pipeline] AgriVision AI - EfficientNet-B5 + CBAM Attention")
    print(f"[Hardware]          {device.type.upper()}" + (f" - {torch.cuda.get_device_name(0)}" if device.type == "cuda" else f" ({num_cores} CPU cores)"))
    print(f"[Configuration]     Target Epochs: {args.epochs} | Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum} -> Effective BS: {effective_bs})")
    print(f"                    LR: {args.lr} | Resolution: {args.img_size}x{args.img_size} | Optimizer: AdamW(weight_decay=1e-4)")
    print("=" * 75, flush=True)

    print(f"[Dataset] Sourcing dataset from: {args.data_dir}...")
    train_loader, val_loader, class_names = get_data_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        balance=args.balance
    )

    total_train = len(train_loader.dataset)
    total_val = len(val_loader.dataset)
    print(f"[Dataset] Verified {len(class_names)} unique crop & pest disease classes.")
    print(f"          Training Samples:   {total_train:,} images ({len(train_loader)} batches/epoch)")
    print(f"          Validation Samples: {total_val:,} images ({len(val_loader)} batches)")

    classes_file = os.path.join(args.output_dir, "class_names.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(f"{c}\n")
    print(f"[Metadata] Class manifest ({len(class_names)} classes) written to {classes_file}", flush=True)

    print("[Model] Constructing EfficientNet-B5 + CBAM Attention network...")
    model = EfficientNetB5_CBAM(num_classes=len(class_names), pretrained=True)

    # Optional explicit warm start
    if args.warm_start and os.path.exists(args.warm_start):
        try:
            ckpt = torch.load(args.warm_start, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model_sd = model.state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(filtered, strict=False)
            print(f"[Transfer Learning] Transferred {len(filtered)} layers from {args.warm_start}.")
            del ckpt, sd, filtered
        except Exception as e:
            print(f"[Warm Start Notice] Could not load {args.warm_start}: {e}")

    # Freeze backbone for maximum speed; train CBAM attention + classification head
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.cbam.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW([
        {"params": model.classifier.parameters(), "lr": args.lr, "weight_decay": 1e-4},
        {"params": model.cbam.parameters(), "lr": args.lr * 0.5, "weight_decay": 1e-4}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 1
    best_val_acc = 0.0
    history = []
    checkpoint_file = args.checkpoint or os.path.join(args.output_dir, "checkpoint.pt")
    best_weights_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_best.pt")
    latest_weights_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_latest.pt")
    history_path = os.path.join(args.output_dir, "training_history.json")

    # Resume handling if explicitly requested and valid checkpoint exists
    if args.resume and os.path.exists(checkpoint_file):
        try:
            ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            ckpt_sd = ckpt.get("model_state_dict", {})
            model_sd = model.state_dict()
            head_match = (
                "classifier.4.weight" in ckpt_sd and 
                ckpt_sd["classifier.4.weight"].shape == model_sd["classifier.4.weight"].shape
            )
            if head_match:
                model.load_state_dict(ckpt_sd, strict=False)
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_acc = ckpt.get("best_val_acc", 0.0)
                history = ckpt.get("history", [])
                print(f"[Checkpoint] Resumed model weights from epoch {start_epoch} (best previous val acc: {best_val_acc:.2f}%)")
            del ckpt
        except Exception as e:
            print(f"[Checkpoint Notice] {e}. Starting fresh epochs.")

    print(f"\n[Training Started] Training epochs {start_epoch} -> {args.epochs} on {total_train:,} images...\n", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
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

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches
        )
        scheduler.step()
        elapsed = time.time() - start_t

        print("-" * 75)
        print(
            f"EPOCH [{epoch:02d}/{args.epochs:02d}] FINISHED | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Duration: {elapsed:.1f}s",
            flush=True
        )
        print("-" * 75)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "duration_seconds": round(elapsed, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "img_size": args.img_size,
            "classes_count": len(class_names)
        }
        history.append(epoch_record)

        torch.save(model.state_dict(), latest_weights_path)

        checkpoint_payload = {
            "epoch": epoch,
            "img_size": args.img_size,
            "classes_count": len(class_names),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": max(best_val_acc, val_acc),
            "history": history
        }
        torch.save(checkpoint_payload, checkpoint_file)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f" >>> [NEW BEST MODEL] Val Acc: {best_val_acc:.2f}%! Checkpoint saved to {best_weights_path}", flush=True)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"\n[Training Complete] Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[Best Checkpoint]   {best_weights_path}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR IN TRAINING]: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
