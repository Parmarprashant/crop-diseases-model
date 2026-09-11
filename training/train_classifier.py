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
    scaler: torch.amp.GradScaler = None,
    grad_accum_steps: int = 1,
    max_batches: int = 0
):
    model.train()
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

        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss_scaled = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == total_batches:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total_batches:
            batch_acc = (correct / total) * 100.0
            avg_loss = running_loss / total
            elapsed = time.time() - epoch_start
            pct = 100.0 * (i + 1) / total_batches
            vram_str = f" | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB" if device.type == "cuda" else ""
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

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train / Continue EfficientNet-B5 + CBAM on full MAIN DATA dataset (GPU, 456px)")
    parser.add_argument("--data_dir", type=str, default="MAIN DATA", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=12, help="Total target epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Training batch size (EfficientNet-B5 @ 456px on 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=456, help="Image resolution - must match EfficientNet-B5 native (456) and inference")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows stability)")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory for GPU transfer")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--balance", action="store_true", default=True, help="Balance classes via WeightedRandomSampler")
    parser.add_argument("--no_balance", dest="balance", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if found")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to specific checkpoint file")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Max batches per epoch (0 for full dataset)")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Max val batches (0 for full validation)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_safe_device()
    num_cores = os.cpu_count() or 8
    torch.backends.cudnn.benchmark = True
    print("=" * 75)
    print(f"[Training Pipeline] AgriVision AI - EfficientNet-B5 + CBAM Attention")
    print(f"[Hardware]          {device.type.upper()}" + (f" - {torch.cuda.get_device_name(0)}" if device.type == "cuda" else f" ({num_cores} CPU cores)"))
    print(f"[Configuration]     Target Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"                    Resolution: {args.img_size}x{args.img_size} | Grad Accum: {args.grad_accum} | Balanced: {args.balance}")
    print("=" * 75, flush=True)

    print(f"[Dataset] Sourcing full dataset from: {args.data_dir}...")
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
    print(f"          Training Samples:   {total_train:,} images ({len(train_loader)} batches/epoch with sampler)")
    print(f"          Validation Samples: {total_val:,} images ({len(val_loader)} batches)")

    classes_file = os.path.join(args.output_dir, "class_names.txt")
    with open(classes_file, "w") as f:
        for c in class_names:
            f.write(f"{c}\n")
    print(f"[Metadata] Class manifest written to {classes_file}", flush=True)

    print("[Model] Constructing EfficientNet-B5 + CBAM Attention network...")
    model = EfficientNetB5_CBAM(num_classes=len(class_names), pretrained=True)
    model.to(device)

    # No label smoothing: softmax confidence must be able to exceed 0.80 gate
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_val_acc = 0.0
    history = []
    checkpoint_file = args.checkpoint or os.path.join(args.output_dir, "checkpoint.pt")
    best_weights_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_best.pt")
    latest_weights_path = os.path.join(args.output_dir, "efficientnet_b5_cbam_latest.pt")
    history_path = os.path.join(args.output_dir, "training_history.json")

    if args.resume:
        if os.path.exists(checkpoint_file):
            print(f"[Checkpoint] Resuming from full state checkpoint: {checkpoint_file}")
            try:
                ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_acc = ckpt.get("best_val_acc", 0.0)
                history = ckpt.get("history", [])
                if ckpt.get("img_size") != args.img_size and ckpt.get("img_size") is not None:
                    print(f"[Checkpoint Warning] Checkpoint was trained at {ckpt.get('img_size')}px; current run is {args.img_size}px. Starting fresh weights.")
                    start_epoch, best_val_acc, history = 1, 0.0, []
                else:
                    print(f"[Checkpoint] Successfully resumed at epoch {start_epoch} (best previous val acc: {best_val_acc:.2f}%)")
            except Exception as e:
                print(f"[Checkpoint Warning] Could not load state from {checkpoint_file} ({e}). Starting fresh.")
        elif os.path.exists(best_weights_path):
            print(f"[Weights] No checkpoint found but existing best weights found: {best_weights_path}")
            try:
                state_dict = torch.load(best_weights_path, map_location=device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Weights] Loaded 256px-era weights as warm start (head may mismatch until first save).")
            except Exception as e:
                print(f"[Weights Warning] Could not load weights: {e}")

    if start_epoch > args.epochs:
        print(f"[Info] Target epochs ({args.epochs}) already achieved (current epoch {start_epoch-1}).")
        print(f"[Info] Increasing target epochs to {start_epoch + 5} to continue training...")
        args.epochs = start_epoch + 5
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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
            scaler=scaler,
            grad_accum_steps=args.grad_accum,
            max_batches=args.max_train_batches
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
            "img_size": args.img_size
        }
        history.append(epoch_record)

        torch.save(model.state_dict(), latest_weights_path)

        checkpoint_payload = {
            "epoch": epoch,
            "img_size": args.img_size,
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

        with open(history_path, "w") as f:
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
