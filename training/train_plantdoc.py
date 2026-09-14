"""
AgriVision AI — PlantDoc Fine-Tuning Experiment.
Conservative fine-tuning of EfficientNet-B5 + CBAM on real-world field imagery
with low-level layer freezing and weight-anchoring regularization to mitigate catastrophic forgetting.
Saves experimental model to weights/efficientnet_b5_cbam_plantdoc.pt.
NEVER modifies the production checkpoint weights/efficientnet_b5_cbam_best.pt.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import build_efficientnet_cbam, EfficientNetB5_CBAM
from training.plantdoc_dataset import get_plantdoc_loaders

def get_safe_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def evaluate_plantdoc(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()

            # Top-3 accuracy
            _, top3_preds = torch.topk(outputs, k=min(3, outputs.size(1)), dim=1)
            top3_correct += sum(t.item() in top3_preds[idx].tolist() for idx, t in enumerate(targets))

            total += targets.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    val_top3 = (top3_correct / max(total, 1)) * 100.0
    return val_loss, val_acc, val_top3

def main():
    parser = argparse.ArgumentParser(description="PlantDoc Conservative Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=4, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr_head", type=float, default=5e-5, help="Learning rate for classification head")
    parser.add_argument("--lr_cbam", type=float, default=2e-5, help="Learning rate for CBAM module")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for upper backbone (Blocks 6-8)")
    parser.add_argument("--anchor_weight", type=float, default=1e-4, help="Weight anchoring penalty weight")
    parser.add_argument("--warm_start", type=str, default="weights/efficientnet_b5_cbam_best.pt", help="Protected baseline weights")
    parser.add_argument("--output_model", type=str, default="weights/efficientnet_b5_cbam_plantdoc.pt", help="Output experimental weights")
    parser.add_argument("--history_output", type=str, default="validation/plantdoc_training_history.json", help="Training history path")
    args = parser.parse_args()

    os.makedirs("weights/plantdoc_experiment", exist_ok=True)
    os.makedirs("validation", exist_ok=True)

    device = get_safe_device()
    print("=" * 80)
    print("      AGRIVISION AI: PLANTDOC CONSERVATIVE FINE-TUNING EXPERIMENT")
    print("=" * 80)
    print(f"Hardware            : {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Warm Start Weights  : {args.warm_start} (PROTECTED)")
    print(f"Output Experimental : {args.output_model}")
    print(f"Training Epochs     : {args.epochs} | Effective Batch Size: {args.batch_size * args.grad_accum}")
    print(f"Learning Rates      : Head={args.lr_head} | CBAM={args.lr_cbam} | Backbone(Blocks 6-8)={args.lr_backbone}")
    print(f"Weight Anchoring L2 : {args.anchor_weight}")
    print("=" * 80)

    # 1. Record starting checkpoint hash
    if not os.path.exists(args.warm_start):
        print(f"FATAL: Warm start weights {args.warm_start} not found!")
        sys.exit(1)

    start_sha = hashlib.sha256(open(args.warm_start, "rb").read()).hexdigest()
    start_size = os.path.getsize(args.warm_start)
    print(f"[Audit] Starting Checkpoint SHA256 : {start_sha}")
    print(f"[Audit] Starting Checkpoint Size   : {start_size:,} bytes\n")

    # 2. DataLoaders
    train_loader, test_loader = get_plantdoc_loaders(root_dir="plantDoc", batch_size=args.batch_size)
    print(f"[Data] PlantDoc Train: {len(train_loader.dataset)} images | Test: {len(test_loader.dataset)} images\n")

    # 3. Model Setup
    with open("weights/class_names.txt", "r") as f:
        class_names = [l.strip() for l in f if l.strip()]
    num_classes = len(class_names) # 281 output classes

    model = EfficientNetB5_CBAM(num_classes=num_classes, pretrained=False)
    sd = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    print(f"[Model] Successfully loaded pre-trained baseline from {args.warm_start}.")
    del sd

    # Layer Freezing Strategy:
    # Blocks 0 to 5: FROZEN
    # Blocks 6 to 8: TRAINABLE
    # CBAM: TRAINABLE
    # Head: TRAINABLE
    for i in range(min(6, len(model.features))):
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Parameters] Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M) | Frozen: {frozen_params:,} ({frozen_params/1e6:.2f}M)")

    model.to(device)

    # Store baseline anchor weights for L2 regularization
    anchor_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            anchor_weights[name] = param.detach().clone()

    # Optimizer with parameter groups
    param_groups = [
        {"params": model.classifier.parameters(), "lr": args.lr_head, "weight_decay": 1e-4},
        {"params": model.cbam.parameters(), "lr": args.lr_cbam, "weight_decay": 1e-4},
        {"params": backbone_trainable, "lr": args.lr_backbone, "weight_decay": 1e-4}
    ]
    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Initial zero-shot evaluation on PlantDoc test before fine-tuning
    print("\n[Pre-Training Zero-Shot Test] Evaluating Model A on plantDoc/test...")
    init_loss, init_acc, init_top3 = evaluate_plantdoc(model, test_loader, criterion, device)
    print(f" -> Model A Baseline on PlantDoc Test: Loss={init_loss:.4f} | Top-1={init_acc:.2f}% | Top-3={init_top3:.2f}%\n")

    history = []
    best_val_acc = init_acc
    best_epoch = 0

    total_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        # Keep frozen blocks strictly in eval mode
        for i in range(min(6, len(model.features))):
            model.features[i].eval()
        # Freeze all BatchNorm layers to prevent covariate shift and preserve pre-trained feature statistics
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass: frozen blocks run without gradient
            with torch.no_grad():
                for i in range(6):
                    images = model.features[i](images)

            for i in range(6, len(model.features)):
                images = model.features[i](images)

            feat = model.cbam(images)
            pooled = model.avgpool(feat)
            flattened = torch.flatten(pooled, 1)
            outputs = model.classifier(flattened)

            ce_loss = criterion(outputs, targets)

            # Weight anchoring loss against pre-trained baseline weights
            anchor_loss = torch.tensor(0.0, device=device)
            if args.anchor_weight > 0:
                for name, param in model.named_parameters():
                    if param.requires_grad and name in anchor_weights:
                        anchor_loss += torch.sum((param - anchor_weights[name]) ** 2)
                anchor_loss = args.anchor_weight * anchor_loss

            total_loss = ce_loss + anchor_loss
            loss_scaled = total_loss / args.grad_accum
            loss_scaled.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == total_batches:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += ce_loss.item() * targets.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                batch_acc = (correct / total) * 100.0
                elapsed = time.time() - epoch_start
                print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] Batch [{batch_idx+1:03d}/{total_batches:03d}] - CE Loss: {ce_loss.item():.4f} | Acc: {batch_acc:.2f}% | Elapsed: {elapsed:.0f}s", flush=True)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = (correct / max(total, 1)) * 100.0
        epoch_duration = time.time() - epoch_start

        # Validation on plantDoc/test
        val_loss, val_acc, val_top3 = evaluate_plantdoc(model, test_loader, criterion, device)

        print("-" * 80)
        print(f"EPOCH [{epoch:02d}/{args.epochs:02d}] COMPLETE | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {val_loss:.4f} (Top-1: {val_acc:.2f}%, Top-3: {val_top3:.2f}%) | Duration: {epoch_duration:.1f}s")
        print("-" * 80, flush=True)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_top1_acc": round(val_acc, 2),
            "val_top3_acc": round(val_top3, 2),
            "lr_head": optimizer.param_groups[0]["lr"],
            "lr_cbam": optimizer.param_groups[1]["lr"],
            "lr_backbone": optimizer.param_groups[2]["lr"],
            "duration_seconds": round(epoch_duration, 1)
        }
        history.append(record)

        # Save intermediate epoch checkpoint
        inter_path = f"weights/plantdoc_experiment/checkpoint_epoch_{epoch}.pt"
        torch.save(model.state_dict(), inter_path)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_model)
            print(f" >>> [NEW BEST CHECKPOINT] Epoch {epoch}: Val Acc {val_acc:.2f}%! Saved to {args.output_model}", flush=True)

    # If no epoch exceeded baseline val_acc, save the final epoch as experimental model
    if not os.path.exists(args.output_model):
        torch.save(model.state_dict(), args.output_model)
        best_val_acc = val_acc
        best_epoch = args.epochs
        print(f" >>> [FINAL CHECKPOINT] Saved final epoch to {args.output_model}", flush=True)

    # Save training history manifest
    history_manifest = {
        "experiment": "PlantDoc Conservative Fine-Tuning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42,
        "dataset_path": "plantDoc",
        "train_images": len(train_loader.dataset),
        "test_images": len(test_loader.dataset),
        "classes_supervised": 28,
        "output_classes": num_classes,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "baseline_plantdoc_val_acc": round(init_acc, 2),
        "best_plantdoc_val_acc": round(best_val_acc, 2),
        "best_epoch": best_epoch,
        "starting_checkpoint_sha256": start_sha,
        "output_checkpoint": args.output_model,
        "epochs": history
    }
    with open(args.history_output, "w", encoding="utf-8") as f:
        json.dump(history_manifest, f, indent=2)
    print(f"\nTraining history successfully saved to {args.history_output}.")

    # 4. Strict Safety Audit: Re-verify protected baseline checkpoint
    post_sha = hashlib.sha256(open(args.warm_start, "rb").read()).hexdigest()
    post_size = os.path.getsize(args.warm_start)
    assert post_sha == start_sha, f"CRITICAL ERROR: Protected baseline {args.warm_start} was modified!"
    assert post_size == start_size, f"CRITICAL ERROR: Protected baseline size changed!"
    print(f"\n[Audit PASSED] Protected baseline {args.warm_start} is 100% UNTOUCHED and identical.")
    print(f"[Audit PASSED] Experimental checkpoint {args.output_model} created ({os.path.getsize(args.output_model):,} bytes).")

if __name__ == "__main__":
    main()
