"""
AgriVision AI — Hierarchical Multi-Task Training with Teacher Distillation.
Trains EfficientNetB5_CBAM_Hierarchical using:
1. Shared backbone (Blocks 0-5 frozen, Blocks 6-8 trainable)
2. CBAM attention module
3. Warm disease head initialized from Model A
4. Dedicated 41-class crop head
5. Temperature-scaled KL divergence teacher distillation from frozen Model A
6. Pre-declared validation checkpoint selection metric on val_split.csv:
   Score_dev = Acc_disease,dev + 0.5 * Acc_crop,dev - 2.0 * Rate_cross-crop,dev
Saves candidate checkpoint to weights/efficientnet_b5_cbam_hierarchical_candidate.pt.
NEVER modifies weights/efficientnet_b5_cbam_best.pt.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM
from models.hierarchical_cbam import EfficientNetB5_CBAM_Hierarchical
from training.hierarchical_dataset import get_hierarchical_loaders

def evaluate_dev(
    model: nn.Module,
    val_loader,
    device: torch.device,
    crop_names: list,
    class_names: list,
    canonical_map: dict
):
    """
    Evaluates candidate model on val_split.csv computing:
    1. Acc_disease,dev (Top-1 canonical disease accuracy)
    2. Acc_crop,dev (Top-1 crop accuracy)
    3. Rate_cross-crop,dev (Cross-crop misprediction rate)
    4. Score_dev
    """
    model.eval()
    total_samples = 0
    correct_crop = 0
    correct_disease = 0
    cross_crop_errors = 0

    crop_to_idx = {c: i for i, c in enumerate(crop_names)}

    with torch.no_grad():
        for images, disease_targets, crop_targets in val_loader:
            images = images.to(device, non_blocking=True)
            disease_targets = disease_targets.to(device, non_blocking=True)
            crop_targets = crop_targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                disease_logits, crop_logits = model(images, return_crop=True)

            # Crop prediction
            _, pred_crops = torch.max(crop_logits, dim=1)
            correct_crop += (pred_crops == crop_targets).sum().item()

            # Disease prediction (aggregated to canonical class)
            _, pred_disease_raw = torch.max(disease_logits, dim=1)
            
            for idx in range(images.size(0)):
                true_c_idx = crop_targets[idx].item()
                pred_c_idx = pred_crops[idx].item()
                true_crop = crop_names[true_c_idx]
                pred_crop = crop_names[pred_c_idx]

                true_d_idx = disease_targets[idx].item()
                pred_d_raw_idx = pred_disease_raw[idx].item()
                
                true_d_name = class_names[true_d_idx]
                pred_d_name = class_names[pred_d_raw_idx]

                true_canon = canonical_map.get(true_d_name, true_d_name)
                pred_canon = canonical_map.get(pred_d_name, pred_d_name)

                # Canonical match
                if true_canon.lower().strip() == pred_canon.lower().strip():
                    correct_disease += 1

                # Cross-crop error: predicted crop differs from true crop
                # (accounting for rice/paddy synonym)
                is_crop_match = (pred_crop == true_crop) or (true_crop in ["rice", "paddy"] and pred_crop in ["rice", "paddy"])
                if not is_crop_match:
                    cross_crop_errors += 1

            total_samples += images.size(0)

    acc_crop = (correct_crop / max(total_samples, 1)) * 100.0
    acc_disease = (correct_disease / max(total_samples, 1)) * 100.0
    rate_cross_crop = (cross_crop_errors / max(total_samples, 1)) * 100.0

    score_dev = acc_disease + (0.5 * acc_crop) - (2.0 * rate_cross_crop)

    return {
        "acc_disease": round(acc_disease, 2),
        "acc_crop": round(acc_crop, 2),
        "rate_cross_crop": round(rate_cross_crop, 2),
        "score_dev": round(score_dev, 2),
        "total_samples": total_samples,
        "correct_crop": correct_crop,
        "correct_disease": correct_disease,
        "cross_crop_errors": cross_crop_errors
    }

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Multi-Task Training")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_crop_head", type=float, default=1e-4, help="LR for newly initialized crop head")
    parser.add_argument("--lr_disease_head", type=float, default=5e-5, help="LR for pre-trained disease head")
    parser.add_argument("--lr_cbam", type=float, default=2e-5, help="LR for CBAM attention")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="LR for upper backbone (Blocks 6-8)")
    parser.add_argument("--lambda_disease", type=float, default=0.70, help="Loss weight for disease classification")
    parser.add_argument("--lambda_crop", type=float, default=0.30, help="Loss weight for crop classification")
    parser.add_argument("--lambda_distill", type=float, default=0.50, help="Loss weight for teacher distillation")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature T")
    parser.add_argument("--model_a_weights", type=str, default="weights/efficientnet_b5_cbam_best.pt", help="Protected baseline")
    parser.add_argument("--output_checkpoint", type=str, default="weights/efficientnet_b5_cbam_hierarchical_candidate.pt", help="Output candidate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      AGRIVISION AI: HIERARCHICAL MULTI-TASK CANDIDATE TRAINING")
    print("=" * 80)
    print(f"Device               : {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Model A (Teacher)    : {args.model_a_weights} (FROZEN)")
    print(f"Candidate Checkpoint : {args.output_checkpoint}")
    print(f"Epochs               : {args.epochs} | Batch Size: {args.batch_size} (Effective: {args.batch_size * args.grad_accum})")
    print(f"Loss Weights         : Disease={args.lambda_disease} | Crop={args.lambda_crop} | Distill={args.lambda_distill} (T={args.temperature})")
    print(f"Learning Rates       : CropHead={args.lr_crop_head} | DiseaseHead={args.lr_disease_head} | CBAM={args.lr_cbam} | Backbone={args.lr_backbone}")
    print("=" * 80)

    # 1. Cryptographic Audit of Protected Baseline
    start_sha = hashlib.sha256(open(args.model_a_weights, "rb").read()).hexdigest()
    start_size = os.path.getsize(args.model_a_weights)
    print(f"[Audit] Model A SHA256: {start_sha}")
    print(f"[Audit] Model A Size  : {start_size:,} bytes\n")

    # 2. Load Taxonomy & Names
    with open("weights/class_names.txt", "r", encoding="utf-8") as f:
        class_names = [l.strip() for l in f if l.strip()]
    num_disease_classes = len(class_names) # 281

    with open("weights/crop_names.txt", "r", encoding="utf-8") as f:
        crop_names = [l.strip() for l in f if l.strip()]
    num_crop_classes = len(crop_names) # 41

    with open("weights/canonical_class_map.json", "r", encoding="utf-8") as f:
        canonical_map = json.load(f)

    # 3. Instantiate Teacher Model A (Frozen)
    print("[Teacher] Instantiating frozen Teacher Model A...")
    teacher = EfficientNetB5_CBAM(num_classes=num_disease_classes, pretrained=False)
    teacher.load_state_dict(torch.load(args.model_a_weights, map_location="cpu", weights_only=False), strict=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("[Teacher] Teacher Model A successfully frozen.\n")

    # 4. Instantiate Student Candidate Model
    print("[Student] Instantiating Hierarchical Candidate Model...")
    student = EfficientNetB5_CBAM_Hierarchical(
        num_disease_classes=num_disease_classes,
        num_crop_classes=num_crop_classes,
        pretrained=False
    )
    student.load_model_a_weights(args.model_a_weights)

    # Layer Freezing Strategy:
    # Blocks 0 to 5: FROZEN
    # Blocks 6 to 8: TRAINABLE
    for i in range(min(6, len(student.features))):
        for p in student.features[i].parameters():
            p.requires_grad = False

    backbone_trainable = []
    for i in range(6, len(student.features)):
        for p in student.features[i].parameters():
            p.requires_grad = True
            backbone_trainable.append(p)

    for p in student.cbam.parameters():
        p.requires_grad = True

    for p in student.classifier.parameters():
        p.requires_grad = True

    for p in student.crop_classifier.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in student.parameters() if not p.requires_grad)
    print(f"[Parameters] Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M) | Frozen: {frozen_params:,} ({frozen_params/1e6:.2f}M)\n")

    student.to(device)

    # 5. Dataloaders
    train_loader, val_loader = get_hierarchical_loaders(
        train_csv="Data/outputs/outputs/train_split.csv",
        val_csv="Data/outputs/outputs/val_split.csv",
        batch_size=args.batch_size
    )

    # 6. Optimizer & Scheduler
    param_groups = [
        {"params": student.crop_classifier.parameters(), "lr": args.lr_crop_head, "weight_decay": 1e-4},
        {"params": student.classifier.parameters(), "lr": args.lr_disease_head, "weight_decay": 1e-4},
        {"params": student.cbam.parameters(), "lr": args.lr_cbam, "weight_decay": 1e-4},
        {"params": backbone_trainable, "lr": args.lr_backbone, "weight_decay": 1e-4}
    ]
    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    ce_loss_disease = nn.CrossEntropyLoss()
    ce_loss_crop = nn.CrossEntropyLoss()
    kl_loss_func = nn.KLDivLoss(reduction="batchmean")

    # Record Pre-Training Configuration
    config_manifest = {
        "experiment": "Hierarchical Multi-Task Training with Teacher Distillation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_checkpoint": args.model_a_weights,
        "teacher_sha256": start_sha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lambda_disease": args.lambda_disease,
        "lambda_crop": args.lambda_crop,
        "lambda_distill": args.lambda_distill,
        "temperature": args.temperature,
        "learning_rates": {
            "crop_head": args.lr_crop_head,
            "disease_head": args.lr_disease_head,
            "cbam": args.lr_cbam,
            "backbone": args.lr_backbone
        },
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "selection_metric_formula": "Score_dev = Acc_disease + 0.5*Acc_crop - 2.0*Rate_cross_crop"
    }
    with open("validation/hierarchical_training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_manifest, f, indent=2)
    print("Pre-training configuration recorded to: validation/hierarchical_training_config.json")

    # 7. Initial Dev Evaluation Before Training
    print("\n[Baseline Dev Benchmark] Evaluating initial warm-start on val_split.csv...")
    init_dev = evaluate_dev(student, val_loader, device, crop_names, class_names, canonical_map)
    print(f" -> Warm Start on Dev Set: Disease Acc={init_dev['acc_disease']:.2f}% | Crop Acc={init_dev['acc_crop']:.2f}% | Cross-Crop={init_dev['rate_cross_crop']:.2f}% | Score_dev={init_dev['score_dev']:.2f}\n")

    history = []
    best_score_dev = init_dev["score_dev"]
    best_epoch = 0

    total_batches = len(train_loader)
    T = args.temperature

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        student.train()
        
        # Keep frozen blocks and BatchNorm in eval mode
        for i in range(min(6, len(student.features))):
            student.features[i].eval()
        for mod in student.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

        running_loss = 0.0
        running_d_loss = 0.0
        running_c_loss = 0.0
        running_kd_loss = 0.0

        optimizer.zero_grad()

        for batch_idx, (images, disease_targets, crop_targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            disease_targets = disease_targets.to(device, non_blocking=True)
            crop_targets = crop_targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                # Student Forward pass
                disease_logits, crop_logits = student(images, return_crop=True)

                # Teacher Forward pass
                with torch.no_grad():
                    teacher_logits = teacher(images)

                # 1. Supervised Disease Loss
                l_disease = ce_loss_disease(disease_logits, disease_targets)

                # 2. Supervised Crop Loss
                l_crop = ce_loss_crop(crop_logits, crop_targets)

                # 3. Temperature-Scaled KL Teacher Distillation Loss
                # p_s = log_softmax(z_s / T), p_t = softmax(z_t / T)
                # L_KD = T^2 * KL(p_t || p_s)
                log_p_student = F.log_softmax(disease_logits / T, dim=1)
                p_teacher = F.softmax(teacher_logits / T, dim=1)
                l_distill = kl_loss_func(log_p_student, p_teacher) * (T ** 2)

                # Multi-Task Combined Loss
                total_loss = (
                    args.lambda_disease * l_disease +
                    args.lambda_crop * l_crop +
                    args.lambda_distill * l_distill
                )

                loss_scaled = total_loss / args.grad_accum

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += total_loss.item()
            running_d_loss += l_disease.item()
            running_c_loss += l_crop.item()
            running_kd_loss += l_distill.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - epoch_start
                avg_l = running_loss / (batch_idx + 1)
                avg_d = running_d_loss / (batch_idx + 1)
                avg_c = running_c_loss / (batch_idx + 1)
                avg_kd = running_kd_loss / (batch_idx + 1)
                print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] Batch [{batch_idx+1:04d}/{total_batches:04d}] - Loss: {avg_l:.4f} (Dis: {avg_d:.4f}, Crop: {avg_c:.4f}, KD: {avg_kd:.4f}) | Elapsed: {elapsed:.0f}s", flush=True)

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        # Validation on val_split.csv
        print(f"\nEvaluating Epoch {epoch} on development validation set...")
        dev_metrics = evaluate_dev(student, val_loader, device, crop_names, class_names, canonical_map)

        print("-" * 80)
        print(f"EPOCH [{epoch:02d}/{args.epochs:02d}] DEV METRICS:")
        print(f"  Disease Acc : {dev_metrics['acc_disease']:.2f}% (Top-1 Canonical)")
        print(f"  Crop Acc    : {dev_metrics['acc_crop']:.2f}% (Top-1 41-Crop)")
        print(f"  Cross-Crop  : {dev_metrics['rate_cross_crop']:.2f}%")
        print(f"  Score_dev   : {dev_metrics['score_dev']:.2f} (Previous Best: {best_score_dev:.2f})")
        print(f"  Duration    : {epoch_duration:.1f}s")
        print("-" * 80, flush=True)

        record = {
            "epoch": epoch,
            "loss_total": round(running_loss / total_batches, 4),
            "loss_disease": round(running_d_loss / total_batches, 4),
            "loss_crop": round(running_c_loss / total_batches, 4),
            "loss_distill": round(running_kd_loss / total_batches, 4),
            "dev_disease_acc": dev_metrics["acc_disease"],
            "dev_crop_acc": dev_metrics["acc_crop"],
            "dev_cross_crop_rate": dev_metrics["rate_cross_crop"],
            "dev_score": dev_metrics["score_dev"],
            "duration_seconds": round(epoch_duration, 1)
        }
        history.append(record)

        # Checkpoint Selection strictly using Score_dev
        if dev_metrics["score_dev"] >= best_score_dev:
            best_score_dev = dev_metrics["score_dev"]
            best_epoch = epoch
            torch.save(student.state_dict(), args.output_checkpoint)
            print(f" >>> [NEW BEST CHECKPOINT] Epoch {epoch}: Score_dev {dev_metrics['score_dev']:.2f}! Saved to {args.output_checkpoint}", flush=True)

    if not os.path.exists(args.output_checkpoint):
        torch.save(student.state_dict(), args.output_checkpoint)
        print(f" >>> [SAVED FINAL CHECKPOINT] Saved final epoch to {args.output_checkpoint}", flush=True)

    # Save History
    training_manifest = {
        "experiment": "Hierarchical Multi-Task Training with Teacher Distillation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": best_epoch,
        "best_score_dev": best_score_dev,
        "output_checkpoint": args.output_checkpoint,
        "epochs": history
    }
    with open("validation/hierarchical_training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, indent=2)
    print(f"\nTraining history successfully saved to validation/hierarchical_training_history.json.")

    # 8. Post-Training Integrity Audit of Protected Baseline
    post_sha = hashlib.sha256(open(args.model_a_weights, "rb").read()).hexdigest()
    post_size = os.path.getsize(args.model_a_weights)
    assert post_sha == start_sha, f"CRITICAL ERROR: Protected baseline {args.model_a_weights} was modified!"
    assert post_size == start_size, f"CRITICAL ERROR: Protected baseline size changed!"
    print(f"\n[Audit PASSED] Protected baseline {args.model_a_weights} is 100% UNTOUCHED and identical.")
    print(f"[Audit PASSED] Candidate checkpoint {args.output_checkpoint} created ({os.path.getsize(args.output_checkpoint):,} bytes).")

if __name__ == "__main__":
    main()
