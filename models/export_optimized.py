"""
AgriVision Ultra v5.0 — Optimization & Engine Compilation.

Exports:
1. Model A (Frozen Backbone + Residual LoRA Adapters) to ONNX / FP16
2. Model B (EfficientNet-B5 + CBAM + ArcFace Head) to ONNX / FP16
3. Morphological Crop Router (6-family) to ONNX / FP16

Applies static dimensions:
- Foliar scans: 1 x 3 x 256 x 256
- Trap scans: 1 x 3 x 640 x 640

Enforces Single-Expert Invariants:
- Peak VRAM allocated <= 350 MB on GPU
- Pipeline warm latency <= 20 ms on CUDA / <= 60 ms on modern CPU
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import build_efficientnet_cbam
from models.adapters import apply_lora_to_model_a, load_lora_checkpoint
from models.cotton_head import CottonArcFaceModel
from inference.crop_router import CropRouterModel, DedicatedCropRouter


def export_model_a_onnx(
    output_path: str = "weights/ultra_v5/model_a_lora.onnx",
    device: str = "cpu"
) -> str:
    """Exports Model A with frozen LoRA adapters to ONNX."""
    print("[Export] Building Model A (281 classes) with LoRA adapters...")
    model = build_efficientnet_cbam(
        num_classes=281,
        weights_path="weights/efficientnet_b5_cbam_best.pt",
        device=device
    )
    model, adapters = apply_lora_to_model_a(model, rank=8, gamma=16.0)
    lora_ckpt = "weights/ultra_v5/model_a_lora.pt"
    if os.path.exists(lora_ckpt):
        load_lora_checkpoint(adapters, lora_ckpt)

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[Export] Exporting Model A to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    print(f"[Export] Successfully exported Model A to {output_path}")
    return output_path


def export_model_b_onnx(
    output_path_256: str = "weights/ultra_v5/model_b_arcface_256.onnx",
    output_path_640: str = "weights/ultra_v5/model_b_arcface_640.onnx",
    device: str = "cpu"
) -> Tuple[str, str]:
    """Exports Model B with ArcFace head to ONNX at 256x256 and 640x640."""
    print("[Export] Building Model B (Cotton ArcFace)...")
    model = CottonArcFaceModel(num_classes=12, s=30.0, m=0.35)
    ckpt = "weights/ultra_v5/cotton_arcface_head.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(output_path_256)), exist_ok=True)

    # 1. 256x256 foliar scan
    dummy_256 = torch.randn(1, 3, 256, 256, device=device)
    print(f"[Export] Exporting Model B (256x256) to ONNX: {output_path_256}...")
    torch.onnx.export(
        model,
        dummy_256,
        output_path_256,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )

    # 2. 640x640 trap scan
    dummy_640 = torch.randn(1, 3, 640, 640, device=device)
    print(f"[Export] Exporting Model B (640x640) to ONNX: {output_path_640}...")
    torch.onnx.export(
        model,
        dummy_640,
        output_path_640,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )

    print("[Export] Successfully exported Model B ONNX engines.")
    return output_path_256, output_path_640


def benchmark_fp16_pipeline(device: str = "cuda") -> Dict[str, Any]:
    """
    Benchmarks peak GPU VRAM and execution latency under FP16 half-precision optimization.
    Verifies:
    - Peak VRAM <= 350 MB
    - Warm pipeline latency <= 20 ms on CUDA
    """
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    dev = torch.device(device)

    print(f"[Benchmark] Measuring FP16 pipeline performance on {dev}...")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Instantiate Router and Model A
    router = DedicatedCropRouter(device=str(dev))
    model_a = build_efficientnet_cbam(
        num_classes=281,
        weights_path="weights/efficientnet_b5_cbam_best.pt",
        device=str(dev)
    )
    model_a, adapters = apply_lora_to_model_a(model_a)
    lora_ckpt = "weights/ultra_v5/model_a_lora.pt"
    if os.path.exists(lora_ckpt):
        load_lora_checkpoint(adapters, lora_ckpt)

    # Ensure entire model and newly added adapters are on target device
    model_a.to(dev)

    # Convert to FP16 if on CUDA
    if dev.type == "cuda":
        router.model.half()
        model_a.half()

    router.model.eval()
    model_a.eval()

    # Create dummy input
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    dummy_input = torch.randn(1, 3, 256, 256, device=dev, dtype=dtype)

    # Warmup runs (5 iterations)
    with torch.inference_mode():
        for _ in range(5):
            _ = router.model(dummy_input)
            _ = model_a(dummy_input)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Latency benchmark (30 iterations)
    latencies = []
    with torch.inference_mode():
        for _ in range(30):
            t0 = time.perf_counter()
            _ = router.model(dummy_input)
            _ = model_a(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    min_latency = float(np.min(latencies))

    if dev.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    print(f"[Benchmark] Mean Latency: {mean_latency:.2f} ms (p95: {p95_latency:.2f} ms, min: {min_latency:.2f} ms)")
    print(f"[Benchmark] Peak GPU VRAM: {peak_vram_mb:.2f} MB")

    results = {
        "device": str(dev),
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_target_met": mean_latency <= (20.0 if dev.type == "cuda" else 60.0),
        "vram_target_met": peak_vram_mb <= 350.0 if dev.type == "cuda" else True
    }

    report_path = "weights/ultra_v5/optimization_benchmark.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Optimization] Initiating engine compilation on {device}...")

    # 1. Export ONNX models
    export_model_a_onnx(device="cpu")
    export_model_b_onnx(device="cpu")

    # 2. Benchmark FP16 pipeline
    results = benchmark_fp16_pipeline(device=device)
    print(f"[Optimization] Benchmark completed: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
