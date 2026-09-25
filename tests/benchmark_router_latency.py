import os
import sys
import time
import json
from PIL import Image
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import DedicatedCropRouter
from inference.model_wrappers import DiseaseExpertWrapper
from inference.pipeline import HierarchicalAgriDiagnosticPipeline


def benchmark():
    print("=" * 60)
    print("AGRIVISION — HIERARCHICAL MULTI-MODEL LATENCY & MEMORY BENCHMARK")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on: {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Initial VRAM Allocated: {vram_start:.2f} MB")
        
    t0 = time.perf_counter()
    router_ckpt = "weights/research/crop_router_v4.pt" if os.path.exists("weights/research/crop_router_v4.pt") else "weights/research/crop_router_v1.pt"
    router = DedicatedCropRouter(
        checkpoint_path=router_ckpt,
        thresholds_path="weights/router_thresholds.json",
        device=device
    )
    t_router_load = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    model_a = DiseaseExpertWrapper(
        model_name="generalist_281",
        checkpoint_path="weights/efficientnet_b5_cbam_best.pt",
        class_names_path="weights/class_names.txt",
        device=device
    )
    t_model_a_load = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    model_b = DiseaseExpertWrapper(
        model_name="cotton_specialist_42",
        checkpoint_path="weights/backup_42class/efficientnet_b5_cbam_best.pt",
        class_names_path="weights/backup_42class/class_names.txt",
        device=device
    )
    t_model_b_load = (time.perf_counter() - t0) * 1000
    
    pipeline = HierarchicalAgriDiagnosticPipeline(
        crop_router=router,
        model_a=model_a,
        model_b=model_b,
        thresholds_path="weights/calibration_thresholds.json",
        enable_gemini=False
    )
    
    if device == "cuda":
        vram_loaded = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\nResident VRAM Allocated: {vram_loaded:.2f} MB ({vram_loaded/1024:.2f} GB)")
        print(f"Resident VRAM Reserved:  {vram_reserved:.2f} MB ({vram_reserved/1024:.2f} GB)")
        print(f"Target (< 2.5 GB Resident): {'PASSED' if vram_loaded < 2560 else 'FAILED'}")
        
    print(f"\nModel Loading Latencies:")
    print(f"  Crop Router: {t_router_load:.1f} ms")
    print(f"  Model A (281): {t_model_a_load:.1f} ms")
    print(f"  Model B (42):  {t_model_b_load:.1f} ms")
    
    # Create test dummy image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    
    # 1. Warm-up
    for _ in range(3):
        _ = router.route(dummy_img)
        _ = model_a.predict(dummy_img)
        _ = model_b.predict(dummy_img)
        
    # 2. Benchmark Router Alone
    n_runs = 20
    times_router = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = router.route(dummy_img)
        times_router.append((time.perf_counter() - t0) * 1000)
    avg_router = sum(times_router) / n_runs
    
    # 3. Benchmark Model A Alone
    times_a = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_a.predict(dummy_img)
        times_a.append((time.perf_counter() - t0) * 1000)
    avg_a = sum(times_a) / n_runs
    
    # 4. Benchmark Model B Alone
    times_b = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_b.predict(dummy_img)
        times_b.append((time.perf_counter() - t0) * 1000)
    avg_b = sum(times_b) / n_runs
    
    # 5. Benchmark Pipeline (Quality Gate + Router + 1 Expert)
    times_pipe = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = pipeline.diagnose(dummy_img, user_crop_hint="cotton")
        times_pipe.append((time.perf_counter() - t0) * 1000)
    avg_pipe = sum(times_pipe) / n_runs
    
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nPeak VRAM During Inference: {peak_vram:.2f} MB ({peak_vram/1024:.2f} GB)")
        
    print(f"\nWarm Inference Latencies (Average over {n_runs} runs):")
    print(f"  Crop Router:             {avg_router:.2f} ms")
    print(f"  Model A (281) Alone:     {avg_a:.2f} ms")
    print(f"  Model B (42) Alone:      {avg_b:.2f} ms")
    print(f"  Router + Expert Pipeline:{avg_pipe:.2f} ms (Single Expert Executed)")
    print("=" * 60)
    
    results = {
        "device": device,
        "vram_allocated_mb": round(vram_loaded, 2) if device == "cuda" else 0.0,
        "vram_reserved_mb": round(vram_reserved, 2) if device == "cuda" else 0.0,
        "peak_vram_mb": round(peak_vram, 2) if device == "cuda" else 0.0,
        "vram_pass": (vram_loaded < 2560) if device == "cuda" else True,
        "latencies_ms": {
            "crop_router": round(avg_router, 2),
            "model_a_alone": round(avg_a, 2),
            "model_b_alone": round(avg_b, 2),
            "router_expert_pipeline": round(avg_pipe, 2)
        }
    }
    
    out_path = "research/cotton286/latency_benchmark_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved latency benchmark report to {out_path}")
    return results


if __name__ == "__main__":
    benchmark()
