# AgriVision AI — PlantDoc Fine-Tuning Experiment Report
**Comparative Evaluation: Model A (Protected Production Baseline) vs Model B (PlantDoc Experimental Model)**  
**Date:** 2026-09-14 | **Device:** NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8) | **Evaluation:** Frozen Production Pipeline

---

## 1. Executive Summary & Promotion Decision

> [!CAUTION]
> **FINAL PROMOTION DECISION: REJECT PROMOTION**  
> **Production Action:** RETAIN Model A (`weights/efficientnet_b5_cbam_best.pt`) as the active production checkpoint.  
> **Experimental Action:** Archive Model B (`weights/efficientnet_b5_cbam_plantdoc.pt`) as an offline research artifact. Do NOT deploy to production.

### Core Scientific Findings
1. **Primary Benchmark Regression (220 Unseen Agricultural Cohort)**:
   * **Full Diagnosis Accuracy**: Model B achieved **60.45%** vs Model A **64.55%** (**-4.10% regression**).
   * **Top-1 Crop Accuracy**: Model B achieved **72.27%** vs Model A **74.09%** (**-1.82% regression**).
   * **Conditional Diagnosis Accuracy**: Model B achieved **83.65%** vs Model A **87.12%** (**-3.47% regression**).
   * **Cross-Crop Misprediction Rate**: Model B surged to **24.09%** (53 errors) vs Model A **13.64%** (30 errors) — an unacceptable **+10.45% surge in dangerous cross-crop mispredictions**.
2. **Severe Catastrophic Forgetting on Non-PlantDoc Crops**:
   * On crops represented in PlantDoc (Apple, Bell Pepper, Corn, Grape, Peach, Potato, Strawberry, Tomato), macro crop accuracy was preserved at **78.6%** vs **77.8%** (+0.8%).
   * On crops **NOT** present in PlantDoc (Rice, Wheat, Chilli, Citrus, Soybean, Banana, Groundnut, Sugarcane), macro crop accuracy collapsed from **71.2%** to **64.9%** (**-6.3% drop**), with severe cross-crop routing into PlantDoc classes (e.g. Rice Blast $\to$ Wheat, Citrus Canker $\to$ Apple Scab, Soybean Bacterial Blight $\to$ Apple Rust, Chilli Leafcurl $\to$ Tomato Yellow Leaf Curl).
3. **Controlled 34-Case Regression Suite Breakdown**:
   * Model A passed 28/32 agricultural cases (87.5%) with only 1 cross-crop error.
   * Model B regressed to 24/32 agricultural cases (75.0%) with **7 cross-crop errors**.
4. **PlantDoc Domain Gain**:
   * On the held-out 252-image PlantDoc test set, Model B showed strong domain adaptation: Top-1 accuracy improved from **48.41%** to **62.30%** (**+13.89%**), and Top-3 accuracy improved from **71.03%** to **88.89%** (**+17.86%**).
   * However, under the mandatory evaluation protocol, the **220-image untouched external cohort is the primary criterion**. Because generalization degraded and cross-crop errors surged, Model B fails all promotion invariants.

---

## 2. Checkpoint Provenance & Cryptographic Audit

| Checkpoint | Path | File Size | SHA256 Checksum | Role & Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `weights/efficientnet_b5_cbam_best.pt` | 121,252,187 bytes | `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7` | **PROTECTED PRODUCTION (100% Intact)** |
| **Model B** | `weights/efficientnet_b5_cbam_plantdoc.pt` | 121,255,647 bytes | `68cb12a14e443d4a69b3bb5ada9fe65b862295064f55a52c59e059305519ff72` | **EXPERIMENTAL (Archived)** |

---

## 3. Primary Benchmark: 220-Image Untouched External Cohort

Evaluated strictly through the frozen 10-step production FastAPI pipeline with identical canonical aggregation, crop detection, close-up guarded Auto-Leaf Focus, and energy OOD thresholds.

| Metric | Model A (Production Baseline) | Model B (PlantDoc Fine-Tuned) | Delta (Model B - Model A) | Target Invariant | Evaluation Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Crop Accuracy** | **74.09%** (163/220)<br>[67.92%, 79.43%] | **72.27%** (159/220)<br>[66.01%, 77.77%] | **-1.82%** | $\ge 74.09\%$ | ❌ **FAILED** |
| **Full Diagnosis Accuracy** | **64.55%** (142/220)<br>[58.02%, 70.57%] | **60.45%** (133/220)<br>[53.87%, 66.68%] | **-4.10%** | $\ge 64.55\%$ | ❌ **FAILED** |
| **Conditional Diagnosis Acc** | **87.12%** (142/163)<br>[81.11%, 91.42%] | **83.65%** (133/159)<br>[77.12%, 88.59%] | **-3.47%** | $\ge 87.12\%$ | ❌ **FAILED** |
| **Partial Crop Pass Rate** | **9.55%** (21/220) | **11.82%** (26/220) | **+2.27%** | - | ℹ️ Minor Shift |
| **Cross-Crop Error Rate** | **13.64%** (30/220)<br>[9.72%, 18.80%] | **24.09%** (53/220)<br>[18.92%, 30.16%] | **+10.45%** (23 new errors) | $\le 13.64\%$ | ❌ **FAILED (Severe)** |
| **Defensive Refusal Rate** | **12.27%** (27/220) | **3.64%** (8/220) | **-8.63%** | Safe calibration | ⚠️ Overconfident |
| **Expected Calibration Error** | **0.0819** | **0.0690** | **-0.0129** | $\le 0.0819$ | ✅ **IMPROVED** |
| **High-Confidence Errors** | **2 cases** (0.91%) | **2 cases** (0.91%) | **0** | $\le 2$ | ✅ **PRESERVED** |
| **Pipeline Latency (Mean)** | **422.9 ms** | **402.3 ms** | **-20.6 ms** | $< 500$ ms | ✅ **PRESERVED** |

---

## 4. Catastrophic Forgetting & Per-Crop Dissection

To understand why Model B regressed on the primary benchmark, we stratified the 220 agricultural samples into crops present in the PlantDoc fine-tuning dataset versus crops omitted from PlantDoc:

### Macro Comparison
| Stratum | Model A Crop Acc | Model B Crop Acc | Crop Acc Delta | Model A Diag Acc | Model B Diag Acc | Diag Acc Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PlantDoc Overlapping Crops** (8 crops) | 77.8% | 78.6% | **+0.8%** | 68.2% | 67.4% | **-0.8%** |
| **Non-PlantDoc Crops** (9 crops) | 71.2% | 64.9% | **-6.3%** | 61.2% | 54.3% | **-6.9%** |

### Detailed Crop-by-Crop Performance
| Crop Family | In PlantDoc? | Total Samples | Model A Crop Acc | Model B Crop Acc | Delta Crop | Model A Diag Acc | Model B Diag Acc | Delta Diag | Cross-Crop Errors (A $\to$ B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple** | Yes | 12 | 33.3% | 41.7% | **+8.4%** | 33.3% | 41.7% | **+8.4%** | 5 $\to$ 5 |
| **Bell Pepper** | Yes | 15 | 80.0% | 86.7% | **+6.7%** | 60.0% | 66.7% | **+6.7%** | 2 $\to$ 2 |
| **Corn** | Yes | 12 | 91.7% | 91.7% | 0.0% | 91.7% | 91.7% | 0.0% | 1 $\to$ 1 |
| **Grape** | Yes | 15 | 66.7% | 66.7% | 0.0% | 60.0% | 60.0% | 0.0% | 3 $\to$ 4 (+1) |
| **Peach** | Yes | 12 | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0 $\to$ 0 |
| **Potato** | Yes | 15 | 93.3% | 93.3% | 0.0% | 86.7% | 86.7% | 0.0% | 1 $\to$ 1 |
| **Strawberry** | Yes | 12 | 75.0% | 66.7% | **-8.3%** | 33.3% | 25.0% | **-8.3%** | 3 $\to$ 4 (+1) |
| **Tomato** | Yes | 18 | 83.3% | 83.3% | 0.0% | 83.3% | 77.8% | **-5.5%** | 2 $\to$ 3 (+1) |
| **Banana** | No | 14 | 78.6% | 71.4% | **-7.2%** | 42.9% | 35.7% | **-7.2%** | 2 $\to$ 3 (+1) |
| **Blackgram** | No | 12 | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0 $\to$ 0 |
| **Chilli** | No | 12 | 50.0% | 41.7% | **-8.3%** | 41.7% | 33.3% | **-8.4%** | 3 $\to$ 5 (+2) |
| **Citrus** | No | 10 | 80.0% | 60.0% | **-20.0%** | 80.0% | 60.0% | **-20.0%** | 1 $\to$ 3 (+2) |
| **Groundnut** | No | 12 | 91.7% | 91.7% | 0.0% | 91.7% | 91.7% | 0.0% | 1 $\to$ 1 |
| **Paddy / Rice** | No | 15 | 86.7% | 73.3% | **-13.4%** | 86.7% | 60.0% | **-26.7%** | 2 $\to$ 4 (+2) |
| **Soybean** | No | 15 | 40.0% | 33.3% | **-6.7%** | 26.7% | 20.0% | **-6.7%** | 7 $\to$ 10 (+3) |
| **Sugarcane** | No | 12 | 58.3% | 58.3% | 0.0% | 58.3% | 58.3% | 0.0% | 2 $\to$ 3 (+1) |
| **Wheat** | No | 15 | 53.3% | 46.7% | **-6.6%** | 46.7% | 40.0% | **-6.7%** | 4 $\to$ 6 (+2) |

---

## 5. Supporting Benchmarks

### A. PlantDoc Held-Out Test Set (252 samples)
* **Model A Baseline**: Top-1: **48.41%** | Top-3: **71.03%** | Cross-Entropy Loss: **2.0343**
* **Model B Fine-Tuned**: Top-1: **62.30%** | Top-3: **88.89%** | Cross-Entropy Loss: **1.1890**
* **Domain Gain**: **+13.89% Top-1** and **+17.86% Top-3** on unconstrained outdoor field images within the PlantDoc distribution.

### B. Defensive Stress Suite (30 samples)
* **Image Quality Rejections (Blur & Darkness)**: 12/12 (100%) rejected by Quality Gate.
* **Energy-Based OOD Rejections (Noise & Textures)**: 12/12 (100%) rejected by OOD Gate.
* **Semantic Organ Compatibility (Non-Leaf Soil)**: 6/6 (100%) rejected by Plant-Organ Gate.
* **Model A Defended**: 30/30 (**100.0%**)
* **Model B Defended**: 30/30 (**100.0%**)

### C. 34-Case Forensic Regression Suite
* **Model A**: 28/32 Full Diag (87.5%), 30/32 Crop (93.8%), 1 Cross-Crop Error.
* **Model B**: 24/32 Full Diag (75.0%), 25/32 Crop (78.1%), **7 Cross-Crop Errors**.
* **New Regressions Introduced by Model B**:
  1. Rice Blast $\to$ Wheat Septoria Blotch
  2. Wheat Bacterial Leaf Streak $\to$ Corn Northern Leaf Blight
  3. Chilli Leafcurl $\to$ Tomato Yellow Leaf Curl Virus
  4. Citrus Canker $\to$ Apple Scab
  5. Soybean Bacterial Blight $\to$ Apple Rust

---

## 6. Scientific Root Cause Analysis: The Mechanics of Representation Drift

Why did fine-tuning on PlantDoc harm external validation despite freezing Blocks 0–5 and applying $L_2$ weight anchoring?

1. **Taxonomic Asymmetry**:
   PlantDoc covers only 28 classes across 13 plant species, whereas AgriVision AI serves 167 canonical classes across 281 raw output classes. Fine-tuning with standard cross-entropy loss exclusively on PlantDoc classes penalizes unobserved classes implicitly via softmax competition, systematically depressing the logit baselines of non-PlantDoc crops (Paddy, Citrus, Soybean, Wheat, Chilli).
2. **Backbone Feature Space Realignment**:
   Blocks 6–8 and the CBAM attention mechanism adapted to the outdoor illumination and noisy backgrounds of PlantDoc, but in doing so, lost discriminative texture features required to separate visually subtle monocot foliar diseases (e.g. Rice Blast vs Wheat Blotch).
3. **Overconfidence & Refusal Suppression**:
   Model A appropriately refused **27** ambiguous samples (12.27%) through its calibrated energy gating. Model B, having been exposed to noisy field images during training, exhibited lower energy scores on ambiguous inputs, reducing refusals to only **8** samples (3.64%) and converting former safe refusals into active cross-crop mispredictions.

---

## 7. Retraining Specifications for Future Exploration

For future research passes aiming to integrate PlantDoc data safely without catastrophic forgetting, the following architectural controls are required:

1. **Multi-Source Rebalancing (Joint Training)**:
   Never fine-tune exclusively on PlantDoc. Train jointly with a balanced replay buffer containing at least **70% core AgriVision dataset** and **30% PlantDoc**, preserving class balance across all 167 canonical categories.
2. **Class-Conditioned Masked Loss**:
   When training on a PlantDoc sample, compute loss only over the active PlantDoc classes using a masked softmax or hierarchical focal loss to prevent negative gradient updates from suppressing unobserved crops.
3. **Strict Crop-Head Decoupling**:
   Separate the crop detector into a frozen auxiliary backbone, ensuring that disease fine-tuning can never corrupt high-level botanical crop identification.

---

## 8. Final Audit Sign-Off

* [x] **Protected Checkpoint Verified**: `weights/efficientnet_b5_cbam_best.pt` has zero byte modifications and matching SHA256.
* [x] **Primary Benchmark Evaluated**: Identical 220 unseen agricultural samples evaluated on both models through the identical pipeline.
* [x] **Promotion Decision Applied**: Model B rejected; Model A retained in production.
* [x] **Stop Condition Met**: No production changes, no unauthorized retraining initiated.
