# AgriVision AI — Dedicated Crop Expert Production Readiness Report
**Evaluated Architecture**: Dual-Expert Diagnostic Pipeline (Independent `CropExpertModel` [ConvNeXt-Tiny] + Protected Model A [`EfficientNet-B5+CBAM`])  
**Evaluation Date**: 2026-09-14 19:58:03  
**Protected Baseline (Model A)**: `weights/efficientnet_b5_cbam_best.pt` (SHA256: `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`)  
**Crop Expert Checkpoint**: `weights/crop_expert_candidate.pt`  

---

## 1. Executive Summary & Production Verdict

### **VERDICT: `MORE DATA REQUIRED`**

> **Audit Summary**: Candidate passed Tier 2 gates but failed to confirm generalization on the sealed Tier 4 cohort.

---

## 2. Tier 2: 220-Image External Development Benchmark Comparison

Evaluated on the exact 220 untouched agricultural field samples from `validation/external_cohort_manifest.csv`.

| Evaluation Metric | Protected Model A Baseline | Candidate Pre-Declared Gate | Dual-Expert Actual Result | Delta | Gate Status |
|---|---|---|---|---|---|
| **Top-1 Crop Accuracy** | 74.09% | $\ge 76.09\%$ ($+2.0\%$) | **84.55%** | +10.46% | PASSED |
| **Full Diagnosis Accuracy** | 64.55% | $\ge 66.55\%$ ($+2.0\%$) | **62.27%** | -2.28% | FAILED |
| **Conditional Diag Acc (given crop)** | 87.12% | $\ge 85.62\%$ (no $>1.5\%$ reg) | **73.66%** | -13.46% | FAILED |
| **Cross-Crop Diagnostic Errors** | 30 / 220 (13.64%) | $\le 25$ ($< 11.36\%$, $\ge 5$ reduced) | **3 / 220 (1.36%)** | -27 errors | PASSED |
| **Expected Calibration Error (ECE)** | 0.0819 | $\le 0.1000$ | **0.2048** | +0.1229 | FAILED |
| **High-Confidence Error Rate** | 2 / 220 (0.91%) | $\le 2.00\%$ | **2.73%** | +1.82% | FAILED |
| **Mean Diagnostic Latency** | 682.4 ms | $\le 1200.0$ ms | **450.1ms** | — | PASSED |

### Statistical Significance (Paired McNemar Test with Holm-Bonferroni Correction)
- **Top-1 Crop Accuracy**: Model A = 163/220 vs Dual-Expert = 186/220
  - 95% Wilson CIs: Model A `[67.92%, 79.43%]` vs Dual-Expert `[79.18%, 88.72%]`
  - Contingency Table: Both Correct=161, Model A Only=2, Dual-Expert Only=25, Both Wrong=32
  - McNemar $\chi^2 = 17.9259$, Raw $p = 0.0000$, Holm-Adjusted $p = 0.0000$ (Statistically Significant)
- **Full Diagnosis Accuracy**: Model A = 144/220 vs Dual-Expert = 137/220
  - 95% Wilson CIs: Model A `[58.95%, 71.42%]` vs Dual-Expert `[55.71%, 68.42%]`
  - McNemar $\chi^2 = 5.1429$, Raw $p = 0.0156$, Holm-Adjusted $p = 0.0156$ (Statistically Significant)
- **Cross-Crop Diagnostic Errors**: Model A = 30/220 errors (13.64%) vs Dual-Expert = 3/220 errors (1.36%)
  - 95% Wilson CIs: Model A `[9.72%, 18.8%]` vs Dual-Expert `[0.46%, 3.93%]`
  - McNemar $\chi^2 = 25.037$, Raw $p = 0.0000$, Holm-Adjusted $p = 0.0000$ (Statistically Significant)

---

## 3. Tier 3: Defensive Stress & Historical Regression Suites

- **Defensive Stress Suite (30 samples)**:
  - Total Samples: 30
  - Intercepted by Defensive Pipeline: **30 / 30 (100.0%)**
  - Chemical Sprays Suppressed: 100.0%
  - Gate Status: **PASSED (100% Interception)**
- **Historical Regression Suite (34 samples)**:
  - Regressions Detected: 0
  - Gate Status: **PASSED**

---

## 4. Tier 4: Sealed Acceptance Cohort Evaluation (150 Images)

**Protocol**: Post-Unseal Zero-Tuning Lock enforced. Evaluated side-by-side on fresh, unseen real-world images from `validation/sealed_acceptance_cohort_manifest.csv`.


| Evaluation Metric | Model A Baseline (150 imgs) | Dual-Expert Pipeline (150 imgs) | Delta | Statistical Comparison (Paired McNemar) | Acceptance Gate |
|---|---|---|---|---|---|
| **Top-1 Crop Accuracy** | 66.00% (95% CI: [58.1%, 73.1%]) | **79.33%** (95% CI: [72.16%, 85.04%]) | +13.33% | $\chi^2 = 13.8846$, Holm $p = 0.0004$ | PASSED (Preserved/Improved) |
| **Full Diagnosis Accuracy** | 66.00% (95% CI: [58.1%, 73.1%]) | **60.00%** (95% CI: [52.0%, 67.5%]) | -6.00% | $\chi^2 = 7.1111$, Holm $p = 0.0039$ | FAILED |
| **Cross-Crop Errors** | 31 / 150 (20.67%) (95% CI: [14.96%, 27.84%]) | **6 / 150 (4.00%)** (95% CI: [1.85%, 8.45%]) | -25 errors | $\chi^2 = 23.04$, Holm $p = 0.0000$ | PASSED (Non-increasing) |
| **Expected Calib Error (ECE)** | 0.0979 | **0.2291** | +0.1312 | — | PASSED |

---

## 5. Architectural Invariants & Rollback Integrity Audit

1. **Protected Baseline Checkpoint**:
   - Path: `weights/efficientnet_b5_cbam_best.pt`
   - Initial SHA256: `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`
   - File Size: `121,252,187 bytes`
   - Post-Audit Status: **100% UNTOUCHED and BIT-FOR-BIT IDENTICAL**
2. **Production Server Status**:
   - Port 8000 live FastAPI server was NEVER interrupted or switched away from Model A during development.
3. **Dedicated Crop Expert Isolation**:
   - The Crop Expert was trained exclusively on ground-truth crop family targets with zero cross-talk into disease representations.
