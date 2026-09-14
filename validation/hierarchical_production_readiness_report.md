# AgriVision AI — Production Readiness Audit Report
**Candidate Architecture**: `EfficientNetB5_CBAM_Hierarchical`
**Evaluation Date**: 2026-09-14 17:27:58
**Protected Baseline (Model A)**: `weights/efficientnet_b5_cbam_best.pt` (SHA256: `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`)
**Evaluated Candidate Checkpoint**: `weights/efficientnet_b5_cbam_hierarchical_candidate.pt`

---

## 1. Executive Summary & Production Verdict

### **VERDICT: `KEEP MODEL A`**

> **Audit Recommendation**: Candidate failed one or more absolute safety, crop accuracy, or cross-crop error reduction gates.

---

## 2. Tier 2: 220-Image External Development Benchmark Comparison

Evaluated on the exact 220 untouched agricultural field samples from `validation/external_cohort_manifest.csv`.

| Evaluation Metric | Protected Model A Baseline | Candidate Pre-Declared Gate | Candidate Actual Result | Delta | Gate Status |
|---|---|---|---|---|---|
| **Top-1 Crop Accuracy** | 74.09% | $\ge 76.09\%$ ($+2.0\%$) | **65.00%** | -9.09% | FAILED |
| **Full Diagnosis Accuracy** | 64.55% | $\ge 66.55\%$ ($+2.0\%$) | **64.09%** | -0.46% | FAILED |
| **Conditional Diag Acc (given crop)** | 87.12% | $\ge 85.62\%$ (no $>1.5\%$ reg) | **98.60%** | +11.48% | PASSED |
| **Cross-Crop Diagnostic Errors** | 30 / 220 (13.64%) | $\le 25$ ($< 11.36\%$, $\ge 5$ reduced) | **59 / 220 (26.82%)** | +29 errors | FAILED |
| **Expected Calibration Error (ECE)** | 0.0819 | $\le 0.1000$ | **0.0873** | +0.0054 | PASSED |
| **High-Confidence Error Rate** | 2 / 220 (0.91%) | $\le 2.00\%$ | **1.82%** | +0.91% | PASSED |
| **Mean Diagnostic Latency** | 682.4ms | $\le 1200.0$ ms | **422.0ms** | — | PASSED |

### Statistical Significance (Paired McNemar Test with Holm-Bonferroni Correction)
- **Top-1 Crop Accuracy**: Model A = 163/220 vs Candidate = 143/220
  - 95% Wilson CIs: Model A `[67.92%, 79.43%]` vs Candidate `[58.49%, 71.0%]`
  - Contingency Table: Both Correct=137, Model A Only=26, Candidate Only=6, Both Wrong=51
  - McNemar $\chi^2 = 11.2812$, Raw $p = 0.0008$, Holm-Adjusted $p = 0.0016$ (Statistically Significant)
- **Full Diagnosis Accuracy**: Model A = 144/220 vs Candidate = 141/220
  - 95% Wilson CIs: Model A `[58.95%, 71.42%]` vs Candidate `[57.56%, 70.14%]`
  - McNemar $\chi^2 = 0.129$, Raw $p = 0.7194$, Holm-Adjusted $p = 0.7194$ (Not Statistically Significant)
- **Cross-Crop Diagnostic Errors**: Model A = 30/220 errors (13.64%) vs Candidate = 59/220 errors (26.82%)
  - 95% Wilson CIs: Model A `[9.72%, 18.8%]` vs Candidate `[21.4%, 33.03%]`
  - McNemar $\chi^2 = 20.1026$, Raw $p = 0.0000$, Holm-Adjusted $p = 0.0000$ (Statistically Significant)

---

## 3. Tier 3: Defensive Stress & Historical Regression Suites

- **Defensive Stress Suite (30 samples)**:
  - Total Samples: 30
  - Intercepted by Pipeline Safeguards (Blur / Low Lighting / OOD Gate / Unknown Crop): **30 / 30 (100.0%)**
  - Chemical Sprays Suppressed: 100.0%
  - Gate Status: **PASSED (100% Interception)**
- **Historical Regression Suite (34 samples)**:
  - Regressions Detected: 0
  - Gate Status: **PASSED**

---

## 4. Tier 4: Sealed Acceptance Cohort Evaluation (150 Images)

**Protocol**: Post-Unseal Zero-Tuning Lock enforced. Evaluated side-by-side on fresh, unseen real-world images from `validation/sealed_acceptance_cohort_manifest.csv`.


| Evaluation Metric | Model A Baseline (150 imgs) | Candidate Model (150 imgs) | Delta | Statistical Comparison (Paired McNemar) | Acceptance Gate |
|---|---|---|---|---|---|
| **Top-1 Crop Accuracy** | 66.00% (95% CI: [58.1%, 73.1%]) | **67.33%** (95% CI: [59.48%, 74.32%]) | +1.33% | $\chi^2 = 0.0455$, Holm $p = 1.0000$ | PASSED (Preserved/Improved) |
| **Full Diagnosis Accuracy** | 69.33% (95% CI: [61.55%, 76.15%]) | **70.67%** (95% CI: [62.94%, 77.36%]) | +1.34% | $\chi^2 = 0.05$, Holm $p = 1.0000$ | PASSED (Preserved/Improved) |
| **Cross-Crop Errors** | 31 / 150 (20.67%) (95% CI: [14.96%, 27.84%]) | **39 / 150 (26.00%)** (95% CI: [19.64%, 33.56%]) | +8 errors | $\chi^2 = 1.8846$, Holm $p = 0.5094$ | FAILED |
| **Expected Calib Error (ECE)** | 0.1271 | **0.1237** | -0.0034 | — | PASSED |

---

## 5. Architectural Invariants & Rollback Integrity Audit

1. **Protected Baseline Checkpoint**:
   - Path: `weights/efficientnet_b5_cbam_best.pt`
   - Initial SHA256: `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`
   - File Size: `121,252,187 bytes`
   - Post-Audit Status: **100% UNTOUCHED and BIT-FOR-BIT IDENTICAL**
2. **Production Server Status**:
   - Port 8000 live FastAPI server was NEVER interrupted or switched away from Model A during the entire lifecycle.
3. **Sealed Holdout Isolation**:
   - `validation/sealed_acceptance_cohort_manifest.csv` was never accessed during training, loss tuning, or hyperparameter selection.
