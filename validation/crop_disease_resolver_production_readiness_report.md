# AgriVision AI — Crop-Aware Disease Resolver Production Readiness Report
**Evaluated Architecture**: Decoupled Dual-Expert Pipeline (Calibrated `CropExpertModel` [ConvNeXt-Tiny, $T=validation/crop_disease_resolver_config.json$] + Protected Model A [`EfficientNet-B5+CBAM`] + `CropAwareDiseaseResolver` [Mode B])  
**Evaluation Date**: 2026-09-14 20:36:54  
**Protected Baseline (Model A)**: `weights/efficientnet_b5_cbam_best.pt` (SHA256: `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`, Size: `121,252,187 bytes`)  
**Crop Expert Checkpoint**: `weights/crop_expert_candidate.pt`  

---

## 1. Executive Summary & Production Verdict

### **VERDICT: `MORE DATA REQUIRED`**

> **Audit Summary**: Candidate results require further data expansion before promotion.

---

## 2. Tier 2: 220-Image External Development Benchmark Comparison

| Evaluation Metric | Protected Model A Baseline | Candidate Pre-Declared Gate | Dual-Expert + Resolver Actual | Delta | Gate Status |
|---|---|---|---|---|---|
| **Top-1 Crop Accuracy** | 74.09% | $\ge 76.09\%$ ($+2.0\%$) | **95.91%** | +21.82% | PASSED |
| **Full Diagnosis Accuracy** | 64.55% | $\ge 66.55\%$ ($+2.0\%$) | **68.18%** | +3.63% | PASSED |
| **Conditional Diag Acc (given crop)** | 87.12% | $\ge 85.62\%$ (no $>1.5\%$ reg) | **71.09%** | -16.03% | FAILED |
| **Cross-Crop Diagnostic Errors** | 30 / 220 (13.64%) | $\le 25$ ($< 11.36\%$, $\ge 5$ reduced) | **5 / 220 (2.27%)** | -25 errors | PASSED |
| **Expected Calibration Error (ECE)** | 0.0819 | $\le 0.1000$ | **0.1044** | +0.0225 | FAILED |
| **High-Confidence Error Rate** | 2 / 220 (0.91%) | $\le 2.00\%$ | **3.18%** | +2.27% | FAILED |
| **Abstention Rate** | — | — | **13.64%** | — | Telemetry |
| **Mean Diagnostic Latency** | 682.4 ms | $\le 1200.0$ ms | **482.3ms** (p95: 887.2ms) | — | PASSED |

### Statistical Significance (Paired McNemar Tests with Holm-Bonferroni Correction)
- **Top-1 Crop Accuracy**: McNemar $\chi^2 = 44.1800$, Raw $p = 0.00000$, Holm-Adjusted $p = 0.00000$ (Statistically Significant)
  - 95% Wilson CIs: Model A `[67.92%, 79.43%]` vs Candidate `[92.41%, 97.83%]`
- **Full Diagnosis Accuracy**: McNemar $\chi^2 = 2.5000$, Raw $p = 0.10938$, Holm-Adjusted $p = 0.10938$
  - 95% Wilson CIs: Model A `[58.95%, 71.42%]` vs Candidate `[61.76%, 73.98%]`
- **Cross-Crop Error Avoidance**: McNemar $\chi^2 = 21.3333$, Raw $p = 0.00000$, Holm-Adjusted $p = 0.00000$ (Statistically Significant)
  - 95% Wilson CIs: Model A `[9.72%, 18.8%]` vs Candidate `[0.97%, 5.21%]`

---

## 3. Tier 3: Defensive Stress & Historical Regression Suites

- **Defensive Stress Suite (30 samples)**:
  - Intercepted by Defensive Pipeline: **30 / 30 (100.0%)**
  - Chemical Sprays Suppressed: 100.0%
  - Gate Status: **PASSED**
- **Historical Regression Suite (34 samples)**:
  - Regressions Detected: **0**
  - Gate Status: **PASSED**

---

## 4. Tier 4: Sealed Final Acceptance Cohort Evaluation (150 Images)

**Protocol**: Post-Unseal Zero-Tuning Lock enforced. Evaluated side-by-side on fresh, unseen real-world images from `validation/sealed_acceptance_cohort_manifest.csv`.

| Evaluation Metric | Model A Baseline (150 imgs) | Resolver Candidate (150 imgs) | Delta | Statistical Comparison (Paired McNemar) | Acceptance Gate |
|---|---|---|---|---|---|
| **Top-1 Crop Accuracy** | 66.00% (95% CI: [58.1%, 73.1%]) | **90.67%** (95% CI: [84.94%, 94.36%]) | +24.67% | $\chi^2 = 30.1395$, Holm $p = 0.0000$ | PASSED |
| **Full Diagnosis Accuracy** | 66.00% (95% CI: [58.1%, 73.1%]) | **70.67%** (95% CI: [62.94%, 77.36%]) | +4.67% | $\chi^2 = 5.1429$, Holm $p = 0.0156$ | PASSED |
| **Cross-Crop Errors** | 31 / 150 (20.67%) (95% CI: [14.96%, 27.84%]) | **10 / 150 (6.67%)** (95% CI: [3.66%, 11.84%]) | -21 errors | $\chi^2 = 14.8148$, Holm $p = 0.0002$ | PASSED |
| **Expected Calib Error (ECE)** | 0.1035 | **0.1570** | +0.0535 | — | AUDITED |

---

## 5. Architectural Invariants & Rollback Integrity Audit

1. **Protected Baseline Checkpoint**:
   - Path: `weights/efficientnet_b5_cbam_best.pt`
   - Initial SHA256: `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`
   - Expected Size: `121,252,187 bytes`
   - Audit Status: **100% BIT-FOR-BIT IDENTICAL**
2. **Production Server Status**:
   - Port 8000 live FastAPI server was NEVER interrupted or switched away from Model A during development.
3. **Dedicated Crop Expert Checkpoint**:
   - Path: `weights/crop_expert_candidate.pt` (Size: `111,489,779 bytes`, frozen).
