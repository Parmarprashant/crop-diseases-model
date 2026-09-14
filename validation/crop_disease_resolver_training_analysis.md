# AgriVision AI — Tier 1 Resolver Mode Comparison & Calibration Analysis

## 1. Temperature-Scaled Calibration (Tier 1: `val_split.csv`)
- **Dataset**: `Data/outputs/outputs/val_split.csv` (6,670 samples)
- **Optimal Temperature ($T_{\text{crop}}$)**: **0.9085**
- **Uncalibrated ECE**: 0.0180 $\rightarrow$ **Calibrated ECE**: **0.0080** (**-55.3% reduction**)
- **Top-1 Crop Accuracy**: **97.12%** (Classification invariant strictly preserved)

---

## 2. Multi-Mode Empirical Comparison (Tier 1 Stratified Cohort: 400 samples)

| Configuration Name | Mode Type | Top-1 Crop Acc | Full Diag Acc | Cond Diag Acc | Cross-Crop Errors | ECE | Status |
|---|---|---|---|---|---|---|---|
| **Mode A: Advisory Baseline** | `mode_a` | 72.00% | 59.75% | 82.99% | 98 (24.50%) | 0.0855 | Evaluated |
| **Mode B: Conservative (mass=0.10, dom=0.30)** | `mode_b` | 86.75% | 65.75% | 75.79% | 26 (6.50%) | 0.1102 | Evaluated |
| **Mode B: Balanced Recovery (mass=0.08, dom=0.25)** | `mode_b` | 86.75% | 65.75% | 75.79% | 26 (6.50%) | 0.1108 | **SELECTED (FROZEN)** |
| **Mode B: Sensitive Recovery (mass=0.05, dom=0.20)** | `mode_b` | 87.00% | 65.75% | 75.57% | 25 (6.25%) | 0.1123 | Evaluated |
| **Mode C: Confidence-Weighted Fusion** | `mode_c` | 86.00% | 67.25% | 78.20% | 29 (7.25%) | 0.2098 | Evaluated (High ECE) |

---

## 3. Decision Rationale & Hyperparameter Freeze
- **Selected Architecture**: **Mode B: Balanced Recovery (mass=0.08, dom=0.25)**
- **Key Parameters**:
  - $\tau_{\text{crop\_conf}} = 0.45$
  - $\tau_{\text{compat\_mass}} = 0.08$
  - $\tau_{\text{within\_crop\_dom}} = 0.25$
  - $\tau_{\text{single\_compat}} = 0.05$
- **Agronomic Safety**:
  - Eliminates single-disease probability penalty by evaluating cumulative in-crop evidence ($S_{\text{compat}}$) and conditional dominance.
  - Recovers legitimate Model A diagnoses without increasing cross-crop errors.
  - Frozen into `validation/crop_disease_resolver_config.json`.
