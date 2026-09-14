# AgriVision AI — Resolver Forensic Failure & Abstention Audit

## 1. Forensic Taxonomy of Abstentions (Tier 2 Benchmark: 30 Abstentions)

| Abstention Category | Count | Agronomic Rationale & Mechanism |
|---|---|---|
| **1. Correct Safe Refusal** | 3 | Image quality / non-foliar / severe blur correctly intercepted to suppress spray recommendations. |
| **2. Unnecessary Refusal** | 0 | Model A possessed sufficient signal but conservative gating withheld acceptance. |
| **3. Recoverable Disease** | 0 | Within-crop disease candidate was present but suppressed by strict margin. |
| **4. True Crop/Disease Contradiction** | 1 | Crop Expert and Model A pointed to incompatible crop families without resolvable evidence. |
| **5. Model A Disease Uncertainty** | 26 | Model A flat entropy / no confident disease candidate within verified crop family. |
| **6. Crop Calibration Failure** | 0 | Probability overconfidence on incorrect crop family. |
| **7. Taxonomy Mismatch** | 0 | Target pathogen class not present in 167-class canonical taxonomy. |

---

## 2. Critical Confusing Crop Pairs Audit

| Confusing Pair | Cohort Samples Evaluated | Observed Cross-Crop Confusions | Status |
|---|---|---|---|
| **apple <-> peach** | 26 | **0** | ✅ Completely Resolved |
| **soybean <-> blackgram** | 24 | **0** | ✅ Completely Resolved |
| **grape <-> tomato** | 31 | **0** | ✅ Completely Resolved |
| **strawberry <-> raspberry** | 12 | **0** | ✅ Completely Resolved |
| **wheat <-> corn** | 26 | **1** | ⚠️ 1 confusions |
| **rice <-> wheat** | 14 | **0** | ✅ Completely Resolved |
| **chilli <-> tomato** | 31 | **0** | ✅ Completely Resolved |
| **potato <-> tomato** | 36 | **1** | ⚠️ 1 confusions |

---

## 3. Comparison of Diagnostic Trajectory Across Generations
- **Model A Baseline**: Crop Acc 74.09%, Full Diag 64.55%, Cross-Crop Errors 30/220 (Surging chemical spray hazard).
- **Previous Crop Expert (Old Fusion)**: Crop Acc 84.55%, Full Diag 62.27%, Cross-Crop Errors 3/220 (Over-conservative rejection to UNKNOWN).
- **New Crop-Aware Resolver (Mode B)**: Crop Acc 95.91%, Full Diag 68.18%, Cross-Crop Errors 5/220.
