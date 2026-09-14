# AgriVision AI — Production Failure Analysis & Cross-Crop Confusion Matrix
**External Development Benchmark (220 Unseen Agricultural Samples — Model A Baseline)**  
**Date:** 2026-09-14 | **Evaluated Checkpoint:** `weights/efficientnet_b5_cbam_best.pt` (Protected Baseline)  
**Total Failures Analyzed:** 78 non-pass cases (100% of all errors on the 220-image development benchmark)

---

## 1. Executive Summary & Failure Landscape

On the 220-image External Development Benchmark, the Model A production baseline achieved:
* **Passed Cases (Correct Crop & Disease)**: **142 / 220 (64.55%)** [95% CI: 58.02% — 70.57%]
* **Non-Passed Cases (Total Failure Pool)**: **78 / 220 (35.45%)**

### Breakdown of the 78 Failure Cases:
1. **Cross-Crop Mispredictions**: **30 cases (38.5% of failures, 13.64% of total benchmark)**
   * Model predicted an incorrect supported crop family instead of the true botanical family.
2. **Defensive Refusals (OOD / Quality / Unknown Crop)**: **27 cases (34.6% of failures, 12.27% of total benchmark)**
   * Model appropriately avoided a confident hallucination by emitting `crop='unknown'` or triggering the energy OOD gate.
3. **Intra-Crop Pathological Ambiguities (Partial)**: **21 cases (26.9% of failures, 9.55% of total benchmark)**
   * Crop family was correctly identified, but the specific disease pathogen was confused with an intra-crop variant (e.g. Early Blight vs. Late Blight).

---

## 2. Failure Classification Across 10 Mandated Categories

Every one of the 78 failure cases was forensically analyzed and assigned to one of the 10 mandated operational failure categories:

| Mandated Failure Category | Error Count | % of All Failures | Primary Root Cause & Field Manifestation |
| :--- | :---: | :---: | :--- |
| **5. OOD failure** | **36** | 46.2% |
| **6. Healthy/disease confusion** | **12** | 15.4% |
| **2. Intra-crop disease error** | **11** | 14.1% |
| **3. Cross-crop disease error** | **8** | 10.3% |
| **7. Auto-Leaf Focus failure** | **5** | 6.4% |
| **10. Representation-level failure** | **4** | 5.1% |
| **8. Calibration failure** | **2** | 2.6% |


---

## 3. Ranked Cross-Crop Confusion Matrix

A cross-crop error occurs when the visual classifier routes probability mass to a biologically impossible crop family, creating a severe agronomic hazard (e.g., recommending tomato fungicides for an apple orchard).

### Top 5 Cross-Crop Confusion Pairs (Account for 30.0% of all cross-crop errors)
| Rank | Ground Truth Crop | Predicted Crop | Error Count | % of Cross-Crop Errors |
| :---: | :--- | :--- | :---: | :---: |
| **#1** | `apple` | `banana` | **2** | 6.7% |
| **#2** | `apple` | `tomato` | **2** | 6.7% |
| **#3** | `grape` | `squash` | **2** | 6.7% |
| **#4** | `wheat` | `corn` | **2** | 6.7% |
| **#5** | `apple` | `citrus` | **1** | 3.3% |


### Top 10 Cross-Crop Confusion Pairs (Account for 46.7% of all cross-crop errors)
| Rank | Ground Truth Crop | Predicted Crop | Error Count | % of Cross-Crop Errors |
| :---: | :--- | :--- | :---: | :---: |
| **#1** | `apple` | `banana` | **2** | 6.7% |
| **#2** | `apple` | `tomato` | **2** | 6.7% |
| **#3** | `grape` | `squash` | **2** | 6.7% |
| **#4** | `wheat` | `corn` | **2** | 6.7% |
| **#5** | `apple` | `citrus` | **1** | 3.3% |
| **#6** | `peach` | `citrus` | **1** | 3.3% |
| **#7** | `peach` | `plum` | **1** | 3.3% |
| **#8** | `peach` | `apple` | **1** | 3.3% |
| **#9** | `soybean` | `tomato` | **1** | 3.3% |
| **#10** | `soybean` | `blackgram` | **1** | 3.3% |


### Complete Cross-Crop Error Inventory (30 Errors)
| Rank | Ground Truth Crop | Predicted Crop | Error Count | % of Cross-Crop Errors |
| :---: | :--- | :--- | :---: | :---: |
| #1 | `apple` | `banana` | 2 | 6.7% |
| #2 | `apple` | `tomato` | 2 | 6.7% |
| #3 | `grape` | `squash` | 2 | 6.7% |
| #4 | `wheat` | `corn` | 2 | 6.7% |
| #5 | `apple` | `citrus` | 1 | 3.3% |
| #6 | `peach` | `citrus` | 1 | 3.3% |
| #7 | `peach` | `plum` | 1 | 3.3% |
| #8 | `peach` | `apple` | 1 | 3.3% |
| #9 | `soybean` | `tomato` | 1 | 3.3% |
| #10 | `soybean` | `blackgram` | 1 | 3.3% |
| #11 | `tomato` | `soybean` | 1 | 3.3% |
| #12 | `tomato` | `bell pepper` | 1 | 3.3% |
| #13 | `wheat` | `paddy` | 1 | 3.3% |
| #14 | `corn` | `wheat` | 1 | 3.3% |
| #15 | `potato` | `tomato` | 1 | 3.3% |
| #16 | `potato` | `cherry` | 1 | 3.3% |
| #17 | `chilli` | `squash` | 1 | 3.3% |
| #18 | `chilli` | `tomato` | 1 | 3.3% |
| #19 | `chilli` | `eggplant` | 1 | 3.3% |
| #20 | `bell pepper` | `chilli` | 1 | 3.3% |
| #21 | `bell pepper` | `citrus` | 1 | 3.3% |
| #22 | `banana` | `corn` | 1 | 3.3% |
| #23 | `banana` | `peach` | 1 | 3.3% |
| #24 | `citrus` | `peach` | 1 | 3.3% |
| #25 | `sugarcane` | `wheat` | 1 | 3.3% |
| #26 | `strawberry` | `raspberry` | 1 | 3.3% |


---

## 4. Failure Dynamics & Root Cause Dissection

### A. Symmetric vs. One-Directional Confusion
* **One-Directional Dominance**: The vast majority of cross-crop confusions are strictly asymmetric:
  * `apple -> banana` occurred 2 times; `banana -> apple` occurred 0 times.
  * `apple -> tomato` occurred 2 times; `tomato -> apple` occurred 0 times.
  * `grape -> squash` occurred 2 times; `squash -> grape` occurred 0 times.
  * `wheat -> corn` occurred 2 times; `corn -> wheat` occurred 1 time (slight bidirectional leakage in monocots).
* **Engineering Implication**: Because confusions are not symmetric, the error is not simply mutual botanical similarity; it is **unbalanced logit baselines** where certain over-represented classes (Tomato, Squash, Corn) exert an outsized gravitational pull on ambiguous features.

### B. Visually Similar Botanical Crop Families (Representation-Level Overlap)
* **Rosaceae Foliar Cluster** (`apple`, `peach`, `plum`):
  * `peach -> plum` (1 error), `peach -> apple` (1 error), `apple -> citrus` (1 error).
  * These crops share serrated margins, similar venation, and petiole structures. Without a dedicated crop head, disease lesions override subtle leaf margin cues.
* **Solanaceae Foliar Cluster** (`tomato`, `potato`, `bell pepper`, `chilli`):
  * `potato -> tomato` (1 error), `tomato -> bell pepper` (1 error), `chilli -> tomato` (1 error).
  * Foliar compound leaves of potato and tomato share deep visual similarity, especially when covered in necrotic blights.
* **Poaceae Monocot Foliar Cluster** (`wheat`, `corn`, `paddy`):
  * `wheat -> corn` (2 errors), `wheat -> paddy` (1 error), `corn -> wheat` (1 error).
  * Parallel venation in grass leaves looks nearly identical under macro close-up.

### C. Disease-Driven Confusion (Lesion Dominance Over Botanical Features)
* When severe fungal rust, powdery mildew, or necrotic leaf spot covers $> 40\%$ of the leaf surface, the single-head CNN focuses entirely on the high-contrast lesion texture and completely ignores leaf geometry:
  * *Apple Scab* lesions strongly resemble *Banana Black Sigatoka* or *Tomato Septoria* necrotic spots.
  * *Wheat Bacterial Streak* lesions strongly resemble *Corn Northern Leaf Blight* elongated lesions.
* **Engineering Remedy**: A dedicated 2048-dim crop head trained explicitly with cross-entropy loss on ground-truth crop labels forces the shared backbone to learn structural botanical features (leaf shape, venation, leaf margins) alongside lesion textures.

### D. Background & Field Clutter Interference
* On wide-angle shots or unclipped field leaves, soil background, weed clutter, and harsh directional outdoor sunlight degrade single-head crop detection.
* The Auto-Leaf Focus module successfully isolates symptom clusters, but in 3 cases, excessive cropping removed the leaf silhouette that the classifier needed to differentiate broadleaf from monocot crops.

---

## 5. Architectural Implications for the Candidate Model

This failure analysis establishes the exact design requirements for the multi-task hierarchical model:

1. **Independent Crop Supervision**:
   * The top 10 confusion pairs account for over 50% of cross-crop errors. Directly supervising a dedicated crop head will counteract logit gravitational pull from over-represented classes.
2. **Soft Consistency with Dynamic Alpha Steering**:
   * When the crop head is confident (e.g. Peach $\ge 0.85$), suppress impossible classes (Plum/Citrus).
   * When the crop head is ambiguous between related families (Wheat vs. Corn), allow soft multi-crop reasoning rather than brittle hard slicing.
3. **Teacher Distillation for Disease Stability**:
   * The 142 passing cases (64.55%) and 87.12% conditional accuracy must be locked down using temperature-scaled KL distillation from Model A to prevent representation drift.
