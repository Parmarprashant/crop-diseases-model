# AgriVision — Cotton 286-Class Research Expansion
## Final Scientific Report & Evaluation Walkthrough

> **RESEARCH EXPERIMENT OUTCOME: `COTTON_EXPANSION_FAILED`**  
> **PRODUCTION STATUS**: RELEASE FROZEN & IMMUTABLE (NO PRODUCTION FILES MODIFIED)  
> **DATE**: September 21, 2026  
> **TARGET EXPERIMENT**: EfficientNet-B5 + CBAM 281 $\rightarrow$ 286 Class Expansion  

---

## 1. Executive Summary

This experiment tested whether the historical `crop-diseases-model` EfficientNet-B5 + CBAM classifier could be expanded from 281 classes to 286 classes by adding 5 Cotton classes (`Cotton - Aphids`, `Cotton - Bacterial Blight`, `Cotton - Healthy`, `Cotton - Powdery Mildew`, `Cotton - Target Spot`) using the newly downloaded `Cotton_Leaves` dataset (~197 images).

Under the predefined, strict scientific acceptance criteria:
- **Cotton Sealed Test Performance**: Macro F1 reached **71.48%** (below the $\ge 85.0\%$ threshold).
- **Old 281-Class Regression**: The model suffered a **92.79% regression** on original classes (Preservation Rate: **7.21%**, far below the $\ge 96.0\%$ threshold).
- **Broadleaf Negative Specificity**: False Cotton Acceptance on non-cotton broadleaves reached **95.26%** (violating the $\le 5.0\%$ tolerance).
- **OOD Behavior**: Mean Maximum Softmax Probability on OOD images spiked from **0.1761** to **0.7219** (drift: $+0.5458$).

### Final Decision
$$\mathbf{FINAL\;DECISION:\;COTTON\_EXPANSION\_FAILED}$$

**The candidate checkpoint `weights/research/efficientnet_b5_cbam_cotton286_v1.pt` MUST NOT replace production Model A.**

---

## 2. Production Firewall Verification

Both production models were verified via SHA-256 before and after the experiment. Neither file was modified, overwritten, or redeployed:

| Model Component | Checkpoint Path | Expected SHA-256 | Verified SHA-256 | Firewall Status |
| :--- | :--- | :--- | :--- | :--- |
| **Production Model A** | `weights/efficientnet_b5_cbam_best.pt` | `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7` | `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7` | **IMMUTABLE / FROZEN** |
| **Production Campaign 2** | `Model/weights/crop_ood_expert_campaign2.pt` | `d71680b970be84f77eeb77b81118874ba2a2fd362790ed5e2d25b6e34a515f9b` | `d71680b970be84f77eeb77b81118874ba2a2fd362790ed5e2d25b6e34a515f9b` | **IMMUTABLE / FROZEN** |

---

## 3. Dataset Audit & Stratified Partitioning

### Audit Findings ([`dataset_audit.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/dataset_audit.json))
- **Total images**: 197 valid images (0 corrupt).
- **Exact duplicate files discovered**: 3 pairs
  - `Aphids edited/14.jpg` $\leftrightarrow$ `Aphids edited/27.jpg`
  - `Bacterial Blight edited/19.jpg` $\leftrightarrow$ `Bacterial Blight edited/31.jpg`
  - `Target spot edited/30.jpg` $\leftrightarrow$ `Target spot edited/7.jpg`

### Partitioning ([`cotton_split_manifest.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/cotton_split_manifest.json))
To prevent data leakage, duplicate clusters were forced into the same partition. The dataset was split deterministically (`seed=42`):

| Class Name | Total | Train (70%) | Dev (15%) | Locked Sealed Test (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Cotton - Aphids** | 39 | 26 | 7 | 6 |
| **Cotton - Bacterial Blight** | 40 | 27 | 6 | 7 |
| **Cotton - Healthy** | 39 | 27 | 6 | 6 |
| **Cotton - Powdery Mildew** | 38 | 26 | 6 | 6 |
| **Cotton - Target Spot** | 41 | 29 | 6 | 6 |
| **TOTAL** | **197** | **135** | **31** | **31** |

> **Statistical Limitation**: The sealed test set contains ~6 images per class (31 total). Binomial confidence intervals are wide ($\pm 12\%$), as expected given dataset scale.

---

## 4. Rehearsal Data Audit ([`rehearsal_manifest.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/rehearsal_manifest.json))

As mandated by the research protocol:
> *"DO NOT train the 286-class model using only the ~200 Cotton images. The training dataset must contain: Cotton training images PLUS representative rehearsal data from the original 281 classes."*

### Key Finding:
- An inspection of `training/dataset.py` and search paths across the local drive confirmed that the original **50-100 GB 281-class training corpus (`Data/outputs/outputs/train_split.csv`) is NOT present on this local machine**.
- The only local non-validation training data available was **50 images of Ginger** (`Model/data/curated_ginger`).
- Strict firewall rules prevented pulling from validation/test cohorts (`field_test_cohort`, `open_set_v2_cohort`, etc.).
- Consequently, only 1 of the 281 original classes had training rehearsal samples.

---

## 5. Class Expansion & Initialization ([`class_expansion_audit.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/class_expansion_audit.json))

- `classifier.4` was expanded from `Linear(512, 281)` to `Linear(512, 286)`.
- **Bitwise verification**:
  - `torch.equal(new_weight[:281], old_weight)` $\rightarrow$ **TRUE**
  - `torch.equal(new_bias[:281], old_bias)` $\rightarrow$ **TRUE**
- **New rows (281..285)**:
  - Initialized with deterministic Kaiming Normal (`seed=42`).
  - Checkpoint saved to `research/cotton286/init_286.pt` (SHA256: `e38abbeb48fbe903...`).

---

## 6. Training Dynamics & Multi-Seed Results

Training was executed with Phase 1 representation freeze (backbone frozen, CBAM and classifier trainable) using the combined objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new\_class}} + 1.0 \cdot \mathcal{L}_{\text{old\_class}} + 2.0 \cdot \mathcal{L}_{\text{distill}}$$

| Seed | Epochs | Final Loss | Cotton Train Acc | Rehearsal Acc | Best Dev Acc | Checkpoint Path |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 9.3083 | 50.4% | 4.0% | **51.61%** | `research/cotton286/cotton286_seed42.pt` |
| **43** | 10 | 9.5653 | 43.7% | 6.0% | **48.39%** | `research/cotton286/cotton286_seed43.pt` |

Seed 42 was selected as the research candidate and copied to `weights/research/efficientnet_b5_cbam_cotton286_v1.pt`.

---

## 7. Multi-Dimensional Benchmark Evaluation

### 7.1 Sealed Cotton Test Set ([`cotton_metrics.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/cotton_metrics.json))
- **Overall Accuracy**: 67.74% (21 / 31)
- **Balanced Accuracy**: 67.62%
- **Macro F1**: **71.48%** (Threshold: $\ge 85.0\%$ $\rightarrow$ **FAIL**)
- **Per-Class Metrics**:
  - `Cotton - Aphids`: Recall 83.33%, Precision 83.33%, F1 **83.33%**
  - `Cotton - Bacterial Blight`: Recall 57.14%, Precision 80.00%, F1 **66.67%**
  - `Cotton - Healthy`: Recall 66.67%, Precision 66.67%, F1 **66.67%**
  - `Cotton - Powdery Mildew`: Recall 50.00%, Precision 60.00%, F1 **54.55%**
  - `Cotton - Target Spot`: Recall 83.33%, Precision 83.33%, F1 **83.33%**

### 7.2 Old 281-Class Regression ([`old281_regression.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/old281_regression.json))
- **Samples Evaluated**: 111 samples across non-cotton crops (Guava, Eggplant, Papaya, Corn, Pumpkin, Chilli, Rose, Mango, Onion, Ginger).
- **Teacher-Student Agreement**: 8 / 111
- **Old Class Preservation Rate**: **7.21%** (Tolerance: $\ge 96.0\%$ $\rightarrow$ **SEVERE FAILURE**)
- **Regression Delta**: **92.79%**

### 7.3 Broadleaf Negative Specificity ([`cotton_negative_test.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/cotton_negative_test.json))
- **Samples Evaluated**: 211 negative broadleaf images (Field test, Sunflower, Rose, Ginger).
- **False Cotton Predictions**: 201 / 211
- **False Cotton Acceptance Rate**: **95.26%** (Tolerance: $\le 5.0\%$ $\rightarrow$ **SEVERE FAILURE**)
- **Behavioral Pathology**: The model collapsed into predicting `Cotton - Bacterial Blight` (Class 282), `Cotton - Powdery Mildew` (Class 284), or `Cotton - Target Spot` (Class 285) on almost every green broadleaf leaf.

### 7.4 OOD Regression ([`ood_regression.json`](file:///D:/1winbackup/desktop/Ganpat%20University/crop-diseases-model/research/cotton286/ood_regression.json))
- **Teacher MSP on OOD**: 0.1761 (appropriately uncertain)
- **Student MSP on OOD**: 0.7219 (over-confident hallucination)
- **MSP Drift**: $+0.5458$ (Tolerance: $< 0.05$ $\rightarrow$ **FAIL**)

---

## 8. Scientific Root Cause & Recommendations

### Why Did the Expansion Fail?
1. **Asymmetric Gradient Dynamics**:
   Because the full multi-gigabyte 281-class training dataset was absent from the local machine, only Cotton (135 images) and Ginger (50 images) received positive cross-entropy gradients.
2. **Logit Inflation**:
   The remaining 280 classes received zero gradient support. Even with teacher-student distillation on Ginger, the unconstrained Cotton logits grew substantially larger than the dormant historical logits.
3. **Over-Generalization**:
   The model effectively learned that "any green leaf = Cotton," destroying its discriminative capability across all 281 existing crops.

### Recommendations
1. **DO NOT DEPLOY**: The research candidate must remain isolated in `weights/research/` and never replace `weights/efficientnet_b5_cbam_best.pt`.
2. **USE THE 42-CLASS STACK FOR COTTON**:
   The existing 42-class model (`weights/backup_42class/efficientnet_b5_cbam_best.pt`) already contains **12 validated Cotton classes** (Aphids, Bacterial Blight, Healthy, Anthracnose, Bollrot, Bollworm, Mealybug, Whitefly, Pink Bollworm, Red Cotton Bug, Thrips, American Bollworm) trained jointly with Rice, Wheat, Maize, and Sugarcane. For Cotton diagnostics, route requests through the 42-class specialist model.
3. **FUTURE 286-CLASS PREREQUISITE**:
   Any future unified 286-class training must first restore the full 281-class training dataset so that balanced rehearsal can be performed across all classes simultaneously.
