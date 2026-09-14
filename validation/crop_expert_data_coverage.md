# AgriVision AI — Crop Expert Data Coverage & Taxonomy Reconciliation Report

## 1. Crop Taxonomy Reconciliation (Single Source of Truth)

### Resolution of Documentation Discrepancy
* **Identified Discrepancy**: Earlier documentation and report texts referenced "32 canonical botanical crop families", whereas the underlying codebase (`weights/crop_names.txt`, `models/hierarchical_cbam.py`, `train_split.csv`, `val_split.csv`) has consistently contained **41 canonical crops**.
* **Root Cause**: The string "32" originated as a documentation transcription typo from the default DataLoader training batch size (`batch_size = 32`).
* **Single Source of Truth**: The active, canonical crop list is derived strictly from [`weights/crop_names.txt`](file:///d:/1winbackup/desktop/Ganpat%20University/Model/weights/crop_names.txt), which specifies **$N_{\text{crop}} = 41$** botanical families:
  `apple, banana, basil, bean, bell pepper, blackgram, blueberry, broccoli, cabbage, carrot, cauliflower, celery, cherry, chilli, citrus, coffee, corn, cucumber, eggplant, garlic, ginger, grape, grapevine, groundnut, lettuce, maple, paddy, peach, plum, potato, radish, raspberry, rice, soybean, squash, strawberry, sugarcane, tobacco, tomato, wheat, zucchini`
* **Split Verification**:
  - `train_split.csv`: Exactly 41 unique crops (0 missing, 0 extraneous).
  - `val_split.csv`: Exactly 41 unique crops (0 missing, 0 extraneous).
  - Pre-training check status: **STRICT ASSERTION PASSED**.

---

## 2. Dataset Distribution & Image Availability Audit

* **Total Samples in Training Set**: 53,360 samples
* **Total Samples in Validation Set**: 6,670 samples
* **Master Images Available**: 66,953 physical image files in `Data/master_images/master_images/images`
* **Physical File Availability**:
  - Training set: **53,360 / 53,360 (100.00%)**
  - Validation set: **6,670 / 6,670 (100.00%)**
  - Data integrity status: **100.0% VERIFIED**.

---

## 3. Laboratory vs. Real-World Field Distribution

| Dataset Source | Image Count | % of Training Set | Domain Characteristic |
|---|---|---|---|
| **PlantVillageDataset/PlantVillage** | 16,493 | 30.91% | Laboratory (detached leaves on neutral/grey background) |
| **Multicrop (train/valid/test)** | 17,356 | 32.53% | Real-world field imagery (India: Banana, Groundnut, Radish, Chilli) |
| **plantseg** | 8,174 | 15.32% | Field vegetation with segmentation ground-truth |
| **paddy-disease-classification** | 8,161 | 15.29% | In-field canopy photography (Paddy / Rice) |
| **PlantDoc-Dataset (train/test)** | 2,001 | 3.75% | High-complexity in-the-wild agricultural photography |
| **blackgram/BPLD** | 806 | 1.51% | Field pulse photography |
| **Sugarcane Leaf Disease Dataset** | 369 | 0.69% | Field monocot crop photography |

* **Laboratory Backgrounds**: 16,493 (30.9%)
* **Real-World Field Backgrounds**: 36,867 (69.1%)

---

## 4. Extreme Class Imbalance Analysis & Sampling Strategy

The dataset exhibits an acute **280:1 class imbalance ratio**:
* **Top 5 Majority Crops**: Tomato (14,003), Paddy (8,161), Banana (7,566), Groundnut (6,753), Bell Pepper (2,237).
* **Bottom 5 Minority Crops**: Celery (50), Basil (50), Grapevine (57), Ginger (66), Broccoli (68).
* **Median Crop Sample Count**: 230.0 samples.

### Mitigation in Crop Expert DataLoader
Without balanced sampling, standard empirical risk minimization collapses crop prediction towards the top 4 crops (explaining why the previous hierarchical candidate overpredicted Banana, Tomato, and Citrus). The Crop Expert DataLoader must use **Inverse Square-Root Frequency Sampling**:
$$w_c = \frac{1}{\sqrt{N_c}}$$
combined with hard-negative oversampling for visually confusing crop pairs.
