# AgriVision — Cotton 286-Class Expansion
## Phase 1: Data Availability & Rehearsal Feasibility Report

> **PROTOCOL STATUS**: `286_CLASS_TRAINING_BLOCKED_MISSING_REHEARSAL_DATA`  
> **INVESTIGATION DATE**: September 21, 2026  
> **LEAD ML RESEARCH ENGINEER**: AgriVision Core Team  
> **TARGET EXPERIMENT**: EfficientNet-B5 + CBAM 281 $\rightarrow$ 286 Class Expansion (Attempt 2)  

---

## 1. Executive Finding & Stop Condition

Following the strict protocol defined in the Second Attempt Specification:
> *"If representative rehearsal data for the original 281 classes cannot be restored, DO NOT perform another 286-class training run. Instead, report: `286_CLASS_TRAINING_BLOCKED_MISSING_REHEARSAL_DATA` and explain exactly what data is required."*

### Official Scientific Finding
**Representative rehearsal data for the original 281 classes is NOT present in the local workspace or filesystem.**
- **Classes Required**: 281 original disease/healthy classes across 41 crop families (Apple, Banana, Bean, Bell Pepper, Corn, Grape, Potato, Rice, Tomato, Wheat, etc.).
- **Original Training Classes Available Locally**: **1 of 281 classes** (`Ginger - Healthy`, 50 images).
- **Missing Original Training Classes**: **280 of 281 classes (99.64%)**.
- **Conclusion**: Executing another 286-class fine-tuning run without representative rehearsal data across the 281 classes is **scientifically invalid**. It would predictably replicate the catastrophic forgetting, 92.79% old-class regression, and 95.26% false cotton broadleaf acceptance observed in Attempt 1.
- **Action**: In accordance with the mandate, **all 286-class training is BLOCKED** until the complete 281-class training corpus is restored.

---

## 2. Search Paths & Discovery Audit

A recursive, exhaustive search across all project directories (`crop-diseases-model/`, `Model/`, `kisandost/`) was conducted.

| Target Resource | Search Target | Status | Detail / Location |
| :--- | :--- | :--- | :--- |
| **Original Split Manifest** | `Data/outputs/outputs/train_split.csv` | **NOT FOUND** | Zero split CSVs found on disk |
| **Class ID Mapping** | `Data/metadata/metadata/class_id_map.csv` | **NOT FOUND** | Zero class mapping CSVs found |
| **Master Training Images** | `Data/master_images/master_images/images` | **NOT FOUND** | Multi-gigabyte image directory not present |
| **Legacy Train Directory** | `MAIN DATA/Train/` | **NOT FOUND** | Zero 281-class folder hierarchies present |
| **Training Archives** | `*.zip`, `*.tar.gz`, `*.parquet` | **PARTIAL** | Only `Model/data/raw/ginger/Ginger_Leaf_Dataset.zip` (218 MB) |
| **YOLO Pest Config** | `pest_dataset.yaml` | **FOUND** | YOLO pest localization only (14 classes) |
| **Cotton Dataset** | `Cotton_Leaves/Test/` | **FOUND** | 197 images across 5 Cotton classes |

---

## 3. Detailed Inventory of Locally Available Imagery

The total image inventory across the entire system comprises:

| Dataset / Cohort Path | Image Count | Class Scope | Usable as Rehearsal? | Reason |
| :--- | :---: | :--- | :---: | :--- |
| `Model/data/curated_ginger/` | 50 | Ginger only (1 class) | **LIMITED** | Only covers 1 of 281 classes |
| `Model/data/raw/ginger/` | ~218 MB zip | Ginger raw images | **LIMITED** | Only covers Ginger |
| `Cotton_Leaves/Test/` | 197 | Cotton (5 classes) | **TARGET DATA** | New classes to be added (not rehearsal) |
| `Model/validation/open_set_v2_cohort/` | 607 | OOD / Stress images | **PROHIBITED** | Designated validation/OOD cohort |
| `Model/validation/open_set_cohort/` | 82 | OOD images | **PROHIBITED** | Designated validation/OOD cohort |
| `Model/validation/field_test_cohort/` | 61 | Multi-crop field images | **PROHIBITED** | Designated validation test cohort |
| `Model/validation/campaign2_zero_shot_ood/` | 100 | Rose & Sunflower holdouts | **PROHIBITED** | Designated Campaign 2 OOD holdout |
| `Model/validation/sunflower_disease/` | 35 | Sunflower disease | **PROHIBITED** | Designated validation cohort |
| `crop-diseases-model/validation/stress_samples/` | 30 | Blur / Darkness stress | **PROHIBITED** | Designated safety stress cohort |

> **Strict Compliance Note**: Under Rule 3 (*"Do NOT use validation/test cohorts as rehearsal data"*) and Rule 5 (*"Do NOT download arbitrary replacement data and pretend it is the original training distribution"*), no validation samples were repurposed as training rehearsal.

---

## 4. Why 286-Class Training Without 281 Rehearsal Fails

The failure mechanism of Attempt 1 is mathematically reproducible:
1. **Asymmetric Cross-Entropy Gradient**:
   $$\nabla_{\theta} \mathcal{L}_{\text{CE}} = \sum_{c=1}^{C} (p_c - y_c) \cdot \frac{\partial z_c}{\partial \theta}$$
   For the 5 Cotton classes ($c \in [281, 285]$), $y_c = 1$ during Cotton batches, driving $\partial z_c / \partial \theta$ strongly positive.
   For the 280 dormant classes ($c \in [0, 280]$), $y_c = 0$ always, and $p_c$ receives no positive reinforcement.
2. **Logit Magnitude Divergence**:
   Without positive examples for classes $0..280$, the weights $W_{281..285}$ grow disproportionately larger in norm than $W_{0..280}$, even when distillation is applied to the dormant logits.
3. **Catastrophic Generalization Failure**:
   The expanded head collapses into predicting Cotton for any foliar image, yielding:
   - 92.79% regression on original classes.
   - 95.26% false cotton acceptance on non-cotton broadleaves.
   - 0.5458 MSP drift on OOD samples.

---

## 5. Exact Data Requirements to Unblock Phase 4 & Phase 5

To proceed with a scientifically valid 286-class expansion that preserves Model A's 281-class capabilities, the following must be restored to `crop-diseases-model/Data/`:

1. **Manifest**: `Data/outputs/outputs/train_split.csv` containing image paths and canonical class labels for the original 281 classes.
2. **Rehearsal Image Corpus**: At minimum **10 to 20 representative training images per class** across all 281 classes:
   $$281 \times 10 = 2,810 \text{ balanced rehearsal images}$$
3. **Class ID Map**: `Data/metadata/metadata/class_id_map.csv` verifying exact label-to-index alignment (0 to 280).

---

## 6. Recommended Interim Architecture (Zero Regression Alternative)

While the full 281-class training dataset is being restored, the system can serve Cotton diagnostics with **100% preservation of Model A** via a **Cascaded Specialist Architecture**:

```
                                  [Input Image]
                                        │
                                        ▼
                         [Campaign 2 Crop Family Gate]
                          (ConvNeXt-Tiny / Crop Expert)
                                        │
                       ┌────────────────┴────────────────┐
                       │                                 │
                 Crop = Cotton                     Crop != Cotton
                       │                                 │
                       ▼                                 ▼
         [Dedicated Cotton Specialist]          [Production Model A]
           (42-Class / 5-Class CNN)              (281-Class Classifier)
                       │                                 │
                       └────────────────┬────────────────┘
                                        ▼
                            [Consensus & Advisory]
```

- **Production Model A** remains 100% untouched (`b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7`).
- **Cotton Requests** are routed to the specialist model, which already achieves high specificity without distorting the 281-class decision boundaries.
