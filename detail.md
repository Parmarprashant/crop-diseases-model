# AgriVision AI: Complete Architecture & Model Technical Specifications

## 1. Executive Summary

**AgriVision AI** is an industrial-grade, scientifically defensible agricultural diagnostic system designed for high-accuracy foliar disease classification, pest localization, lesion severity segmentation, and safe agronomic advisory. 

The core scientific premise of the system is:
$$\mathbf{Agreement \neq Validation}$$

Rather than treating deep learning models as unconstrained black-box predictors or blindly trusting LLM hallucinations, the platform enforces a **hierarchical, multi-stage safety pipeline**:
1. Image Quality Gating (Laplacian blur & illumination thresholds)
2. Calibrated Crop Family Identification (with first-class `UNKNOWN` state)
3. Plant Organ Detection & Compatibility (Organ-Crop domain matrix)
4. Energy-Based Out-of-Distribution (OOD) Calibration (AUROC 98.02%)
5. Dual-Model Independent Reasoning (Closed-set CNN $\leftrightarrow$ Gemini Vision)
6. Conservative 3-Tier Semantic Consensus Resolver (6 canonical resolution states)
7. Fail-Safe Agronomic Advisory Engine (Post-resolution chemical suppression)

---

## 2. Complete Model Parameters & Compute Footprint

The system operates a hybrid edge-cloud deep learning ensemble combining **three specialized local PyTorch vision models** running concurrently on GPU with a **cloud-based multimodal foundation model**.

| Model Component | Architecture / Backbone | Tasks / Outputs | Parameter Count | Checkpoint Size | Precision / Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Classifier** | EfficientNet-B5 + CBAM (Convolutional Block Attention Module) | 42 Disease/Healthy Foliar Classes + Spatial/Channel Attention Heatmap | **~28.51 Million** | `120.8 MB` (`efficientnet_b5_cbam_best.pt`) | FP32 / CUDA (RTX 5060) |
| **Lesion Segmenter** | ResNet-34 Encoder + U-Net Feature Decoder | Pixel-wise necrotic lesion mask, infected area %, pixel counts | **~24.43 Million** | `116.5 MB` (`unet_resnet34_best.pt`) | FP32 / CUDA (RTX 5060) |
| **Pest Detector** | Ultralytics YOLOv8n (Nano) | Bounding box localization for 14 agricultural pest classes | **~3.16 Million** | `6.55 MB` (`yolov8n.pt`) | FP32 / CUDA (RTX 5060) |
| **Multimodal Secondary Assistant** | Gemini 2.5 Flash / Gemini Vision | Independent un-biased visual pathology assessment, organ identification | Cloud Foundation Model (~Billions of params) | Zero local disk / VRAM | Cloud API (google-genai / HTTPS) |
| **TOTAL LOCAL STACK** | **Tri-Model PyTorch Ensemble** | **Classification + Attention + Segmentation + Pest Localization** | **~56.10 Million** | **~243.8 MB** | **~1.2 GB VRAM** |

### Computational Efficiency & Runtime Benchmarks
- **Inference Latency (Local Stack on RTX 5060 GPU)**: $\sim 85\text{ ms} - 120\text{ ms}$
- **Inference Latency (End-to-End Concurrent with Gemini Vision API)**: $\sim 1.1\text{ s} - 1.8\text{ s}$
- **GPU Memory Allocation**: $\sim 1,220\text{ MB}$ dedicated VRAM commit charge
- **Input Resolution**:
  - Classifier: $256 \times 256 \times 3$ (training/eval) / supports up to $456 \times 456$ native EfficientNet-B5
  - Segmenter: $256 \times 256 \times 3$
  - Pest Detector: $640 \times 640 \times 3$ (auto-scaled)

---

## 3. Training Dataset & Class Taxonomy (`MAIN DATA`)

The closed-set classifier was trained and evaluated strictly on the benchmark `MAIN DATA` corpus covering **42 distinct crop pathology classes across 5 staple crop families**.

### Crop Distribution Breakdown

```
MAIN DATA (42 Classes)
├── Cotton (7 Classes)
│   ├── Healthy cotton
│   ├── bacterial_blight in Cotton
│   ├── Anthracnose on Cotton
│   ├── bollrot on Cotton
│   ├── Leaf Curl
│   ├── American Bollworm on Cotton
│   └── Cotton Aphid
├── Rice (6 Classes)
│   ├── Healthy Rice (represented in clean leaf samples)
│   ├── Becterial Blight in Rice
│   ├── Brownspot
│   ├── Leaf smut
│   ├── Rice Blast
│   └── Tungro
├── Wheat (11 Classes)
│   ├── Healthy Wheat
│   ├── Wheat Brown leaf Rust
│   ├── Wheat black rust
│   ├── Wheat yellow rust
│   ├── Septoria
│   ├── Powdery Mildew
│   ├── Flag Smut
│   ├── Wheat Stem fly
│   ├── Wheat aphid
│   ├── Wheat mite
│   └── stripe rust
├── Maize (7 Classes)
│   ├── Healthy Maize
│   ├── Common_Rust
│   ├── Gray_Leaf_Spot
│   ├── maize ear rot
│   ├── maize stem borer
│   ├── maize fall armyworm
│   └── Army worm
└── Sugarcane (5 Classes)
    ├── Sugarcane Healthy
    ├── RedRot sugarcane
    ├── RedRust sugarcane
    ├── Mosaic sugarcane
    └── Yellow Leaf
```

### Training Convergence Metrics

The model was trained using AdamW optimization with Cosine Annealing Learning Rate scheduling and Cross-Entropy loss over 5 full epochs:

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Learning Rate | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.9209 | 53.95% | 0.6145 | 84.38% | $4.53 \times 10^{-5}$ | 777.1 s |
| 2 | 0.4225 | 87.83% | 0.3386 | 91.61% | $3.31 \times 10^{-5}$ | 557.4 s |
| 3 | 0.2270 | 93.25% | 0.3342 | 92.27% | $1.79 \times 10^{-5}$ | 537.5 s |
| 4 | 0.1655 | 94.99% | 0.3378 | 92.40% | $5.68 \times 10^{-6}$ | 549.9 s |
| **5 (Best)** | **0.1433** | **95.57%** | **0.3264** | **92.40%** | **$1.00 \times 10^{-6}$** | **565.2 s** |

---

## 4. Safety Gates & Defense-in-Depth Architecture

```
                       ORIGINAL IMAGE
                             │
            [Gate 1: Image Quality Evaluator]
            (Blur < 100, Illumination 35-225)
                             │ Pass
                             ├──────────────────────────┐
                             ↓                          ↓
                  [Local PyTorch Models]        [Gemini Vision API]
                   - EfficientNet-B5+CBAM       - Independent Assessment
                   - YOLOv8n Pest Detector      - Schema: Status, Organ,
                   - ResNet-34 U-Net              Diagnosis, Strength
                             │                          │
            [Gate 2: Calibrated Crop Detector]          │
            (Threshold >= 0.35, Margin >= 0.08)         │
                             │                          │
            [Gate 3: Organ & Domain Gating]             │
            (Organ Compatibility Matrix)                │
                             │                          │
            [Gate 4: Energy-Based OOD Gating]           │
            (Free Energy <= -5.327, Entropy <= 0.466)   │
                             │                          │
                             └──────────┬───────────────┘
                                        ↓
                       [3-Tier Conservative Matcher]
                                        ↓
                        [Central Consensus Resolver]
                        (6 Canonical Evidence States)
                                        ↓
                         [Post-Resolution Advisory]
                        (Chemical Sprays Suppressed
                         on Unconfirmed Conditions)
```

### Detailed Gate Specifications

#### Gate 1: Image Quality Evaluator
- **Laplacian Variance Blur Filter**: Computes the variance of the Laplacian kernel across the luminance channel:
  $$\text{Var}(\Delta I) = \frac{1}{HW} \sum (L(x, y) - \mu_L)^2 \ge 100.0$$
  Blurred images ($\text{score} < 100.0$) fail early, returning `Unreadable Image` without loading heavy inference.
- **Illumination Boundaries**: Mean intensity $\mu$ must satisfy $35 \le \mu \le 225$.
- **Contrast Check**: Standard deviation $\sigma \ge 25.0$.

#### Gate 2: Calibrated Crop Family Detector
- Replaces naive hard checks (`crop_prob == 1.0`) with calibrated probabilistic pooling:
  $$\text{Score}(\text{Crop}) = \sum_{c \in \text{CropClasses}} P(c \mid x) \ge \tau_{\text{crop}} \quad (\tau_{\text{crop}} = 0.35, \text{margin} \ge 0.08)$$
- **Scout Hint Validation**: If a farmer specifies a crop hint (e.g. `Rice`), but the visual predictive mass points to a different crop (e.g. `Maize`) with score $\ge 0.60$ and margin $\ge 0.25$, the hint is rejected and crop status is marked `UNKNOWN`.

#### Gate 3: Plant Part & Organ Compatibility Matrix
- Closed-set models are strictly limited to their training distribution (primarily foliage).
- If the organ is detected as a **Panicle**, **Earhead**, or **Stem**, the model consults `CLASS_TAXONOMY[class_name].supported_plant_parts`.
- For example, *Rice Blast* in the 42-class model only covers leaf symptoms; submitting a rice panicle triggers an automatic organ incompatibility rejection, preventing misdiagnoses such as *Maize Stem Borer*.

#### Gate 4: Energy-Based Out-of-Distribution (OOD) Calibration
Standard Softmax confidence (Maximum Softmax Probability, MSP) suffers from overconfidence on out-of-distribution inputs. AgriVision implements Helmholtz Free Energy scoring:
$$E(x; f) = -T \cdot \log \sum_{i=1}^K \exp(f_i(x) / T)$$
Combined with Normalized Shannon Entropy:
$$H(x) = -\frac{1}{\log K} \sum_{i=1}^K p_i \log p_i$$

**Empirical Calibration Performance (Held-Out Validation Set)**:
- **Locked Energy Acceptance Threshold**: $-5.327$
- **AUROC**: **98.02%**
- **AUPR**: **81.97%**
- **False Positive Rate at 95% TPR (FPR95)**: **5.42%**
- **Held-Out ID Retention (TPR)**: **96.0%**
- **Held-Out OOD Rejection**: **100.0%**

---

## 5. The 6-State Central Consensus Resolver

The resolution engine replaces simplistic "model trust" with an **evidence-state decision machine**:

```
                 ┌── CONSENSUS (CNN Accepted + Gemini Agrees)
                 │
CNN ACCEPTED ────┼── CONFLICT (CNN Accepted + Gemini Disagrees)
                 │
                 └── CNN_ONLY (CNN Accepted + Gemini Unavailable)

                 ┌── GEMINI_SUSPECTED (CNN Rejected + Gemini Valid)
CNN REJECTED ────┤
                 └── INSUFFICIENT_EVIDENCE (CNN Rejected + Gemini Unavailable/Uncertain)

UNKNOWN ────────── (Image Unreadable / Quality Failure / Ambiguous Evidence)
```

### The State Decision Table

| Resolution State | CNN State | Gemini State | Source Field | `validated_by_cnn` | Farmer Headline | Farmer Subheading | Chemical Treatments Allowed? |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **`CONSENSUS`** | Accepted | Same (Agrees) | `CONSENSUS` | `true` | `<Disease Name>` | *AI consensus: CNN + Gemini. Both models independently agree.* | **YES** |
| **`CONFLICT`** | Accepted | Different | `DISAGREEMENT` | `false` | `AI assessments disagree` | *Expert verification recommended* | **NO (Suppressed)** |
| **`GEMINI_SUSPECTED`** | Rejected | Valid (Any) | `GEMINI` | `false` | `Suspected: <Disease Name>` | *AI visual assessment: Gemini. Expert verification recommended.* | **NO (Suppressed)** |
| **`CNN_ONLY`** | Accepted | Unavailable | `CNN` | `true` | `<Disease Name>` | *Primary AI assessment: CNN. Gemini unavailable.* | **YES** |
| **`INSUFFICIENT_EVIDENCE`** | Rejected | Unavailable / Uncertain | `NONE` | `false` | `Unable to determine disease reliably` | *Please provide a clearer image showing the affected plant part.* | **NO (Suppressed)** |
| **`UNKNOWN`** | Bypassed | N/A | `NONE` | `false` | `Unreadable Image` | *Please capture a clean, sharply focused photograph in good daylight.* | **NO (Suppressed)** |

> [!CAUTION]
> **Agreement != Validation Invariant**:
> If the CNN predicts *Cotton Anthracnose (88%)* but was **rejected** by safety gates (e.g. panicle organ mismatch or OOD energy score), and Gemini independently predicts *Anthracnose*, the resolution **MUST REMAIN `GEMINI_SUSPECTED`** with `validated_by_cnn: false`. It cannot be promoted to `CONSENSUS` because the CNN failed domain gating.

---

## 6. Conservative 3-Tier Semantic Matcher

To prevent accidental "fuzzy matching" from turning distinct diseases into false consensus, matching is strictly constrained:

```
Tier 1: Exact Taxonomy Match
        (e.g., 'Anthracnose on Cotton' == 'Anthracnose on Cotton')
        ↓ No Match
Tier 2: Whitelisted Legitimate Alias Match
        (Strictly defined in APPROVED_TAXONOMY_ALIASES per crop)
        (e.g., 'bacterial_blight in Cotton' == 'Angular Leaf Spot')
        (e.g., 'Becterial Blight in Rice' == 'Bacterial Leaf Blight')
        ↓ No Match
Tier 3: Strict Otherwise -> NO MATCH
        ('leaf spot' vs 'brown spot' vs 'blight' vs 'anthracnose' are NEVER interchangeable)
```

---

## 7. Agronomic Advisory Safety & Post-Resolution Logic

The advisory engine executes strictly **after** the Central Consensus Resolver has determined the evidence state.

- **For `CONSENSUS` and `CNN_ONLY`**:
  Full integrated disease management is populated from the agricultural knowledge base, including specific chemical active ingredients (e.g. *Copper Oxychloride, Carbendazim, Streptocycline*), biological controls (e.g. *Trichoderma viride, Pseudomonas fluorescens*), and cultural precautions.
- **For `GEMINI_SUSPECTED`, `CONFLICT`, `UNKNOWN`, and `INSUFFICIENT_EVIDENCE`**:
  - `chemical_control = []` (**Strictly empty list**)
  - `organic_control = []` (**Strictly empty list**)
  - `cultural_practices`: Field scouting instructions, macro-photography recommendations, and agricultural extension officer verification guidance.

---

## 8. API Schema & JSON Response Contract

### Primary Inference Endpoint
`POST /api/v1/diagnose?crop_hint={crop}`

### Response Schema Structure
```json
{
  "crop": {
    "name": "cotton",
    "confidence": 0.92,
    "status": "VERIFIED"
  },
  "plant_part": {
    "name": "leaf",
    "confidence": 0.88,
    "status": "DETECTED"
  },
  "primary_model": {
    "status": "accepted",
    "prediction": "Cotton Anthracnose",
    "confidence": 0.912,
    "raw_top_prediction": "Anthracnose on Cotton",
    "raw_confidence": 0.912,
    "rejection_reason": null,
    "rejection_reasons": [],
    "accepted": true,
    "ood": false,
    "crop_compatible": true,
    "plant_part_compatible": true,
    "gate_status": "ACCEPTED",
    "energy_score": -8.452,
    "entropy": 0.082,
    "top_candidates": [
      { "class_name": "Anthracnose on Cotton", "confidence": 0.912 },
      { "class_name": "bacterial_blight in Cotton", "confidence": 0.045 },
      { "class_name": "Healthy cotton", "confidence": 0.012 }
    ]
  },
  "fallback": {
    "provider": "gemini_vision",
    "status": "invoked",
    "crop": "cotton",
    "plant_part": "leaf",
    "diagnosis": "Anthracnose on Cotton",
    "assessment_strength": "HIGH",
    "model_reported_confidence": 0.88,
    "reasoning": "Circular dark necrotic lesions with concentric rings visible on leaf blade.",
    "evidence": ["circular lesions", "concentric rings", "chlorotic halo"]
  },
  "diagnosis": {
    "name": "Anthracnose on Cotton",
    "type": "disease",
    "confidence": "high",
    "status": "CONSENSUS",
    "consensus_state": "CONSENSUS",
    "resolution": "CONSENSUS",
    "source": "CONSENSUS",
    "validated_by_cnn": true,
    "farmer_headline": "Anthracnose on Cotton",
    "farmer_subheading": "AI consensus: CNN + Gemini. Both models independently agree."
  },
  "comparison": {
    "crop_agreement": true,
    "plant_part_agreement": true,
    "disease_semantic_match": true,
    "overall_agreement": true,
    "match_level": "EXACT_TAXONOMY",
    "explanation": "Exact taxonomy match: 'Anthracnose on Cotton'",
    "canonical_class": "Anthracnose on Cotton"
  },
  "resolution": "CONSENSUS",
  "evidence": [
    "Crop status: Cotton (status: VERIFIED)",
    "Plant organ: Leaf (status: DETECTED)",
    "Resolution state: CONSENSUS (source: CONSENSUS)",
    "Primary CNN prediction ('Anthracnose on Cotton') accepted (91.2%)",
    "Fallback: gemini_vision (invoked)"
  ],
  "recommendation": "Both independent visual assessments agree. Follow standard integrated management below.",
  "requires_expert_verification": false,
  "advisory": {
    "urgency": "MODERATE - Action recommended within 48-72 hours",
    "disease_description": "Caused by Colletotrichum gossypii. Produces small, reddish-brown water-soaked spots on leaves and bolls.",
    "chemical_control": [
      "Spray Carbendazim 50% WP @ 1 g/L or Mancozeb 75% WP @ 2.5 g/L.",
      "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L for systemic suppression."
    ],
    "organic_control": [
      "Seed bio-priming with Trichoderma viride @ 10 g/kg seed.",
      "Foliar spray of fermented cow urine extract (5%) mixed with neem oil."
    ],
    "cultural_practices": [
      "Use certified acid-delinted disease-free seed stocks.",
      "Ensure wider plant spacing to allow adequate sunlight penetration.",
      "Avoid working in the field when foliage is wet with dew or rain."
    ],
    "expert_verification_note": "Routine monitoring advised."
  },
  "pests": { "pest_count": 0, "count": 0, "detections": [] },
  "segmentation": {
    "infected_area_pct": 14.2,
    "severity_category": "Moderate",
    "leaf_pixels": 45200,
    "lesion_pixels": 6418
  },
  "visualizations": {
    "original": "data:image/jpeg;base64,...",
    "attention_heatmap": "data:image/jpeg;base64,...",
    "pest_detections": "data:image/jpeg;base64,...",
    "lesion_segmentation": "data:image/jpeg;base64,..."
  }
}
```

---

## 9. Verification & Quality Assurance Suite

The system includes a dedicated unit and regression test suite in `tests/test_hardened_pipeline.py` verifying all 18 diagnostic criteria:

```bash
python tests/test_hardened_pipeline.py
```

```
test_01_crop_detector_unknown_state                    ... ok
test_02_strict_cross_crop_rejection                    ... ok
test_03_rice_panicle_plant_part_rejection              ... ok
test_04_user_rice_panicle_case_end_to_end              ... ok
test_05_image_quality_blur_rejection                  ... ok
test_06_synthetic_ood_noise_rejection                  ... ok
test_07_gemini_missing_key_truthfulness                ... ok
test_08_advisory_safety_chemical_suppression           ... ok
test_09_known_in_distribution_cotton_leaf              ... ok
test_10_known_in_distribution_rice_blast               ... ok
test_11_zero_crop_violations_guarantee                ... ok
test_12_exact_resolution_table_all_7_states            ... ok
test_13_structural_invariant_enforcement              ... ok
test_14_conservative_semantic_matcher                 ... ok
test_15_advisory_chemical_suppression_invariant        ... ok
test_16_gemini_confidence_is_not_calibrated_probability... ok
test_17_real_world_ood_dataset_evaluation              ... ok
test_18_calibration_and_held_out_evaluation           ... ok
----------------------------------------------------------------------
Ran 18 tests in 5.218s

OK
```

---

## 10. Summary Checklist for Agronomists & Evaluators

- [x] **No Fabricated Diagnoses**: Rice panicle inputs will *never* produce *Maize Stem Borer* or invent *Rice False Smut* unless validated.
- [x] **Calibrated Gating**: Multi-metric OOD energy score (AUROC 98.02%) quarantines non-crop images.
- [x] **Zero Model Bias in Gemini**: Secondary visual inspection runs independently with zero access to CNN predictions or preliminary logits.
- [x] **No Chemical Treatment Without Validation**: Chemical pesticides are suppressed for all non-consensus states.
- [x] **Separation of Farmer UI from Debug Telemetry**: Farmers see clear, actionable plain language; developers and researchers inspect raw logits, energy scores, and comparison matrices in the telemetry drawer.
- [x] **High Performance**: Native NVIDIA RTX GPU acceleration serving requests in under 150 ms locally.
