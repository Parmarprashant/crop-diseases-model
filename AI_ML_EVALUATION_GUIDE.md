# AgriVision AI — Round 1 Evaluation Master Guide (AI/ML Developer)

> **Role**: AI / ML Developer & Computer Vision Engineer  
> **Project**: AgriVision AI — Hierarchical Multi-Model Agricultural Diagnostic System  
> **Target Evaluation**: Round 1 Technical Viva & Jury Evaluation  
> **Stack**: PyTorch (CUDA), EfficientNet-B5 + CBAM, U-Net (ResNet-34), YOLOv8n, Google Gemini 2.5 Flash, FastAPI, Docker, Cloudflare

---

## 📑 Table of Contents
1. [30-Second Elevator Pitch](#1-30-second-elevator-pitch)
2. [High-Level Architecture & Inference Pipeline](#2-high-level-architecture--inference-pipeline)
3. [Deep Dive: The 3 Core Machine Learning Models](#3-deep-dive-the-3-core-machine-learning-models)
4. [The 4 Defensive Safety Gates](#4-the-4-defensive-safety-gates)
5. [The Dual-Model Consensus Resolver (CNN vs. Gemini)](#5-the-dual-model-consensus-resolver-cnn-vs-gemini)
6. [Agronomic Advisory Engine & Safety Principles](#6-agronomic-advisory-engine--safety-principles)
7. [Hardware Telemetry, Calibration & Deployment](#7-hardware-telemetry-calibration--deployment)
8. [Top 20 Toughest Viva / Jury Questions & Model Answers](#8-top-20-toughest-viva--jury-questions--model-answers)

---

## 1. 30-Second Elevator Pitch
> *"I built a hierarchical, production-grade agricultural diagnostic intelligence engine. Traditional crop disease classifiers fail in production because standard CNNs suffer from overconfident misclassifications on out-of-distribution images, unreadable photos, and mismatched crops.*  
> *To solve this, I engineered a **three-tier vision stack** (EfficientNet-B5 with CBAM Attention for 42-class disease diagnosis, ResNet-34 U-Net for pixel-level lesion severity calculation, and YOLOv8 for pest localization) protected by **4 defensive safety gates** (Image Quality, Crop ID, Plant Organ Compatibility, and Energy-based OOD detection). Finally, a **scientifically grounded Consensus Resolver** pairs the CNN with Google Gemini Vision, ensuring no farmer receives aggressive chemical recommendations without algorithmic consensus or expert verification."*

---

## 2. High-Level Architecture & Inference Pipeline

```
                     [ Farmer Uploads Image ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 1. Image Quality Gate │  → (Blur, Darkness, Contrast check)
                     └───────────┬───────────┘
                                 │ Passed
                                 ▼
              ┌─────────────────────────────────────┐
              │ 2. Crop & Plant-Part Detector       │  → (Crop Taxonomy + Organ check)
              └──────────────────┬──────────────────┘
                                 │ Compatible
                                 ▼
              ┌─────────────────────────────────────┐
              │ 3. Parallel Multi-Model Inference   │
              ├──────────────────┬──────────────────┤
              │                  │                  │
              ▼                  ▼                  ▼
    ┌──────────────────┐ ┌───────────────┐ ┌─────────────────┐
    │ EfficientNet-B5  │ │ ResNet-34     │ │ YOLOv8n         │
    │ + CBAM Attention │ │ U-Net         │ │ Pest Detector   │
    │ (42 Diseases)    │ │ (Lesion Area) │ │ (14 Pests)      │
    └─────────┬────────┘ └───────┬───────┘ └────────┬────────┘
              │                  │                  │
              ▼                  │                  │
    ┌──────────────────┐         │                  │
    │ 4. Energy OOD    │         │                  │
    │ Rejection Gate   │         │                  │
    └─────────┬────────┘         │                  │
              │                  │                  │
              ▼                  ▼                  ▼
   ┌────────────────────────────────────────────────────────┐
   │ 5. Dual-Model Consensus Resolver (CNN + Gemini Vision) │
   └────────────────────────────┬───────────────────────────┘
                                │
                                ▼
   ┌────────────────────────────────────────────────────────┐
   │ 6. Standardized Diagnostic Response + Advisory Engine  │
   │    - Disease Name & Calibrated Confidence              │
   │    - Infected Area % & Severity (Mild/Moderate/Severe) │
   │    - Pest Bounding Boxes, Labels & Counts              │
   │    - Grad-CAM / CBAM Heatmap & Lesion Mask             │
   │    - Safe Chemical, Organic & Cultural Recommendations │
   └────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive: The 3 Core Machine Learning Models

### Model 1: Primary Disease Classifier — EfficientNet-B5 + CBAM Attention
- **Purpose**: Multi-class fine-grained disease classification across **42 agricultural classes** (Tomato, Potato, Corn, Rice, Wheat, Grape, Apple, Pepper, etc.).
- **Parameters**: **~30.4 Million parameters**.
- **Input Resolution**: $256 \times 256 \times 3$ RGB.
- **Why EfficientNet-B5?**:
  - Employs **compound scaling** ($\alpha \cdot \text{depth}, \beta \cdot \text{width}, \gamma \cdot \text{resolution}$) to optimize receptive field without exponential FLOPS increase.
  - Significantly outperforms ResNet-50 on fine-grained foliar patterns (interveinal chlorosis, concentric rings, pustules).
- **Why CBAM (Convolutional Block Attention Module)?**:
  - Standard CNNs treat all channels and spatial regions uniformly. CBAM adds two sequential attention mechanisms:
    1. **Channel Attention**: *"What"* features are important. Uses both **Global Average Pooling** and **Global Max Pooling** fed into a shared Multi-Layer Perceptron (MLP) with reduction ratio $r=16$:
       $$M_c(F) = \sigma\Big(\text{MLP}\big(\text{AvgPool}(F)\big) + \text{MLP}\big(\text{MaxPool}(F)\big)\Big)$$
    2. **Spatial Attention**: *"Where"* the lesion is located on the leaf. Concatenates channel-wise Average Pool and Max Pool, followed by a $7 \times 7$ convolution:
       $$M_s(F') = \sigma\Big(f^{7\times7}\big([\text{AvgPool}(F'); \text{MaxPool}(F')]\big)\Big)$$
  - **Explainability**: Enables extracting interpretable **attention heatmaps** showing exactly which leaf lesions triggered the prediction.
- **Loss Function**: Cross-Entropy with **Label Smoothing ($0.1$)** to prevent overconfident softmax probabilities.

---

### Model 2: Lesion Severity Segmenter — U-Net with ResNet-34 Encoder
- **Purpose**: Pixel-level segmentation of necrosis, chlorosis, and lesions to quantify exact disease severity.
- **Parameters**: **~24.4 Million parameters**.
- **Input Resolution**: $256 \times 256 \times 3$ RGB $\rightarrow$ Output: $256 \times 256 \times 1$ binary mask.
- **Why U-Net with ResNet-34?**:
  - Encoder-decoder architecture with **skip connections** that transfer high-resolution spatial feature maps directly from contracting path to expanding path.
  - Preserves crisp lesion borders and micro-punctures that global pooling layers discard.
- **Metric Formulation**:
  $$\text{Infected Surface Area \%} = \frac{\sum \text{Lesion Pixels}}{\sum \text{Leaf Pixels}} \times 100\%$$
- **Severity Categorization**:
  - **Mild**: $< 15\%$ surface area infected.
  - **Moderate**: $15\% - 40\%$ surface area infected.
  - **Severe**: $> 40\%$ surface area infected (triggers systemic fungicide warnings).
- **Training Strategy**: Trained using a composite **Dice Loss + Binary Cross-Entropy (BCE)** loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \Big(1 - \frac{2 |Y \cap \hat{Y}| + \epsilon}{|Y| + |\hat{Y}| + \epsilon}\Big)$$

---

### Model 3: Pest Detection & Localization — YOLOv8n
- **Purpose**: Real-time identification and bounding-box localization of **14 agricultural pest classes** (aphids, armyworms, whiteflies, mites, thrips, beetles, etc.).
- **Parameters**: **~3.2 Million parameters** (Ultra-lightweight).
- **Why YOLOv8n?**:
  - **Anchor-free architecture**: Directly predicts bounding-box centers and dimensions, handling variable insect aspect ratios much better than anchor-based detectors.
  - **Decoupled Head**: Separates classification loss (VFL - Varifocal Loss) from bounding-box regression loss (CIoU + DFL - Distribution Focal Loss).
  - **Latency**: Sub-$15\text{ms}$ inference time, allowing simultaneous execution alongside EfficientNet-B5.

---

## 4. The 4 Defensive Safety Gates

To prevent disastrous real-world failures (e.g. diagnosing a photo of a shoe as "Late Blight"), our pipeline enforces **4 sequential safety gates** before predictions are accepted:

| Gate | Mechanism | Threshold / Rule | Failure Action |
| :--- | :--- | :--- | :--- |
| **1. Image Quality Gate** | Laplacian Variance (Blur) + Michelson Contrast | $\text{Var}(\Delta I) \ge 80.0$, Contrast $\ge 0.15$ | Rejects unreadable, blurry, or pitch-black photos. Bypasses CNN. |
| **2. Crop Identification** | Hierarchical Crop Taxonomy matcher | Must match one of the 42 validated crops | If crop is unknown or incompatible, CNN prediction rejected. |
| **3. Plant Organ Compatibility** | Plant organ classifier (leaf vs. stem vs. fruit) | e.g. Foliar diseases rejected if image is only bark/soil | Prevents applying foliar fungicide to root rot symptoms. |
| **4. Energy-based OOD Gating** | Free Energy Score across output logits | $E(x; T) \ge -5.327$ ($95\%$ TPR threshold) | Rejects Out-of-Distribution images (weeds, non-crops, random objects). |

### 🔬 Why Energy Score instead of Softmax Maximum Probability (MSP)?
Standard classifiers use Softmax:
$$P(y = c | x) = \frac{e^{z_c / T}}{\sum_{j} e^{z_j / T}}$$
**The Problem**: Softmax forces all probabilities to sum to $1.0$. On an out-of-distribution image (e.g., an apple tree photo fed into a corn model), the network still assigns high confidence to the least-wrong class due to the exponential normalization.

**The Solution — Free Energy**:
$$E(x; T) = -T \cdot \log \sum_{j=1}^{K} e^{z_j / T}$$
Energy is theoretically aligned with the input data density $p(x) \propto \sum e^{z_j}$. If an image is Out-of-Distribution, its unnormalized logits $z_j$ are all uniformly low, resulting in a high energy value that cleanly triggers rejection regardless of softmax distortion!

---

## 5. The Dual-Model Consensus Resolver (CNN vs. Gemini)

We implement a **scientifically defensible dual-model resolution policy**:

```
        Image
          │
    ┌─────┴────────────────┐
    ▼                      ▼
CNN + CBAM           Gemini 2.5 Flash
Prediction           Independent Visual Assessment
+ Safety Gates       (Zero knowledge of CNN logits)
    │                      │
    └──────────┬───────────┘
               ▼
       Consensus Resolver
```

### The 6-State Resolution Truth Table

| CNN State | Gemini State | Resulting State | Resolution Action | Chemical Advisory Permitted? |
| :--- | :--- | :--- | :--- | :---: |
| **Accepted** | **Same Diagnosis** | `CONSENSUS` | Definitive diagnosis confirmed by both models. | **YES** |
| **Accepted** | **Different** | `CONFLICT` | Models disagree. Urges farmer to consult agricultural extension officer. | **NO (Safety Lock)** |
| **Rejected (OOD/Low Conf)** | **Identifies Disease** | `GEMINI_SUSPECTED` | CNN was uncertain/rejected, but Gemini identified a condition. Flagged as suspected. | **NO (Scouting only)** |
| **Rejected** | **Uncertain / Insufficient**| `INSUFFICIENT_EVIDENCE` | Neither model could confirm disease. Prompts farmer to take clearer in-field photo. | **NO** |
| **Accepted** | **Unavailable (Offline)** | `CNN_ONLY` | Gemini API offline or key missing. CNN accepted through all 4 safety gates. | **YES** |
| **Rejected** | **Unavailable (Offline)** | `INSUFFICIENT_EVIDENCE` | CNN failed gates and Gemini unavailable. Safe failure state. | **NO** |

> **Critical Design Principle**:  
> To protect farmers from expensive and environmentally damaging chemical misapplications, **disease-specific chemical pesticides are ONLY unlocked in `CONSENSUS` or `CNN_ONLY` states**. All other states lock recommendations to organic remedies, cultural practices, and physical expert verification.

---

## 6. Agronomic Advisory Engine & Safety Principles

- **Three-Pillared Advisory Schema**:
  1. **Chemical Controls**: Recommended active ingredients (e.g., Mancozeb, Azoxystrobin), dosage rates, and pre-harvest intervals (PHI).
  2. **Organic / Biological Controls**: Trichoderma viride, neem oil formulations, copper hydroxide, and bio-fungicides.
  3. **Cultural Practices**: Crop rotation, drip irrigation adjustments (avoiding overhead wetting), pruning infected leaves, and field sanitation.
- **Defensive Safety Lock**: Automatically activated whenever severity is high ($>40\%$) or when models are in conflict, enforcing quarantine and agricultural officer inspection before spraying.

---

## 7. Hardware Telemetry, Calibration & Deployment

- **Inference Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) with CUDA 12.x and PyTorch FP16 mixed precision.
- **Inference Latency**:
  - EfficientNet-B5 + CBAM: **~48ms**
  - U-Net ResNet-34: **~62ms**
  - YOLOv8n: **~12ms**
  - Total Pipeline (GPU): **~145ms** (Sub-second response!)
- **ASGI Cloud Microservice**: FastAPI server exposing OpenAPI/Swagger endpoints on port `8000` (`/api/v1/diagnose`, `/api/v1/health`, `/api/v1/config`, `/docs`).
- **Global Deployment**: Exposing the local GPU server to the public internet via a **Cloudflare Public HTTPS Tunnel**, giving worldwide REST API access with full SSL encryption and DDoS protection.
- **Containerization**: Production `Dockerfile` with multi-stage build, non-root security UID 1000, and dynamic `$PORT` handling.

---

## 8. Top 20 Toughest Viva / Jury Questions & Model Answers

### Q1: Why did you choose EfficientNet-B5 instead of a standard ResNet-50 or a Vision Transformer (ViT)?
> **Answer**:  
> *"ResNet-50 scales only depth, leading to diminishing returns and vanishing gradients for subtle agricultural textures. Vision Transformers (ViTs) require massive pre-training datasets (hundreds of millions of images like JFT-300M) and lack inductive bias for translation invariance, making them prone to overfitting on foliar datasets.  
> EfficientNet-B5 uses principled compound scaling—balancing depth, width, and image resolution simultaneously. It achieves state-of-the-art Top-1 accuracy with only ~30.4M parameters, fitting comfortably into our target deployment latency budget while providing optimal receptive field for microscopic fungal pustules."*

---

### Q2: Explain the exact mechanism of CBAM and why it is superior to Squeeze-and-Excitation (SE-Net).
> **Answer**:  
> *"SE-Net only applies Channel Attention using Average Pooling. However, leaf disease lesions are localized anomalies: a small lesion with high pixel intensity might be completely diluted by average pooling across the entire leaf.  
> CBAM improves this in two ways:  
> 1. In the **Channel Attention Module**, it computes BOTH Average Pooling AND Max Pooling. Max pooling preserves the strongest localized anomaly signals.  
> 2. It introduces a sequential **Spatial Attention Module** that pools across channels and applies a $7 \times 7$ spatial convolution. This tells the network where to look on the leaf surface, which SE-Net cannot do. This spatial attention map also gives us direct model explainability for free."*

---

### Q3: Why is Softmax Maximum Predicted Probability (MSP) dangerous for Out-of-Distribution (OOD) detection?
> **Answer**:  
> *"Softmax contains an inherent normalization constraint: $\sum_{i=1}^K P_i = 1$. When an out-of-distribution image (such as an animal or an uncultivated weed) is passed to the network, all logit values $z_i$ may be small or negative. However, dividing by the sum forces the probabilities to stretch to 1.0, frequently assigning $85\%+$ confidence to an arbitrary class.  
> We solved this by using **Energy-based OOD detection**. The free energy $E(x; T) = -T \cdot \log \sum e^{z_i/T}$ maps directly to the Helmholtz free energy and tracks the unnormalized logit density. When an image is OOD, all logits are low, leading to a high energy score that safely rejects the sample."*

---

### Q4: How did you train the U-Net without having thousands of hand-annotated polygon masks?
> **Answer**:  
> *"We utilized a **semi-supervised pseudo-mask generation pipeline**. In foliar disease imagery, healthy plant tissue and diseased necrotic tissue exhibit distinct color-space clustering in LAB and HSV color spaces ($a^*$ channel represents green-to-magenta, and $b^*$ represents blue-to-yellow).  
> Using an automated GrabCut and Otsu adaptive thresholding algorithm conditioned on the disease taxonomy, we generated high-quality binary pseudo-ground-truth masks for the leaf area and lesion regions. We then trained the ResNet-34 U-Net using a combined Dice + BCE loss with heavy geometric augmentations (random elastic transforms, affine rotations, and color jitter) to make the segmenter robust to lighting and camera variations."*

---

### Q5: What is the risk of using Google Gemini directly, and why do you call it a 'Defensive Dual-Model' architecture?
> **Answer**:  
> *"Large Vision-Language Models like Gemini are open-world models with vast botanical knowledge, but they have two critical weaknesses in production:  
> 1. **Hallucination and stochasticity**: They can produce plausible-sounding but factually incorrect diagnoses on ambiguous images.  
> 2. **Uncalibrated confidence**: An LLM returning 'confidence: 0.90' is expressing linguistic certainty, not a mathematically calibrated Bayesian probability.  
> Therefore, our architecture treats Gemini as an independent second-opinion assessor. Gemini is never exposed to CNN logits or candidate probabilities, ensuring zero confirmation bias. And most importantly, our Consensus Resolver requires both models to agree before high-stakes chemical pesticides can be recommended."*

---

### Q6: What happens if an image of an apple leaf with Scab is uploaded, but the farmer selects 'Tomato' as the crop hint?
> **Answer**:  
> *"Our **Gate 2 & 3 (Domain Compatibility Gating)** detects this contradiction immediately. The taxonomy engine knows that Apple Scab (*Venturia inaequalis*) is taxonomically incompatible with the Solanaceae (Tomato) family.  
> The system flags a compatibility mismatch, suppresses the closed-set prediction, and invokes the secondary Gemini assessor to clarify whether the image is actually an Apple leaf or a Tomato leaf suffering from Early Blight, preventing cross-crop diagnostic errors."*

---

### Q7: Why did you choose a $256 \times 256$ resolution instead of the default $456 \times 456$ for EfficientNet-B5?
> **Answer**:  
> *"We benchmarked the trade-off between inference throughput, VRAM utilization, and classification accuracy. At $456 \times 456$, VRAM consumption during batch inference increases by over $3.1\times$, and latency jumps from ~48ms to ~140ms.  
> At $256 \times 256$, with CBAM attention enabled, the network captures both macroscopic leaf geometry and localized lesion margins with less than $0.6\%$ difference in Top-1 validation accuracy, while allowing the entire 3-model stack (Classifier + Segmenter + YOLOv8) to run concurrently on an 8GB GPU in under $150\text{ms}$ total pipeline time."*

---

### Q8: How do you handle class imbalance in the 42-disease training dataset?
> **Answer**:  
> *"Agricultural datasets suffer from severe class imbalance: common diseases like Early Blight have thousands of samples, while rare viral infections may only have 100 images. We addressed this through a three-pronged strategy:  
> 1. **Class-Aware Weighted Cross-Entropy**: Penalizing errors on under-represented classes inversely proportional to class frequency: $w_c = \frac{N}{K \cdot N_c}$.  
> 2. **Targeted Data Augmentation**: Applying MixUp ($\alpha=0.2$) and CutMix exclusively to minority classes to enforce linear interpolation between feature manifolds.  
> 3. **Focal Loss Formulation**: $\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with $\gamma=2.0$, which down-weights easy examples and forces gradients to focus on hard, rare disease cases."*

---

### Q9: How do you evaluate the reliability of your Energy-based OOD detector?
> **Answer**:  
> *"We evaluate OOD performance using standard rigorous metrics:  
> 1. **AUROC (Area Under the Receiver Operating Characteristic)**: Measures detection capability across all possible thresholds regardless of class priors (our model achieves **$98.02\%$ AUROC**).  
> 2. **FPR at 95% TPR (False Positive Rate at 95% True Positive Rate)**: Measures the percentage of OOD samples mistakenly accepted when 95% of real crop diseases are correctly admitted (our model achieves a low **$5.42\%$ FPR95**).  
> We calibrated these thresholds against both near-OOD samples (healthy leaves of unsupported crops) and far-OOD samples (ImageNet non-plant categories)."*

---

### Q10: How does your system quantify lesion severity, and how is it used in the advisory?
> **Answer**:  
> *"Rather than relying on vague visual guesses, our U-Net model outputs a binary mask where pixel values indicate lesion presence. We compute:  
> $$\text{Severity Ratio} = \frac{\text{Count}(\text{Lesion Pixels})}{\text{Count}(\text{Leaf Tissue Pixels})}$$  
> This numerical percentage directly drives the advisory engine:  
> - If severity is $<15\%$ (Mild), the advisory suggests cultural sanitation and biological controls (e.g. neem extract or copper soap).  
> - If severity exceeds $40\%$ (Severe), the advisory escalates urgency, recommending curative systemic treatments (e.g. Triazoles or Strobilurins) combined with strict pre-harvest interval precautions."*

---

### Q11: What is the role of YOLOv8n when EfficientNet-B5 is already classifying diseases?
> **Answer**:  
> *"Crop damage is often caused by insect pests rather than fungal or bacterial pathogens, and frequently pests act as disease vectors (e.g. Whiteflies transmitting Tomato Yellow Leaf Curl Virus).  
> EfficientNet-B5 performs image-level classification, meaning it cannot count individual insects or pinpoint multiple isolated pests across a canopy. YOLOv8n performs real-time object detection: it outputs bounding boxes, classification tags, and exact insect counts. This allows our advisory engine to deliver dual-action recommendations (e.g., pairing a fungicide for powdery mildew with an insecticide if aphid clusters are detected)."*

---

### Q12: Why did you deploy using a Cloudflare Tunnel instead of a standard port forwarding or basic local server?
> **Answer**:  
> *"Direct port forwarding requires modifying router NAT tables, exposing static public IP addresses, and leaves the host vulnerable to port scanning and DDoS attacks.  
> A Cloudflare Tunnel creates an encrypted, outbound-only QUIC/HTTP2 tunnel from the host to Cloudflare's global edge network. No inbound ports need to be opened on the host router. Cloudflare terminates TLS at the edge, provides automatic DDoS protection, and gives our local GPU machine a secure, globally reachable HTTPS endpoint that any mobile app or external web frontend can communicate with."*

---

### Q13: What happens if a farmer uploads a completely black, white, or out-of-focus photo?
> **Answer**:  
> *"The request is stopped at **Gate 1 (Image Quality Gate)** in under $5\text{ms}$ before any deep learning model is invoked.  
> We calculate the variance of the Laplacian operator $\text{Var}(\nabla^2 I)$. Blurry images have low edge variance ($< 80$). We also calculate the Michelson contrast $\frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$. If contrast is $< 0.15$ or average brightness is $< 20$, the pipeline halts early and returns an explicit error prompt instructing the farmer to retake the photo with better lighting and focus."*

---

### Q14: How are base64 visualization heatmaps generated and returned by the API?
> **Answer**:  
> *"For explainability, our API returns 4 visual artifacts encoded in Base64:  
> 1. **Original Image**: The normalized crop.  
> 2. **CBAM Attention Heatmap**: Generated by extracting the spatial attention tensor $M_s$ from the final stage of EfficientNet-B5, resizing it to the original dimensions with bilinear interpolation, and applying a Jet color map overlaid on the grayscale image.  
> 3. **YOLO Detection Overlay**: The image with rendered bounding boxes, labels, and confidence tags.  
> 4. **U-Net Lesion Mask**: A color-coded overlay (red for necrotic lesions, green for healthy leaf tissue).  
> These base64 strings can be directly rendered in any frontend using standard `<img src='data:image/jpeg;base64,...' />` tags."*

---

### Q15: How do you prevent Gemini Vision from biasing the primary CNN model?
> **Answer**:  
> *"We enforce a strict **zero-leakage boundary**. In `core/gemini_fallback.py`, the payload sent to the Gemini API contains only the raw image bytes and structured botanical examination instructions.  
> Gemini is NEVER sent the CNN's predicted class, logits, candidate probabilities, or model status. Both models perform inference completely independently. Only after both models have produced their independent conclusions does the deterministic `ConsensusResolver` evaluate them against the truth table."*

---

### Q16: What is the total parameter count and memory footprint of your inference server?
> **Answer**:  
> - **EfficientNet-B5 + CBAM**: ~30.4M parameters (~121 MB file size)  
> - **U-Net ResNet-34**: ~24.4M parameters (~116 MB file size)  
> - **YOLOv8n**: ~3.2M parameters (~6.5 MB file size)  
> - **Total Neural Parameters**: **~58.0 Million parameters** (~244 MB total disk footprint)  
> - **In-Memory GPU Footprint**: Under **2.8 GB VRAM** in PyTorch FP16 evaluation mode, allowing lightning-fast sub-$150\text{ms}$ inference on an 8GB NVIDIA RTX 5060 laptop GPU.

---

### Q17: What is the exact API contract of your backend?
> **Answer**:  
> *"Our backend is an asynchronous FastAPI service adhering strictly to OpenAPI 3.0:  
> - `POST /api/v1/diagnose`: Accepts `multipart/form-data` with an image file and optional `crop_hint`. Returns a standardized JSON object validated by Pydantic v2 schemas (`StandardizedDiagnosisResponse`).  
> - `GET /api/v1/health`: Returns server status, active CUDA device name, model load states, and active OOD thresholds.  
> - `GET /api/v1/config`: Exposes active temperature scaling parameters, energy cutoff thresholds, and class taxonomy versions.  
> - `GET /docs`: Interactive Swagger UI sandbox for real-time endpoint testing."*

---

### Q18: What is your contingency if the cloud connection is lost in rural farming environments?
> **Answer**:  
> *"Our system is designed with a **graceful offline degradation policy**. Because the primary CNN, U-Net, and YOLO models run locally on the inference node, they do not require an active internet connection.  
> If the Gemini API is unreachable or internet connectivity drops, the pipeline transitions seamlessly to the `CNN_ONLY` state. If the image passes all 4 safety gates, the local model provides the diagnosis and advisory. Only if the image fails safety gates does it safely advise physical inspection rather than guessing."*

---

### Q19: How did you ensure reproducibility in your model training?
> **Answer**:  
> *"All training scripts (`training/train_classifier.py`, `training/train_unet.py`) set deterministic seeds across Python (`random.seed`), NumPy (`np.random.seed`), and PyTorch (`torch.manual_seed` and `torch.cuda.manual_seed_all`), along with `torch.backends.cudnn.deterministic = True`.  
> Checkpoints store not just model weights, but optimizer states (`AdamW`), learning rate scheduler milestones (`CosineAnnealingLR`), training history JSON logs, and exact dataset class manifests (`weights/class_names.txt`)."*

---

### Q20: What are your key contributions as the AI/ML Developer on this project?
> **Answer**:  
> *"As the AI/ML Developer, my core contributions were:  
> 1. **Architecture Design**: Architecting the multi-model vision stack (EfficientNet-B5 + CBAM attention, U-Net, and YOLOv8).  
> 2. **Defensive AI Engineering**: Replacing naive Softmax confidence with Energy-based Out-of-Distribution detection and multi-gate safety checks.  
> 3. **Consensus Engineering**: Designing the 6-state consensus resolution table pairing local deep learning with multimodal LLM reasoning.  
> 4. **Production Deployment**: Building the end-to-end FastAPI microservice, Dockerizing the runtime, and deploying the live HTTPS Cloudflare tunnel running on local GPU acceleration."*

---

## 🎯 Quick Review Cheat Sheet Before You Walk Into the Evaluation

| Concept | The One-Line Punchline to Tell the Jury |
| :--- | :--- |
| **EfficientNet-B5** | Compound scaling balances depth, width, and resolution for fine-grained foliar features. |
| **CBAM Attention** | Channel Attention finds *what* disease to look for; Spatial Attention finds *where* it is on the leaf. |
| **U-Net ResNet-34** | Skip connections preserve pixel-level lesion boundaries to calculate exact infected surface area $\%$. |
| **YOLOv8n** | Anchor-free real-time detector localizing 14 agricultural pest classes with counts and bounding boxes. |
| **Energy OOD** | Uses unnormalized logits to track data density, preventing overconfident softmax errors on non-crops. |
| **Consensus Resolver** | Prevents hallucinations by requiring deterministic consensus between CNN and Gemini Vision before recommending chemicals. |
| **GPU Inference** | Runs in $\sim 145\text{ms}$ on NVIDIA RTX 5060 with PyTorch FP16 and FastAPI. |
