# AgriVision AI - Final Model Stack (Online / Cloud)

Autonomous multi-model agricultural intelligence platform designed exclusively for cloud inference and high-accuracy crop disease diagnostics, pest localization, and lesion segmentation.

---

## Model Stack Architecture

| Purpose | Model Architecture | Role |
| :--- | :--- | :--- |
| **Main Classification** | **EfficientNet-B5 + CBAM Attention** | Primary disease detection with highest accuracy, featuring Channel and Spatial attention modules and Grad-CAM/CBAM explainability heatmaps. |
| **Pest Detection & Localization** | **YOLOv8n** | Real-time object detection localizing agricultural pests with bounding boxes, labels, and count metrics. |
| **Lesion Segmentation** | **U-Net (ResNet-34 Encoder)** | Pixel-level semantic segmentation calculating exact infected leaf surface area percentage ($\text{Lesion Pixels} / \text{Leaf Pixels} \times 100\%$) and severity categorization (Mild, Moderate, Severe). |
| **Smart Fallback** | **Gemini 2.0 Flash / 1.5 Pro Vision** | Invoked automatically whenever main classification confidence is $< 0.80$, providing definitive multimodal botanical diagnosis and reasoning. |

---

## Inference Execution Flow

```
Farmer uploads photo
        │
        ▼
Server receives image
        │
        ├── EfficientNet-B5 + CBAM     → Disease Classification
        ├── YOLOv8n                    → Pest Detection
        └── U-Net                      → Lesion Segmentation
                │
                ▼
        Confidence Check
                │
                ├── ≥ 0.80  → Return final result + agronomic advisory
                │
                └── < 0.80  → Call Gemini Vision (Smart Fallback)
                                │
                                └── Return Gemini multimodal diagnosis + advisory
```

---

## Project Structure

All application code, models, and training scripts reside strictly within the `Model` directory and reference `MAIN DATA` directly:

```
d:\1winbackup\desktop\Ganpat University\Model\
├── MAIN DATA/                 # Agricultural training dataset (42+ classes)
├── models/
│   ├── cbam.py                # Channel & Spatial Attention PyTorch modules
│   ├── efficientnet_cbam.py   # EfficientNet-B5 with integrated CBAM & Grad-CAM
│   ├── yolo_pest.py           # YOLOv8n pest detection wrapper & box annotator
│   └── unet_segmenter.py      # ResNet-34 U-Net for leaf & lesion segmentation
├── core/
│   ├── pipeline.py            # Master concurrent orchestrator & confidence gate
│   ├── gemini_fallback.py     # Gemini 2.0 Flash / 1.5 Pro Vision fallback
│   └── advisory_engine.py     # Agronomic treatment recommendations
├── api/
│   ├── server.py              # FastAPI cloud application & endpoints
│   └── schemas.py             # Pydantic data schemas
├── training/
│   ├── dataset.py             # PyTorch loader sourcing strictly from MAIN DATA
│   ├── train_classifier.py    # EfficientNet-B5 + CBAM training script (AMP enabled)
│   ├── train_yolo.py          # YOLOv8n pest detector training script
│   └── train_unet.py          # ResNet-34 U-Net lesion training script
├── web/
│   ├── index.html             # High-aesthetic agritech dashboard
│   ├── styles.css             # Glassmorphism dark-theme styling
│   └── app.js                 # Interactive logic, canvas rendering, & visualizer
├── weights/                   # Directory for model checkpoints (.pt)
├── requirements.txt           # Python dependency specifications
├── .env.example               # Environment variables configuration template
└── README.md                  # System documentation
```

---

## Cloud API Endpoints

### 1. `POST /api/v1/diagnose`
Main cloud inference endpoint.
- **Payload**: `multipart/form-data` with `file: UploadFile` (Image)
- **Optional Query**: `threshold_override=0.80`
- **Response**: JSON matching `DiagnosisResponse` schema containing:
  - `decision_path`: `"PRIMARY_MODEL_ACCEPTED"` or `"GEMINI_VISION_FALLBACK"`
  - `diagnosis`: Disease title, confidence, top candidates, reasoning
  - `pests`: Count, bounding box coordinates, labels
  - `segmentation`: `infected_area_pct`, `severity_category`, pixel statistics
  - `advisory`: Chemical controls, organic/biological remedies, cultural practices
  - `visualizations`: Base64 images for Original, CBAM Attention Heatmap, YOLO Bounding Boxes, and U-Net Lesion Mask

### 2. `GET /api/v1/health`
Checks server readiness, GPU telemetry (RTX 5060), and loaded model statuses.

### 3. `GET /api/v1/config`
Inspects active pipeline thresholds and model parameters.

### 4. `GET /api/v1/classes`
Lists all supported disease and pest classes discovered from `MAIN DATA/Train`.

---

## Running the Cloud Server

```bash
# 1. Setup environment variables
copy .env.example .env

# 2. Launch FastAPI Cloud Server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive web dashboard in your browser:
```
http://localhost:8000
```
