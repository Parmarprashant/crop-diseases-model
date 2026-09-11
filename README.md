---
title: AgriVision Crop Disease Model API
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# AgriVision AI - Cloud Model Inference Engine

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

## Cloud Deployment Guide

### Option 1: Hugging Face Spaces (Recommended - Free 16 GB RAM)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set:
   - **Space Name**: `crop-diseases-model`
   - **License**: `mit` or `apache-2.0`
   - **SDK**: **Docker** (Blank)
   - **Hardware**: **CPU basic (2 vCPU · 16 GB RAM · Free)**
3. In **Settings -> Variables and Secrets**:
   - Add Secret: `GEMINI_API_KEY` = your Gemini API key (optional, for Gemini fallback).
4. Connect and push your repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/crop-diseases-model
   git push space main
   ```
   *Note: Hugging Face natively builds the `Dockerfile` and boots on port 7860.*

---

### Option 2: Docker Deployment (Cloud VM / AWS / GCP / DigitalOcean)

```bash
# 1. Build the production container
docker build -t crop-disease-api .

# 2. Run with environment variables
docker run -d \
  -p 7860:7860 \
  -e GEMINI_API_KEY="your_api_key_here" \
  --name crop-api \
  crop-disease-api
```

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

- `GET /`: API discovery and model status metadata.
- `GET /docs`: Interactive Swagger API documentation and testing sandbox.
- `POST /api/v1/diagnose`: Main inference endpoint accepting image upload.
- `GET /api/v1/health`: Telemetry, device, and pipeline readiness check.
- `GET /api/v1/config`: Active thresholds and gating parameters.
- `GET /api/v1/classes`: Supported crop and disease taxonomy manifest.

---

## Local Development Server

```bash
# 1. Setup environment variables
cp .env.example .env

# 2. Launch FastAPI Cloud Server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive web dashboard in your browser:
```
http://localhost:8000
```
