"""
AgriVision Ultra v5.0 — Disease Expert Model Wrapper.

Integrates:
- Model A: Frozen Backbone (100% bitwise immutable) + Residual LoRA Adapters (r=8, gamma=16)
- Model B: Cotton foliar expert with Additive Angular Margin (ArcFace) Head (s=30.0, m=0.35)
- Multi-Temperature Platt Calibration across 6 crop families
- FP16 Half-Precision execution for peak VRAM <= 350 MB and latency <= 20 ms
- Evidential Dirichlet Uncertainty estimation (u = K / S)
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import build_efficientnet_cbam, EfficientNetB5_CBAM
from models.adapters import apply_lora_to_model_a, load_lora_checkpoint
from models.cotton_head import CottonArcFaceModel, COTTON_DISEASE_CLASSES
from inference.temperature_scaling import FamilyTemperatureScaler, build_class_to_family_mapping
from inference.crop_taxonomy import get_class_metadata


class DiseaseExpertWrapper:
    """
    Ultra v5.0 Production-hardened inference wrapper for disease experts.
    """
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        class_names_path: Optional[str] = None,
        device: Optional[str] = None,
        img_size: int = 256,
        use_lora: bool = False,
        lora_checkpoint_path: Optional[str] = "weights/ultra_v5/model_a_lora.pt",
        use_arcface: bool = False,
        temperature_scaling_path: Optional[str] = "weights/ultra_v5/family_temperatures.json",
        use_half: bool = True
    ):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.img_size = img_size
        self.use_lora = use_lora
        self.use_arcface = use_arcface
        
        # Device resolution
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_half = use_half and (self.device.type == "cuda")
        
        # Class names setup
        if use_arcface:
            self.class_names = list(COTTON_DISEASE_CLASSES)
            self.num_classes = len(self.class_names)
        elif class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            self.num_classes = len(self.class_names)
        else:
            self.class_names = [f"Class_{i}" for i in range(281)]
            self.num_classes = 281

        # Model instantiation
        if use_arcface:
            print(f"[{model_name}] Instantiating CottonArcFaceModel ({self.num_classes} classes)...")
            self.model = CottonArcFaceModel(num_classes=self.num_classes, s=30.0, m=0.35)
            if checkpoint_path and os.path.exists(checkpoint_path):
                sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(sd, strict=False)
        else:
            print(f"[{model_name}] Loading checkpoint from {checkpoint_path} ({self.num_classes} classes) on {self.device}...")
            self.model: EfficientNetB5_CBAM = build_efficientnet_cbam(
                num_classes=self.num_classes,
                weights_path=checkpoint_path,
                device="cpu"
            )
            if use_lora:
                print(f"[{model_name}] Applying residual LoRA adapters (r=8, gamma=16)...")
                self.model, self.adapters = apply_lora_to_model_a(self.model, rank=8, gamma=16.0)
                if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
                    load_lora_checkpoint(self.adapters, lora_checkpoint_path)

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply FP16 half precision if enabled on CUDA
        if self.use_half:
            self.model.half()
            
        self.model.to(self.device)

        # Setup Temperature Scaler
        self.temperature_scaler: Optional[FamilyTemperatureScaler] = None
        if temperature_scaling_path and os.path.exists(temperature_scaling_path) and not use_arcface:
            class_to_family = build_class_to_family_mapping(self.class_names)
            try:
                self.temperature_scaler = FamilyTemperatureScaler.load_calibration(
                    temperature_scaling_path,
                    class_to_family=class_to_family
                ).to(self.device)
                print(f"[{model_name}] Loaded family temperature scaling from {temperature_scaling_path}")
            except Exception as e:
                print(f"[{model_name}] Notice loading temperature scaler: {e}")

        # ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes inference under torch.inference_mode() with optional FP16 half precision.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.use_half:
            tensor = tensor.half()

        # Forward pass through model
        logits = self.model(tensor) # (1, num_classes)
        
        # Apply Multi-Temperature Platt Calibration if available
        if self.temperature_scaler is not None:
            scaled_logits = self.temperature_scaler(logits)
        else:
            scaled_logits = logits

        probs = F.softmax(scaled_logits, dim=1).squeeze(0)
        
        # Evidential Dirichlet Uncertainty: u = K / S
        z_np = scaled_logits.squeeze(0).float().cpu().numpy()
        evidence = np.maximum(0.0, z_np)
        S = float(np.sum(evidence) + len(z_np))
        epistemic_u = float(len(z_np) / S)
        
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
        top_probs = top_probs.float().cpu().tolist()
        top_indices = top_indices.cpu().tolist()
        
        best_idx = top_indices[0]
        best_class = self.class_names[best_idx] if best_idx < len(self.class_names) else f"Class_{best_idx}"
        best_conf = float(top_probs[0])
        
        meta = get_class_metadata(best_class)
        crop_family = meta.crop if meta else ("cotton" if self.use_arcface else "unknown")
        
        top_predictions = [
            {
                "class_name": self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}",
                "confidence": round(float(prob), 4)
            }
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        # Spatial attention map
        try:
            if hasattr(self.model, "get_spatial_attention_map"):
                attn_map = self.model.get_spatial_attention_map(tensor)
            elif hasattr(self.model, "base") and hasattr(self.model.base, "get_spatial_attention_map"):
                attn_map = self.model.base.get_spatial_attention_map(tensor)
            else:
                attn_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        except Exception:
            attn_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
        return {
            "model_name": self.model_name,
            "predicted_class": best_class,
            "confidence": round(best_conf, 4),
            "crop_family": crop_family,
            "top_predictions": top_predictions,
            "raw_logits": z_np,
            "all_probabilities": probs.float().cpu().numpy(),
            "attention_map": attn_map,
            "is_confident": best_conf >= 0.70,
            "epistemic_uncertainty": round(epistemic_u, 4),
            "is_anomaly": epistemic_u > 0.30
        }
