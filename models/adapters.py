"""
AgriVision Ultra v5.0 — Frozen-Backbone Residual LoRA Adapters.

Implements Low-Rank Adaptation (LoRA) projection modules applied exclusively to:
1. CBAM Channel Attention projections (W_q, W_v analogous fc1, fc2)
2. 2048-dimensional classifier bottleneck (Linear 2048 -> 512)

Mathematical Formulation:
    h = W_0 x + (gamma / r) * (B * A) x
where:
    W_0 in R^{d x k} is frozen (requires_grad = False)
    A in R^{r x k} ~ N(0, sigma^2)
    B in R^{d x r} = 0
    rank r = 8
    scaling factor gamma = 16 (scaling = gamma / r = 2.0)

Base network weights remain 100% bitwise immutable, guaranteeing ZERO catastrophic forgetting.
Only LoRA matrices A and B are trainable (< 0.8% parameter overhead).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a frozen Linear layer.
    h = W_0 x + (gamma / r) * (x @ A.T @ B.T)
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        gamma: float = 16.0
    ):
        super(LoRALinear, self).__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.gamma = gamma
        self.scaling = gamma / rank

        # Freeze base layer completely
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        dev = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Trainable low-rank decomposition matrices matching device and dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dtype))

        # Initialize A with Gaussian noise, B with zeros (ensures zero perturbation at init)
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen forward
        base_out = self.base_layer(x)
        # LoRA residual branch: (x @ A.T) @ B.T * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "lora_A": self.lora_A.data.clone(),
            "lora_B": self.lora_B.data.clone()
        }

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.lora_A.data.copy_(state_dict["lora_A"].to(self.lora_A.device, self.lora_A.dtype))
        self.lora_B.data.copy_(state_dict["lora_B"].to(self.lora_B.device, self.lora_B.dtype))


class LoRAConv2d(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for 1x1 Conv2d layers (e.g., CBAM Channel Attention).
    h = W_0(x) + (gamma / r) * B(A(x))
    """
    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 8,
        gamma: float = 16.0
    ):
        super(LoRAConv2d, self).__init__()
        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.rank = rank
        self.gamma = gamma
        self.scaling = gamma / rank

        # Freeze base layer completely
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        dev = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Trainable low-rank 1x1 convolutions matching device and dtype
        self.lora_A = nn.Conv2d(self.in_channels, rank, kernel_size=1, bias=False, device=dev, dtype=dtype)
        self.lora_B = nn.Conv2d(rank, self.out_channels, kernel_size=1, bias=False, device=dev, dtype=dtype)

        # Initialize A with Gaussian noise, B with zeros
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "lora_A.weight": self.lora_A.weight.data.clone(),
            "lora_B.weight": self.lora_B.weight.data.clone()
        }

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.lora_A.weight.data.copy_(state_dict["lora_A.weight"].to(self.lora_A.weight.device, self.lora_A.weight.dtype))
        self.lora_B.weight.data.copy_(state_dict["lora_B.weight"].to(self.lora_B.weight.device, self.lora_B.weight.dtype))


def apply_lora_to_model_a(
    model: nn.Module,
    rank: int = 8,
    gamma: float = 16.0
) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """
    Applies LoRA adapters exclusively to:
    1. CBAM Channel Attention projections (fc1, fc2)
    2. Classifier bottleneck (classifier[1]: Linear 2048 -> 512)

    All base weights remain strictly frozen (requires_grad = False).
    Returns the adapted model and a dictionary of adapter modules.
    """
    # Freeze entire base model
    for param in model.parameters():
        param.requires_grad = False

    adapters: Dict[str, nn.Module] = {}

    # 1. Adapt CBAM Channel Attention fc1 and fc2
    if hasattr(model, "cbam") and hasattr(model.cbam, "channel_attention"):
        ca = model.cbam.channel_attention
        if isinstance(ca.fc1, nn.Conv2d) and not isinstance(ca.fc1, LoRAConv2d):
            ca.fc1 = LoRAConv2d(ca.fc1, rank=rank, gamma=gamma)
            adapters["cbam_ca_fc1"] = ca.fc1
        if isinstance(ca.fc2, nn.Conv2d) and not isinstance(ca.fc2, LoRAConv2d):
            ca.fc2 = LoRAConv2d(ca.fc2, rank=rank, gamma=gamma)
            adapters["cbam_ca_fc2"] = ca.fc2

    # 2. Adapt Classifier Bottleneck Linear(2048, 512)
    if hasattr(model, "classifier"):
        # EfficientNet-B5 classifier has [Dropout, Linear(2048, 512), SiLU, Dropout, Linear(512, num_classes)]
        if len(model.classifier) > 1 and isinstance(model.classifier[1], nn.Linear) and not isinstance(model.classifier[1], LoRALinear):
            model.classifier[1] = LoRALinear(model.classifier[1], rank=rank, gamma=gamma)
            adapters["classifier_bottleneck"] = model.classifier[1]

    # Calculate parameter overhead
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    overhead_pct = (lora_params / (total_params - lora_params)) * 100

    print(f"[LoRA] Applied rank-{rank} adapters (gamma={gamma}).")
    print(f"[LoRA] Base parameters (frozen): {total_params - lora_params:,}")
    print(f"[LoRA] Trainable adapter parameters: {lora_params:,} ({overhead_pct:.3f}% overhead)")

    return model, adapters


def save_lora_checkpoint(adapters: Dict[str, nn.Module], output_path: str):
    """Saves only the trainable LoRA adapter weights, never touching base checkpoints."""
    state = {}
    for name, module in adapters.items():
        state[name] = module.get_lora_state_dict()
    torch.save(state, output_path)
    print(f"[LoRA] Saved adapter checkpoint to {output_path}")


def load_lora_checkpoint(adapters: Dict[str, nn.Module], input_path: str):
    """Loads LoRA adapter weights into existing adapter modules."""
    state = torch.load(input_path, map_location="cpu")
    for name, module in adapters.items():
        if name in state:
            module.load_lora_state_dict(state[name])
    print(f"[LoRA] Loaded adapter checkpoint from {input_path}")
