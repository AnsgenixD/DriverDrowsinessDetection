# quantization.py - Utilities to quantize models to INT8 for faster CPU inference

import platform
import os
import torch
import torch.nn as nn
from models.custom_cnn import Custom2DCNN_Fusion
from models.mobilenet_v3 import MobileNetV3Small_Fusion


def setup_quantization_engine():
    """Configure appropriate backend engine for current CPU architecture."""
    supported = torch.backends.quantized.supported_engines
    is_arm = platform.machine() in ["arm64", "aarch64", "armv7l"]

    if is_arm and "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    print(f"[Quantization] Active backend engine: {torch.backends.quantized.engine}")


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """
    Applies Post-Training Dynamic INT8 Quantization to Linear layers.
    Ideal for fast CPU execution with zero calibration data needed.
    """
    setup_quantization_engine()
    model.eval()
    model_cpu = model.to("cpu")

    try:
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )
        print("[Quantization] Dynamic INT8 quantization applied successfully.")
        return quantized_model
    except Exception as e:
        print(f"[Quantization] Warning: Quantization encountered {e}. Returning Float32 model.")
        return model_cpu


def save_quantized_model(model: nn.Module, save_path: str):
    """Save quantized model state dictionary."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[Quantization] Model saved to {save_path} ({size_mb:.2f} MB).")
