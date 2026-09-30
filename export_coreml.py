# export_coreml.py - Converts trained PyTorch model to Apple CoreML (.mlpackage) format

import os
import sys
import argparse
from pathlib import Path
import torch
import coremltools as ct

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.custom_cnn import Custom2DCNN_Fusion
from models.mobilenet_v3 import MobileNetV3Small_Fusion


def export_to_coreml(
    model_type: str = "mobilenet_v3",
    weights_path: str | None = None,
    output_path: str | None = None,
):
    print(f"============================================================")
    print(f" Exporting {model_type.upper()} to Apple CoreML (.mlpackage)")
    print(f"============================================================")

    # 1. Instantiate PyTorch Architecture
    if model_type == "mobilenet_v3":
        model = MobileNetV3Small_Fusion(num_classes=config.NUM_CLASSES, geom_dim=config.GEOM_FEATURE_DIM)
        default_out = "weights/MobileNetV3_DMS.mlpackage"
    else:
        model = Custom2DCNN_Fusion(num_classes=config.NUM_CLASSES, geom_dim=config.GEOM_FEATURE_DIM)
        default_out = "weights/DriverStateClassifier.mlpackage"

    out_file = output_path or default_out
    wt_path = weights_path or str(config.DEFAULT_WEIGHTS_PATH)

    # 2. Load Weights if present
    if os.path.exists(wt_path):
        state_dict = torch.load(wt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[Export] Loaded trained weights from {wt_path}")
    else:
        print(f"[Export] Notice: No weights file found at {wt_path}. Exporting initialized architecture.")

    model.eval()

    # 3. Create Sample Tensors for JIT Tracing
    dummy_img = torch.randn(1, 3, *config.IMAGE_INPUT_SIZE)
    dummy_geom = torch.randn(1, config.GEOM_FEATURE_DIM)
    traced_model = torch.jit.trace(model, (dummy_img, dummy_geom))

    # 4. CoreML Conversion
    print("[Export] Converting JIT model to CoreML MIL graph...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image_input", shape=(1, 3, *config.IMAGE_INPUT_SIZE)),
            ct.TensorType(name="geom_input", shape=(1, config.GEOM_FEATURE_DIM)),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )

    # 5. Metadata for Xcode / Swift Developer
    mlmodel.author = "Edge DMS Engineering Group"
    mlmodel.short_description = "Real-Time Driver Drowsiness and Distraction Detection Model"
    mlmodel.input_description["image_input"] = "Normalized 112x112 composite ocular and perioral crop (1, 3, 112, 112)"
    mlmodel.input_description["geom_input"] = "7D geometric feature vector [EAR_l, EAR_r, EAR_avg, MAR, Yaw, Pitch, Roll]"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mlmodel.save(out_file)

    print(f"\n[Success] CoreML model successfully exported to: {out_file}")
    print(f"Model ready for drag-and-drop into your Xcode Swift project!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch DMS to CoreML")
    parser.add_argument("--model", type=str, choices=["custom_cnn", "mobilenet_v3"], default="mobilenet_v3")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    export_to_coreml(model_type=args.model, weights_path=args.weights, output_path=args.out)
