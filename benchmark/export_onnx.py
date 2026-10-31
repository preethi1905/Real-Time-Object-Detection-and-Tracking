#!/usr/bin/env python3
"""
export_onnx.py
Exports fine-tuned YOLOv8 PyTorch model (.pt) to optimized ONNX (.onnx) format
for high-throughput edge deployment and inference acceleration.
"""

import os
import argparse
from ultralytics import YOLO


def export_model_to_onnx(weights_path, imgsz=640, opset=12, dynamic=False, simplify=True):
    if not os.path.exists(weights_path):
        print(f"Weights file '{weights_path}' not found! Checking fallback 'yolov8n.pt'...")
        weights_path = "yolov8n.pt"

    print("=" * 60)
    print(f"Exporting YOLOv8 model '{weights_path}' to ONNX format")
    print(f"Image Size: {imgsz}x{imgsz} | Opset: {opset} | Dynamic: {dynamic} | Simplify: {simplify}")
    print("=" * 60)

    model = YOLO(weights_path)
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify
    )

    print(f"\nModel exported successfully to: {exported_path}")

    # Optional verification with ONNX library if installed
    try:
        import onnx
        onnx_model = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX graph validation passed: Model is structurally sound!")
    except ImportError:
        pass
    except Exception as e:
        print(f"ONNX check warning: {e}")

    return exported_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model to ONNX.")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt",
                        help="Path to YOLOv8 .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (default 640)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default 12)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch and spatial dimensions")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnxsim simplification")
    args = parser.parse_args()

    export_model_to_onnx(
        args.weights,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()
