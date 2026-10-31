#!/usr/bin/env python3
"""
train.py
Fine-tunes YOLOv8 on KITTI dataset for autonomous driving 3-class detection.
Logs mAP@0.5 and mAP@0.5:0.95 and exports metrics for resume documentation.
"""

import os
import json
import argparse
from ultralytics import YOLO


def train_yolo(args):
    print("=" * 60)
    print("Fine-Tuning YOLOv8 on KITTI Autonomous Driving Dataset")
    print(f"Base Model: {args.model} | Epochs: {args.epochs} | ImgSz: {args.imgsz}")
    print(f"Data config: {args.data} | Device: {args.device}")
    print("=" * 60)

    # 1. Initialize YOLOv8 architecture with pretrained weights
    model = YOLO(args.model)

    # 2. Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        pretrained=True,
        plots=True,
        save=True
    )

    # 3. Validation on validation set
    print("\nRunning comprehensive validation on KITTI val set...")
    metrics = model.val()

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    # Per-class AP50
    class_names = ["Car", "Pedestrian", "Cyclist"]
    per_class_map50 = {}
    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) >= len(class_names):
        for idx, cls_name in enumerate(class_names):
            per_class_map50[cls_name] = float(metrics.box.maps[idx])

    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS (SAVE THESE FOR YOUR RESUME)")
    print("=" * 60)
    print(f"  Precision:       {mp:.4f} ({mp*100:.2f}%)")
    print(f"  Recall:          {mr:.4f} ({mr*100:.2f}%)")
    print(f"  mAP@0.5:         {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95:    {map50_95:.4f} ({map50_95*100:.2f}%)")
    if per_class_map50:
        print("  Per-Class AP@0.5:")
        for cls_name, ap in per_class_map50.items():
            print(f"    - {cls_name:12s}: {ap:.4f} ({ap*100:.2f}%)")
    print("=" * 60)

    # 4. Save results to JSON
    save_dir = os.path.join(args.project, args.name)
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "kitti_detection_metrics.json")
    
    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "precision": mp,
        "recall": mr,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class_mAP50": per_class_map50,
        "best_weights_path": os.path.join(save_dir, "weights", "best.pt")
    }

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Metrics saved to {metrics_path}")
    print(f"Best model weights saved to {summary['best_weights_path']}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on KITTI dataset.")
    parser.add_argument("--data", type=str, default="training/kitti.yaml", help="Path to kitti.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model weights (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID or 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project save directory")
    parser.add_argument("--name", type=str, default="train", help="Run experiment name")
    args = parser.parse_args()

    train_yolo(args)


if __name__ == "__main__":
    main()
