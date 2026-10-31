#!/usr/bin/env python3
"""
convert_kitti.py
Converts KITTI 2D Object Detection annotations to normalized YOLO format.
Filters for the 3 target classes: Car (0), Pedestrian (1), Cyclist (2).
"""

import os
import argparse
import cv2
from tqdm import tqdm

CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def convert_kitti_to_yolo(label_dir, img_dir, out_dir):
    """
    Args:
        label_dir (str): Directory containing KITTI label .txt files (e.g. data_object_label_2/training/label_2)
        img_dir (str): Directory containing KITTI image .png files (e.g. data_object_image_2/training/image_2)
        out_dir (str): Directory where YOLO formatted .txt labels will be saved
    """
    os.makedirs(out_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    print(f"Converting {len(label_files)} annotation files from '{label_dir}'...")

    converted_count = 0
    total_objects = 0

    for f in tqdm(label_files, desc="Converting labels"):
        # Match corresponding image file
        img_name = f.replace(".txt", ".png")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            # Check for .jpg fallback
            img_name = f.replace(".txt", ".jpg")
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

        # Read image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        with open(os.path.join(label_dir, f), "r") as lf:
            for line in lf:
                p = line.strip().split()
                if not p:
                    continue
                obj_type = p[0]
                if obj_type not in CLASSES:
                    continue

                # KITTI 2D box coordinates [left, top, right, bottom]
                l, t, r, b = map(float, p[4:8])

                # Clamp to image boundaries
                l = max(0.0, min(float(w), l))
                r = max(0.0, min(float(w), r))
                t = max(0.0, min(float(h), t))
                b = max(0.0, min(float(h), b))

                if r <= l or b <= t:
                    continue

                # Normalized YOLO format: class_id x_center y_center width height
                x_center = (l + r) / 2.0 / w
                y_center = (t + b) / 2.0 / h
                box_w = (r - l) / w
                box_h = (b - t) / h

                lines.append(f"{CLASSES[obj_type]} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                total_objects += 1

        out_path = os.path.join(out_dir, f)
        with open(out_path, "w") as of:
            of.write("\n".join(lines))
        converted_count += 1

    print(f"Successfully converted {converted_count} files ({total_objects} total target objects).")
    print(f"Output saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert KITTI detection labels to YOLO format.")
    parser.add_argument("--label-dir", type=str, default="data/kitti/detection/training/label_2",
                        help="Path to KITTI ground truth label_2 directory")
    parser.add_argument("--img-dir", type=str, default="data/kitti/detection/training/image_2",
                        help="Path to KITTI image_2 directory")
    parser.add_argument("--out-dir", type=str, default="data/kitti/yolo_labels",
                        help="Output directory for YOLO labels")
    args = parser.parse_args()

    convert_kitti_to_yolo(args.label_dir, args.img_dir, args.out_dir)


if __name__ == "__main__":
    main()
