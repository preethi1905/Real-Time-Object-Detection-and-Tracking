#!/usr/bin/env python3
"""
split_dataset.py
Splits converted KITTI YOLO dataset into an 80/20 train/val split
with the standard directory structure required by Ultralytics YOLOv8.
"""

import os
import shutil
import random
import argparse
from tqdm import tqdm


def split_dataset(img_dir, label_dir, output_root, split_ratio=0.8, seed=42):
    random.seed(seed)

    train_img_dir = os.path.join(output_root, "images", "train")
    val_img_dir = os.path.join(output_root, "images", "val")
    train_lbl_dir = os.path.join(output_root, "labels", "train")
    val_lbl_dir = os.path.join(output_root, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect all image files that have matching label files
    valid_pairs = []
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        base_name = os.path.splitext(f)[0]
        img_name = base_name + ".png"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = base_name + ".jpg"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue
        valid_pairs.append((img_path, os.path.join(label_dir, f), img_name, f))

    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Total dataset samples: {len(valid_pairs)}")
    print(f"Training split: {len(train_pairs)} (80%)")
    print(f"Validation split: {len(val_pairs)} (20%)")

    def copy_subset(pairs, target_img_dir, target_lbl_dir, desc):
        for img_src, lbl_src, img_name, lbl_name in tqdm(pairs, desc=desc):
            shutil.copy2(img_src, os.path.join(target_img_dir, img_name))
            shutil.copy2(lbl_src, os.path.join(target_lbl_dir, lbl_name))

    copy_subset(train_pairs, train_img_dir, train_lbl_dir, "Copying Train set")
    copy_subset(val_pairs, val_img_dir, val_lbl_dir, "Copying Val set")

    print(f"Dataset successfully prepared in: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Split KITTI dataset into Train and Val sets for YOLO.")
    parser.add_argument("--img-dir", type=str, default="data/kitti/detection/training/image_2",
                        help="Path to KITTI image_2 directory")
    parser.add_argument("--label-dir", type=str, default="data/kitti/yolo_labels",
                        help="Path to converted YOLO labels directory")
    parser.add_argument("--out-dir", type=str, default="datasets/kitti",
                        help="Root directory for YOLO dataset")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train split fraction (default 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    args = parser.parse_args()

    split_dataset(args.img_dir, args.label_dir, args.out_dir, args.split, args.seed)


if __name__ == "__main__":
    main()
