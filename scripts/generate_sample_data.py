#!/usr/bin/env python3
"""
generate_sample_data.py
Generates a lightweight synthetic KITTI tracking sequence with realistic moving vehicles,
pedestrians, and ground truth annotations. Enables instant end-to-end testing of the pipeline
without downloading the full multi-gigabyte KITTI dataset.
"""

import os
import cv2
import numpy as np


def generate_synthetic_kitti_sequence(
    output_dir="data/kitti/tracking/training",
    seq_name="0000",
    num_frames=60,
    width=1242,
    height=375
):
    seq_img_dir = os.path.join(output_dir, "image_02", seq_name)
    seq_lbl_dir = os.path.join(output_dir, "label_02")
    os.makedirs(seq_img_dir, exist_ok=True)
    os.makedirs(seq_lbl_dir, exist_ok=True)

    lbl_file = os.path.join(seq_lbl_dir, f"{seq_name}.txt")

    # Define moving objects: [initial_cx, initial_cy, w, h, vx, vy, class_name, track_id]
    objects = [
        {"id": 0, "cls": "Car", "cx": 200.0, "cy": 220.0, "w": 180.0, "h": 90.0, "vx": 5.5, "vy": 0.5, "color": (40, 40, 200)},
        {"id": 1, "cls": "Car", "cx": 950.0, "cy": 210.0, "w": 160.0, "h": 85.0, "vx": -4.0, "vy": 0.2, "color": (200, 40, 40)},
        {"id": 2, "cls": "Pedestrian", "cx": 550.0, "cy": 230.0, "w": 45.0, "h": 110.0, "vx": 1.2, "vy": -0.1, "color": (40, 200, 40)},
    ]

    print(f"Generating synthetic KITTI sequence '{seq_name}' ({num_frames} frames)...")
    gt_lines = []

    for f_idx in range(num_frames):
        # Create road & background synthetic scene
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky
        img[:int(height * 0.45), :] = [180, 160, 140]
        # Horizon & Road
        img[int(height * 0.45):, :] = [65, 65, 65]
        # Road lane markings
        for lx in range(0, width, 80):
            cv2.line(img, (lx, int(height * 0.75)), (lx + 40, int(height * 0.75)), (255, 255, 255), 3)

        # Draw objects and record GT labels
        for obj in objects:
            cx = obj["cx"] + obj["vx"] * f_idx
            cy = obj["cy"] + obj["vy"] * f_idx
            w, h = obj["w"], obj["h"]

            l = max(0.0, cx - w / 2.0)
            t = max(0.0, cy - h / 2.0)
            r = min(float(width), cx + w / 2.0)
            b = min(float(height), cy + h / 2.0)

            # Draw vehicle / pedestrian box on synthetic frame
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), obj["color"], -1)
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (255, 255, 255), 2)
            cv2.putText(
                img, f"{obj['cls']} #{obj['id']}", (int(l), int(t) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

            # KITTI Tracking ground truth format:
            # frame track_id type truncated occluded alpha l t r b h w l x y z rot_y [score]
            gt_line = f"{f_idx} {obj['id']} {obj['cls']} 0 0 -1.0 {l:.2f} {t:.2f} {r:.2f} {b:.2f} 1.6 1.8 4.2 0 0 10 0 1.0"
            gt_lines.append(gt_line)

        # Save frame image (000000.png, 000001.png, ...)
        img_filename = f"{f_idx:06d}.png"
        cv2.imwrite(os.path.join(seq_img_dir, img_filename), img)

    # Write ground truth file
    with open(lbl_file, "w") as f:
        f.write("\n".join(gt_lines) + "\n")

    print(f"Sample images saved to: {seq_img_dir}")
    print(f"Sample GT annotations saved to: {lbl_file}")


def main():
    generate_synthetic_kitti_sequence()


if __name__ == "__main__":
    main()
