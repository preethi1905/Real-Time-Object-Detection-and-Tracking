#!/usr/bin/env python3
"""
kitti_eval_parser.py
Utilities to parse KITTI Tracking ground truth (label_02) and format tracker outputs
for Multiple Object Tracking evaluation.
"""

import os
from collections import defaultdict


VALID_CLASSES = {"Car", "Pedestrian", "Cyclist"}


def parse_kitti_tracking_gt(label_file, target_classes=VALID_CLASSES):
    """
    Parses KITTI tracking ground truth label file (e.g. 0000.txt).
    Format per line:
    frame track_id type truncated occluded alpha left top right bottom ...

    Returns:
        dict: frame_id -> list of {'track_id': int, 'bbox': [l, t, r, b], 'type': str}
    """
    gt_by_frame = defaultdict(list)
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"KITTI GT file '{label_file}' not found.")

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]

            if target_classes and obj_type not in target_classes:
                continue

            l, t, r, b = map(float, parts[6:10])
            gt_by_frame[frame].append({
                "track_id": track_id,
                "bbox": [l, t, r, b],
                "type": obj_type
            })

    return gt_by_frame


def save_kitti_track_predictions(predictions, out_file):
    """
    Saves tracker outputs to a standard format.
    predictions: list of (frame_id, track_id, [l, t, r, b], score, type)
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for pred in predictions:
            frame_id, track_id, bbox, score, obj_type = pred
            l, t, r, b = bbox
            f.write(f"{frame_id} {track_id} {obj_type} -1 -1 -10 {l:.2f} {t:.2f} {r:.2f} {b:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}\n")
