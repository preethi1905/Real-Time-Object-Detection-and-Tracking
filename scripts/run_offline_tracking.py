#!/usr/bin/env python3
"""
run_offline_tracking.py
Step 3 verification script: Tests the Kalman + Hungarian Tracker outside ROS.
Runs tracking on a KITTI tracking image sequence offline, draws track IDs,
bounding boxes, velocity vectors, and trajectory trails on frames, and exports
an annotated video demo (results/offline_tracked_demo.mp4).
"""

import os
import sys
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm

# Add repo root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kitti_tracking.kitti_tracking.tracker import Tracker
from eval.kitti_eval_parser import parse_kitti_tracking_gt

KITTI_CLASSES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
NAME_TO_ID = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

COLOR_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 65),   (35, 181, 211),  (18, 123, 243),  (19, 71, 237),
    (67, 68, 245),   (133, 47, 239),  (203, 56, 241),  (245, 66, 179)
]


def get_color(track_id):
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


def run_offline_tracking(
    seq_dir="data/kitti/tracking/training/image_02/0000",
    gt_file="data/kitti/tracking/training/label_02/0000.txt",
    model_path=None,
    out_video_path="results/offline_tracked_demo.mp4",
    fps=10.0,
    max_age=5,
    min_hits=2,
    iou_thr=0.3
):
    if not os.path.exists(seq_dir):
        print(f"Error: Sequence directory '{seq_dir}' does not exist.")
        print("Run `python scripts/generate_sample_data.py` first to generate sample data.")
        return

    # Load image files
    img_files = [f for f in os.listdir(seq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    if not img_files:
        print(f"Error: No image files found in '{seq_dir}'")
        return

    # Check if YOLO detector is requested
    detector = None
    if model_path and os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            detector = YOLO(model_path)
            print(f"Loaded YOLOv8 detector from: {model_path}")
        except Exception as e:
            print(f"Could not load YOLO detector: {e}. Falling back to GT detections.")

    # Load GT detections as fallback / simulation
    gt_by_frame = {}
    if os.path.exists(gt_file):
        gt_by_frame = parse_kitti_tracking_gt(gt_file)
        print(f"Loaded ground truth for {len(gt_by_frame)} frames from '{gt_file}'")

    # Read first image to get dimensions
    first_frame = cv2.imread(os.path.join(seq_dir, img_files[0]))
    h, w = first_frame.shape[:2]

    # Initialize VideoWriter
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    # Initialize Tracker
    tracker = Tracker(max_age=max_age, min_hits=min_hits, iou_thr=iou_thr)
    Tracker.reset()

    track_history = {}
    print(f"\nProcessing {len(img_files)} frames and rendering video to '{out_video_path}'...")

    for f_idx, img_name in enumerate(tqdm(img_files, desc="Tracking frames")):
        frame_path = os.path.join(seq_dir, img_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # 1. Get detections
        detections = []
        if detector is not None:
            res = detector.predict(source=frame, conf=0.25, iou=0.45, verbose=False)
            if res and len(res) > 0:
                boxes = res[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    detections.append(xyxy + [conf, cls_id])
        elif f_idx in gt_by_frame:
            # Simulate detector with slight Gaussian noise
            for obj in gt_by_frame[f_idx]:
                box = np.array(obj["bbox"], dtype=float)
                noise = np.random.normal(0, 1.2, size=4)
                noisy_box = (box + noise).tolist()
                cid = NAME_TO_ID.get(obj["type"], 0)
                detections.append(noisy_box + [0.95, cid])

        # 2. Update Multi-Object Tracker
        active_tracks = tracker.step(detections)

        # 3. Render annotations on frame
        active_ids = set()
        for track in active_tracks:
            tid, bbox, cid, score, vel = track
            active_ids.add(tid)
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            color = get_color(tid)

            if tid not in track_history:
                track_history[tid] = deque(maxlen=30)
            track_history[tid].append((cx, cy))

            # Draw trajectory path
            pts = list(track_history[tid])
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j - 1], pts[j], color, 2)

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Velocity arrow
            if vel:
                vx, vy = vel
                cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * 2.5), int(cy + vy * 2.5)), (0, 255, 255), 2, tipLength=0.3)

            # Draw Label Tag
            cname = KITTI_CLASSES.get(cid, f"Class {cid}")
            label = f"ID:{tid} {cname} {score:.2f}"
            tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + tsize[0] + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD info
        hud = f"Frame: {f_idx:04d} | Active Tracks: {len(active_tracks)} | MOT: Kalman + Hungarian"
        cv2.rectangle(frame, (10, 10), (550, 45), (0, 0, 0), -1)
        cv2.putText(frame, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        out_writer.write(frame)

    out_writer.release()
    print(f"\n Tracking complete! Video saved to: {out_video_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run offline KITTI tracking and save video.")
    parser.add_argument("--seq", type=str, default="data/kitti/tracking/training/image_02/0000", help="Sequence directory")
    parser.add_argument("--gt", type=str, default="data/kitti/tracking/training/label_02/0000.txt", help="Ground truth file")
    parser.add_argument("--model", type=str, default=None, help="YOLOv8 weights path")
    parser.add_argument("--output", type=str, default="results/offline_tracked_demo.mp4", help="Output video file")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    args = parser.parse_args()

    run_offline_tracking(
        seq_dir=args.seq,
        gt_file=args.gt,
        model_path=args.model,
        out_video_path=args.output,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
