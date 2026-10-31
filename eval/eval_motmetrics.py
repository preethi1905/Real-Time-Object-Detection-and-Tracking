#!/usr/bin/env python3
"""
eval_motmetrics.py
Evaluates Multi-Object Tracking accuracy on KITTI sequences using py-motmetrics.
Reports standard CLEAR MOT metrics: MOTA, IDF1, ID Switches (IDSW), MT, ML, FP, FN.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add repo root to python path so kitti_tracking package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kitti_tracking.kitti_tracking.tracker import Tracker
from eval.kitti_eval_parser import parse_kitti_tracking_gt


def evaluate_sequence(gt_file, detections_by_frame=None, max_age=5, min_hits=2, iou_thr=0.3):
    """
    Evaluates tracking on a single KITTI sequence.
    If detections_by_frame is None, generates realistic detections directly from GT
    with simulated detection recall & localization noise.
    """
    import motmetrics as mm

    gt_by_frame = parse_kitti_tracking_gt(gt_file)
    max_frame = max(gt_by_frame.keys()) if gt_by_frame else 0

    acc = mm.MOTAccumulator(auto_id=True)
    tracker = Tracker(max_age=max_age, min_hits=min_hits, iou_thr=iou_thr)
    Tracker.reset()

    total_gt_objects = sum(len(objs) for objs in gt_by_frame.values())
    print(f"Evaluating {len(gt_by_frame)} frames with {total_gt_objects} total GT object annotations...")

    for frame_idx in range(max_frame + 1):
        gt_objs = gt_by_frame.get(frame_idx, [])
        gt_ids = [o["track_id"] for o in gt_objs]
        gt_boxes = [o["bbox"] for o in gt_objs]

        # Get frame detections
        if detections_by_frame is not None and frame_idx in detections_by_frame:
            frame_dets = detections_by_frame[frame_idx]
        else:
            # Generate detections from GT with realistic sensor noise
            frame_dets = []
            for o in gt_objs:
                # 95% detection rate (5% occasional detector drop)
                if np.random.rand() < 0.95:
                    box = np.array(o["bbox"], dtype=float)
                    # Gaussian jitter on corners (+- 1.5 pixels)
                    noise = np.random.normal(0, 1.5, size=4)
                    noisy_box = (box + noise).tolist()
                    frame_dets.append(noisy_box + [0.92, 0])
            # Occasional false positive
            if np.random.rand() < 0.05:
                frame_dets.append([100.0, 100.0, 150.0, 150.0, 0.40, 0])

        # Step Tracker
        active_tracks = tracker.step(frame_dets)
        hyp_ids = [t[0] for t in active_tracks]
        hyp_boxes = [t[1] for t in active_tracks]

        # Compute IoU distance matrix between GT and Tracker hypotheses
        # py-motmetrics takes [x, y, width, height] format for bounding boxes
        def to_xywh(boxes):
            if not boxes:
                return np.empty((0, 4))
            arr = np.array(boxes)
            x = arr[:, 0]
            y = arr[:, 1]
            w = arr[:, 2] - arr[:, 0]
            h = arr[:, 3] - arr[:, 1]
            return np.column_stack([x, y, w, h])

        gt_xywh = to_xywh(gt_boxes)
        hyp_xywh = to_xywh(hyp_boxes)

        # Distance matrix (1 - IoU), maximum distance allowed is 0.5 (IoU >= 0.5)
        dist_matrix = mm.distances.iou_matrix(gt_xywh, hyp_xywh, max_iou=0.5)

        # Update accumulator
        acc.update(gt_ids, hyp_ids, dist_matrix)

    # Compute MOT metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked',
            'mostly_lost', 'num_false_positives', 'num_misses',
            'num_switches', 'num_fragmentations', 'precision', 'recall'
        ],
        name=os.path.basename(gt_file)
    )

    return summary


def print_and_save_summary(summary, output_dir="results"):
    import motmetrics as mm
    os.makedirs(output_dir, exist_ok=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh_formatters(),
        namemap=mh_namemap()
    )

    print("\n" + "=" * 80)
    print("MULTIPLE OBJECT TRACKING (MOT) BENCHMARK RESULTS")
    print("=" * 80)
    print(strsummary)
    print("=" * 80)

    # Save CSV & Markdown
    csv_path = os.path.join(output_dir, "motmetrics_summary.csv")
    summary.to_csv(csv_path)

    json_path = os.path.join(output_dir, "motmetrics_summary.json")
    summary.to_json(json_path, orient="index", indent=4)

    print(f"Metrics saved to {csv_path} and {json_path}")


def mh_formatters():
    return {
        'num_frames': '{:d}'.format,
        'mota': '{:.2%}'.format,
        'motp': '{:.3f}'.format,
        'idf1': '{:.2%}'.format,
        'mostly_tracked': '{:d}'.format,
        'mostly_lost': '{:d}'.format,
        'num_false_positives': '{:d}'.format,
        'num_misses': '{:d}'.format,
        'num_switches': '{:d}'.format,
        'num_fragmentations': '{:d}'.format,
        'precision': '{:.2%}'.format,
        'recall': '{:.2%}'.format
    }


def mh_namemap():
    return {
        'num_frames': 'Frames',
        'mota': 'MOTA',
        'motp': 'MOTP',
        'idf1': 'IDF1',
        'mostly_tracked': 'MT',
        'mostly_lost': 'ML',
        'num_false_positives': 'FP',
        'num_misses': 'FN',
        'num_switches': 'IDSW',
        'num_fragmentations': 'Frag',
        'precision': 'Prec',
        'recall': 'Rec'
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker on KITTI tracking sequence.")
    parser.add_argument("--gt", type=str, default="data/kitti/tracking/training/label_02/0000.txt",
                        help="Path to KITTI ground truth label file")
    parser.add_argument("--max-age", type=int, default=5, help="Tracker max_age")
    parser.add_argument("--min-hits", type=int, default=2, help="Tracker min_hits")
    parser.add_argument("--iou-thr", type=float, default=0.3, help="Tracker IoU threshold")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    np.random.seed(42)

    if not os.path.exists(args.gt):
        print(f"Ground truth file '{args.gt}' not found.")
        print("Tip: Run `python scripts/generate_sample_data.py` to create mock KITTI data for instant testing.")
        return

    summary = evaluate_sequence(args.gt, max_age=args.max_age, min_hits=args.min_hits, iou_thr=args.iou_thr)
    print_and_save_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
