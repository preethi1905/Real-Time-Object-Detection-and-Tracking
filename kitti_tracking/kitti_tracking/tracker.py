#!/usr/bin/env python3
"""
tracker.py
Core 2D Multiple Object Tracking using Kalman Filter and Hungarian Algorithm (SORT-style).
State vector: [cx, cy, area, aspect_ratio, vx, vy, v_area]
Observation:  [cx, cy, area, aspect_ratio]
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    
    area_a = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_b = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_a + area_b - inter
    
    if union <= 0.0:
        return 0.0
    return inter / union


class KalmanBoxTracker:
    """
    Tracks an individual bounding box using a 7-state Kalman Filter:
      x = [cx, cy, s, r, vx, vy, vs]^T
    where (cx, cy) is box center, s is area (scale), r is aspect ratio (w/h),
    and (vx, vy, vs) are the respective constant velocities.
    """
    count = 0

    def __init__(self, bbox, class_id=0, score=1.0):
        """
        Initialize a tracker instance from bounding box [x1, y1, x2, y2].
        """
        # State transition matrix F (7x7) - Constant Velocity Model
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # s  += vs

        # Measurement matrix H (4x7) - We observe [cx, cy, s, r]
        self.H = np.eye(4, 7, dtype=np.float64)

        # State error covariance P (7x7)
        self.P = np.eye(7, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial uncertainty on velocities

        # Process noise covariance Q (7x7)
        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01   # Small velocity process noise

        # Measurement noise covariance R (4x4)
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0   # Higher uncertainty on scale and aspect ratio

        # State vector x (7x1)
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = self._to_z(bbox)

        # Identification and metadata
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.history = []

    @staticmethod
    def _to_z(b):
        """
        Converts [x1, y1, x2, y2] bounding box to measurement vector [cx, cy, s, r]^T.
        """
        w = max(b[2] - b[0], 1e-4)
        h = max(b[3] - b[1], 1e-4)
        cx = b[0] + w / 2.0
        cy = b[1] + h / 2.0
        s = w * h
        r = w / h
        return np.array([[cx], [cy], [s], [r]], dtype=np.float64)

    def predict(self):
        """
        Advances the Kalman state vector and returns the predicted bounding box.
        """
        # Guard against degenerate area before prediction
        if (self.x[2] + self.x[6]) <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.misses += 1
        pred_box = self.bbox()
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox, class_id=None, score=None):
        """
        Updates the state estimate with observed bounding box [x1, y1, x2, y2].
        """
        z = self._to_z(bbox)
        y = z - self.H @ self.x  # Innovation / residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(7, dtype=np.float64) - K @ self.H) @ self.P
        
        self.hits += 1
        self.misses = 0
        if class_id is not None:
            self.class_id = class_id
        if score is not None:
            self.score = score

    def bbox(self):
        """
        Returns the current bounding box estimate in [x1, y1, x2, y2] format.
        """
        cx, cy, s, r = self.x[:4].ravel()
        # Ensure scale and ratio are physically valid
        s = max(s, 1e-4)
        r = max(r, 1e-4)
        w = np.sqrt(s * r)
        h = s / w
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def get_velocity(self):
        """
        Returns estimated [vx, vy] velocity in pixels/frame.
        """
        return self.x[4:6].ravel().tolist()


class Tracker:
    """
    Multi-Object Tracker utilizing Hungarian algorithm for data association
    and Kalman Filters for state estimation.
    """
    def __init__(self, max_age=5, min_hits=2, iou_thr=0.3, class_aware=False):
        """
        Args:
            max_age (int): Maximum frames to keep track alive without detections.
            min_hits (int): Minimum detections before track is reported.
            iou_thr (float): Minimum IoU threshold for matching.
            class_aware (bool): If True, only match detections of the same class.
        """
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.class_aware = class_aware

    @staticmethod
    def reset():
        """Resets the global ID counter."""
        KalmanBoxTracker.count = 0

    def step(self, detections):
        """
        Performs one tracking step given current frame detections.
        
        Args:
            detections: List of detections. Each detection can be:
                - [x1, y1, x2, y2]
                - [x1, y1, x2, y2, score, class_id]
                - dict with keys {'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int}
        
        Returns:
            List of confirmed active tracks:
                [(track_id, [x1, y1, x2, y2], class_id, score, [vx, vy])]
        """
        # Standardize detections into list of (bbox, score, class_id)
        parsed_dets = []
        for d in detections:
            if isinstance(d, dict):
                bbox = d['bbox']
                score = d.get('score', 1.0)
                cls_id = d.get('class_id', 0)
            elif len(d) == 4:
                bbox = d
                score = 1.0
                cls_id = 0
            elif len(d) >= 6:
                bbox = d[:4]
                score = float(d[4])
                cls_id = int(d[5])
            elif len(d) == 5:
                bbox = d[:4]
                score = float(d[4])
                cls_id = 0
            else:
                continue
            parsed_dets.append((bbox, score, cls_id))

        # 1. Predict all current trackers
        preds = [t.predict() for t in self.trackers]

        matched_t, matched_d = set(), set()

        # 2. Match predictions to detections using Hungarian Algorithm
        if len(preds) > 0 and len(parsed_dets) > 0:
            cost = np.zeros((len(preds), len(parsed_dets)), dtype=np.float64)
            for r, pred in enumerate(preds):
                t_cls = self.trackers[r].class_id
                for c, (det_box, _, d_cls) in enumerate(parsed_dets):
                    if self.class_aware and (t_cls != d_cls):
                        cost[r, c] = 1.0  # Maximal cost for class mismatch
                    else:
                        cost[r, c] = 1.0 - iou(pred, det_box)

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost[r, c] >= self.iou_thr:
                    det_box, det_score, det_cls = parsed_dets[c]
                    self.trackers[r].update(det_box, class_id=det_cls, score=det_score)
                    matched_t.add(r)
                    matched_d.add(c)

        # 3. Create new trackers for unmatched detections
        for i, (det_box, det_score, det_cls) in enumerate(parsed_dets):
            if i not in matched_d:
                self.trackers.append(KalmanBoxTracker(det_box, class_id=det_cls, score=det_score))

        # 4. Cull dead tracks (misses exceeded max_age)
        self.trackers = [t for t in self.trackers if t.misses <= self.max_age]

        # 5. Return confirmed active tracks (matched this frame: misses == 0)
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits and t.misses == 0:
                results.append((t.id, t.bbox(), t.class_id, t.score, t.get_velocity()))

        return results


if __name__ == "__main__":
    print("Testing KalmanBoxTracker and Tracker...")
    Tracker.reset()
    tracker = Tracker(max_age=3, min_hits=2, iou_thr=0.3)

    # Frame 1: Detection at [10, 10, 50, 50]
    frame1_dets = [[10.0, 10.0, 50.0, 50.0, 0.95, 0]]
    res1 = tracker.step(frame1_dets)
    print(f"Frame 1 tracks (expected 0 confirmed hits<2): {res1}")

    # Frame 2: Object moved slightly to [15, 12, 55, 52]
    frame2_dets = [[15.0, 12.0, 55.0, 52.0, 0.96, 0]]
    res2 = tracker.step(frame2_dets)
    print(f"Frame 2 tracks (expected 1 confirmed hit): {res2}")
    assert len(res2) == 1, "Expected 1 confirmed track on second detection!"
    track_id, box, cls_id, score, vel = res2[0]
    print(f"Track ID: {track_id}, Box: {np.round(box, 2)}, Vel: {np.round(vel, 2)}")

    # Frame 3: Object moved to [20, 14, 60, 54]
    frame3_dets = [[20.0, 14.0, 60.0, 54.0, 0.97, 0]]
    res3 = tracker.step(frame3_dets)
    print(f"Frame 3 tracks: {res3}")
    assert res3[0][0] == track_id, "Track ID must remain consistent!"
    print("KalmanBoxTracker & Tracker verification passed successfully!")
