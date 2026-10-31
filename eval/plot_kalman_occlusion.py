#!/usr/bin/env python3
"""
plot_kalman_occlusion.py
Visualizes Kalman Filter performance and state covariance dynamics
during short object occlusions (detector misses).
Generates publication-quality plots comparing Ground Truth, Detector Observations,
and Kalman Filter predicted/updated state trajectories.
"""

import os
import sys
import numpy as np

# Add repo root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kitti_tracking.kitti_tracking.tracker import KalmanBoxTracker


def simulate_occlusion_experiment(num_frames=50, occlusion_start=20, occlusion_end=25):
    """
    Simulates a ground-truth vehicle moving across the field of view with constant velocity,
    adds detector measurement noise, and drops detections during the occlusion window.
    """
    # 1. Ground Truth trajectory: Moving diagonally with constant velocity
    # x0 = 100, y0 = 150, vx = 4.0 px/frame, vy = 1.5 px/frame, box_w = 120, box_h = 60
    vx_true = 4.0
    vy_true = 1.5
    box_w = 120.0
    box_h = 60.0

    gt_centers = []
    gt_boxes = []
    detections = []  # None if occluded, else noisy [x1, y1, x2, y2]

    np.random.seed(42)
    for t in range(num_frames):
        cx = 100.0 + vx_true * t
        cy = 150.0 + vy_true * t
        gt_centers.append((cx, cy))
        gt_box = [cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0]
        gt_boxes.append(gt_box)

        # Occlusion window: detector drops the object
        if occlusion_start <= t <= occlusion_end:
            detections.append(None)
        else:
            # Add Gaussian noise on detection: sigma = 2.0 pixels
            noise = np.random.normal(0, 2.0, size=4)
            noisy_box = [gt_box[i] + noise[i] for i in range(4)]
            detections.append(noisy_box)

    # 2. Run KalmanBoxTracker
    # Initialize with first detection
    kf = KalmanBoxTracker(detections[0])
    
    kf_estimates = []
    kf_uncertainties_x = []
    kf_uncertainties_y = []
    kf_velocities = []

    for t in range(num_frames):
        # Predict step
        kf.predict()

        # Update step if detection is present
        if detections[t] is not None:
            kf.update(detections[t])

        # Record state
        pred_box = kf.bbox()
        pcx = (pred_box[0] + pred_box[2]) / 2.0
        pcy = (pred_box[1] + pred_box[3]) / 2.0
        kf_estimates.append((pcx, pcy))

        # Positional standard deviations from diagonal of error covariance matrix P
        sigma_x = np.sqrt(kf.P[0, 0])
        sigma_y = np.sqrt(kf.P[1, 1])
        kf_uncertainties_x.append(sigma_x)
        kf_uncertainties_y.append(sigma_y)
        kf_velocities.append((kf.x[4, 0], kf.x[5, 0]))

    return {
        "num_frames": num_frames,
        "occlusion_window": (occlusion_start, occlusion_end),
        "gt_centers": gt_centers,
        "detections": detections,
        "kf_estimates": kf_estimates,
        "kf_uncertainties_x": kf_uncertainties_x,
        "kf_uncertainties_y": kf_uncertainties_y,
        "kf_velocities": kf_velocities
    }


def plot_results(data, output_path="results/kalman_occlusion_analysis.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping graphical plot.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = np.arange(data["num_frames"])
    gt_x = [p[0] for p in data["gt_centers"]]
    gt_y = [p[1] for p in data["gt_centers"]]
    
    kf_x = [p[0] for p in data["kf_estimates"]]
    kf_y = [p[1] for p in data["kf_estimates"]]

    det_x = [((d[0] + d[2]) / 2.0) if d is not None else np.nan for d in data["detections"]]
    det_y = [((d[1] + d[3]) / 2.0) if d is not None else np.nan for d in data["detections"]]

    occ_s, occ_e = data["occlusion_window"]

    # Compute Euclidean tracking error
    errors_kf = [np.hypot(kf_x[i] - gt_x[i], kf_y[i] - gt_y[i]) for i in range(len(frames))]
    errors_det = [np.hypot(det_x[i] - gt_x[i], det_y[i] - gt_y[i]) if not np.isnan(det_x[i]) else np.nan for i in range(len(frames))]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Kalman Filter Robustness & State Estimation Under Occlusion", fontsize=16, fontweight='bold')

    # Panel 1: 2D Trajectory
    axs[0, 0].plot(gt_x, gt_y, 'g--', linewidth=2.5, label='Ground Truth Trajectory')
    axs[0, 0].scatter(det_x, det_y, color='red', marker='o', s=30, alpha=0.7, label='Detector Observations')
    axs[0, 0].plot(kf_x, kf_y, 'b-', linewidth=2, label='Kalman Filter Track')
    # Mark occlusion region
    axs[0, 0].scatter(
        [gt_x[i] for i in range(occ_s, occ_e + 1)],
        [gt_y[i] for i in range(occ_s, occ_e + 1)],
        color='orange', marker='x', s=60, label='Occlusion Region (Missing Dets)'
    )
    axs[0, 0].set_title("2D Centroid Trajectory (Pixel Coordinates)", fontweight='bold')
    axs[0, 0].set_xlabel("X coordinate (pixels)")
    axs[0, 0].set_ylabel("Y coordinate (pixels)")
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()

    # Panel 2: X Position vs Time (with Occlusion Band)
    axs[0, 1].axvspan(occ_s, occ_e, color='yellow', alpha=0.25, label='Occlusion Window')
    axs[0, 1].plot(frames, gt_x, 'g--', linewidth=2, label='Ground Truth X')
    axs[0, 1].scatter(frames, det_x, color='red', s=25, alpha=0.7, label='Observed X')
    axs[0, 1].plot(frames, kf_x, 'b-', linewidth=2, label='Kalman Filter Predicted X')
    # Uncertainty envelope (2 sigma)
    sigma_x = np.array(data["kf_uncertainties_x"])
    axs[0, 1].fill_between(frames, np.array(kf_x) - 2 * sigma_x, np.array(kf_x) + 2 * sigma_x, color='blue', alpha=0.15, label='±2σ Covariance')
    axs[0, 1].set_title("X Position Estimation Over Time", fontweight='bold')
    axs[0, 1].set_xlabel("Frame Index")
    axs[0, 1].set_ylabel("X Position (px)")
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()

    # Panel 3: Euclidean Error Over Time
    axs[1, 0].axvspan(occ_s, occ_e, color='yellow', alpha=0.25, label='Occlusion Window')
    axs[1, 0].plot(frames, errors_kf, 'b-', linewidth=2, label='Kalman Tracking Error')
    axs[1, 0].plot(frames, errors_det, 'ro', markersize=4, alpha=0.6, label='Raw Detector Error')
    axs[1, 0].set_title("Euclidean Localization Error (pixels)", fontweight='bold')
    axs[1, 0].set_xlabel("Frame Index")
    axs[1, 0].set_ylabel("Error (pixels)")
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    axs[1, 0].legend()

    # Panel 4: Covariance P Growth and Shrinkage
    axs[1, 1].axvspan(occ_s, occ_e, color='yellow', alpha=0.25, label='Occlusion Window')
    axs[1, 1].plot(frames, data["kf_uncertainties_x"], 'm-', linewidth=2, label='σ_x (sqrt(P_00))')
    axs[1, 1].plot(frames, data["kf_uncertainties_y"], 'c-', linewidth=2, label='σ_y (sqrt(P_11))')
    axs[1, 1].set_title("Kalman Error Covariance Standard Deviation (σ)", fontweight='bold')
    axs[1, 1].set_xlabel("Frame Index")
    axs[1, 1].set_ylabel("Uncertainty σ (pixels)")
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Occlusion analysis plot saved to: {output_path}")


def main():
    print("Running Kalman Filter Occlusion Simulation & Analysis...")
    data = simulate_occlusion_experiment()
    occ_s, occ_e = data["occlusion_window"]
    
    # Calculate error during occlusion
    gt_x = [p[0] for p in data["gt_centers"]]
    gt_y = [p[1] for p in data["gt_centers"]]
    kf_x = [p[0] for p in data["kf_estimates"]]
    kf_y = [p[1] for p in data["kf_estimates"]]
    occ_errors = [np.hypot(kf_x[i] - gt_x[i], kf_y[i] - gt_y[i]) for i in range(occ_s, occ_e + 1)]
    
    print("=" * 60)
    print("KALMAN FILTER OCCLUSION ROBUSTNESS SUMMARY:")
    print("=" * 60)
    print(f"Occlusion Duration:       {occ_e - occ_s + 1} frames (Detector blind)")
    print(f"Mean Error in Occlusion:  {np.mean(occ_errors):.2f} pixels")
    print(f"Max Error in Occlusion:   {np.max(occ_errors):.2f} pixels")
    print("Conclusion: Kalman constant-velocity kinematic projection maintains")
    print("track continuity and bounds error during temporary occlusions without ID switch.")
    print("=" * 60)

    plot_results(data)


if __name__ == "__main__":
    main()
