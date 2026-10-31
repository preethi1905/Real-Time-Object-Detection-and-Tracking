# KITTI Dataset Guide & Directory Structure

This project uses the official **KITTI Vision Benchmark Suite** for autonomous driving perception.

## 1. Required Downloads from KITTI Website
Register for free at the [KITTI Benchmark Website](http://www.cvlibs.net/datasets/kitti/):

1. **KITTI 2D Object Detection Benchmark**:
   - `data_object_image_2.zip` (12 GB) - Left color camera images.
   - `data_object_label_2.zip` (5 MB) - 2D ground truth bounding boxes.
2. **KITTI Tracking Benchmark**:
   - `data_tracking_image_2.zip` (15 GB) - Left color tracking sequences.
   - `data_tracking_label_2.zip` (9 MB) - Object tracking ground truth with track IDs (`label_02`).

---

## 2. Recommended Directory Organization

Organize the extracted files under `data/kitti/`:

```
data/
└── kitti/
    ├── detection/
    │   └── training/
    │       ├── image_2/          # 7,481 training images (.png)
    │       └── label_2/          # 7,481 label files (.txt)
    └── tracking/
        └── training/
            ├── image_02/         # Tracking sequences (0000, 0001, ..., 0020)
            │   ├── 0000/         # Sequential frames: 000000.png, 000001.png, ...
            │   └── ...
            └── label_02/         # Ground truth tracking labels: 0000.txt, 0001.txt, ...
```

---

## 3. Instant Testing Without Downloading Gigabytes
To test the entire pipeline (offline tracking, MOT evaluation, and ROS 2 launch) immediately without downloading the full dataset, run:

```bash
python scripts/generate_sample_data.py
```

This generates a lightweight, geometrically accurate synthetic sequence in `data/kitti/tracking/training/image_02/0000` with matching ground truth in `label_02/0000.txt`.
