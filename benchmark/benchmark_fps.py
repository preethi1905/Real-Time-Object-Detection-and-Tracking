#!/usr/bin/env python3
"""
benchmark_fps.py
Inference speed benchmark: PyTorch (.pt) vs ONNX Runtime (.onnx).
Evaluates latency (ms), throughput (FPS), min, max, and P95 latency across 200 test images.
Exports markdown & JSON summary for documentation and resume bullets.
"""

import os
import time
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def prepare_test_frames(images_dir, num_frames=200, imgsz=640):
    frames = []
    if images_dir and os.path.exists(images_dir):
        files = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        files.sort()
        for f in files[:num_frames]:
            img = cv2.imread(f)
            if img is not None:
                frames.append(img)

    # If fewer than num_frames available, fill with synthetic frames matching KITTI resolution (1242x375)
    while len(frames) < num_frames:
        synthetic = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        frames.append(synthetic)

    return frames[:num_frames]


def benchmark_pytorch(pt_path, frames, device='cpu', imgsz=640):
    from ultralytics import YOLO
    print(f"\n--> Benchmarking PyTorch (.pt): {pt_path} on {device}...")
    model = YOLO(pt_path)

    # Warmup
    print("    Running 15 warmup frames...")
    for i in range(min(15, len(frames))):
        _ = model.predict(source=frames[i], imgsz=imgsz, device=device, verbose=False)

    latencies = []
    print(f"    Benchmarking {len(frames)} frames...")
    for img in tqdm(frames, desc="PyTorch Inference"):
        t0 = time.perf_counter()
        _ = model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return compute_stats("PyTorch (.pt)", latencies, device)


def benchmark_onnxruntime(onnx_path, frames, device='cpu', imgsz=640):
    import onnxruntime as ort
    print(f"\n--> Benchmarking ONNX Runtime (.onnx): {onnx_path} on {device}...")

    providers = ['CPUExecutionProvider']
    if device in ['0', 'cuda', 'gpu'] and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # Preprocessing function for YOLOv8 ONNX
    def preprocess(img):
        h, w = img.shape[:2]
        r = min(imgsz / h, imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        top = (imgsz - nh) // 2
        left = (imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0)

    # Warmup
    print("    Running 15 warmup frames...")
    for i in range(min(15, len(frames))):
        inp = preprocess(frames[i])
        _ = session.run(None, {input_name: inp})

    latencies = []
    print(f"    Benchmarking {len(frames)} frames...")
    for img in tqdm(frames, desc="ONNX Runtime Inference"):
        t0 = time.perf_counter()
        inp = preprocess(img)
        _ = session.run(None, {input_name: inp})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    used_provider = session.get_providers()[0]
    return compute_stats(f"ONNX Runtime ({used_provider})", latencies, device)


def compute_stats(framework_name, latencies_ms, device):
    lat = np.array(latencies_ms)
    avg_latency = float(np.mean(lat))
    median_latency = float(np.median(lat))
    min_latency = float(np.min(lat))
    max_latency = float(np.max(lat))
    p95_latency = float(np.percentile(lat, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "framework": framework_name,
        "device": device,
        "avg_latency_ms": round(avg_latency, 2),
        "median_latency_ms": round(median_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "fps": round(fps, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime FPS.")
    parser.add_argument("--pt", type=str, default="runs/detect/train/weights/best.pt", help="Path to .pt model")
    parser.add_argument("--onnx", type=str, default="runs/detect/train/weights/best.onnx", help="Path to .onnx model")
    parser.add_argument("--images", type=str, default="data/kitti/tracking/training/image_02/0000", help="Images folder")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of benchmark frames")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or '0')")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save summary")
    args = parser.parse_args()

    # Check fallbacks
    pt_path = args.pt if os.path.exists(args.pt) else "yolov8n.pt"
    onnx_path = args.onnx if os.path.exists(args.onnx) else "yolov8n.onnx"

    print("Loading / generating 200 benchmark test frames...")
    frames = prepare_test_frames(args.images, num_frames=args.num_frames, imgsz=args.imgsz)
    print(f"Loaded {len(frames)} benchmark frames.")

    results = []

    # 1. Benchmark PyTorch
    try:
        res_pt = benchmark_pytorch(pt_path, frames, device=args.device, imgsz=args.imgsz)
        results.append(res_pt)
    except Exception as e:
        print(f"PyTorch benchmark error: {e}")

    # 2. Benchmark ONNX Runtime
    try:
        res_onnx = benchmark_onnxruntime(onnx_path, frames, device=args.device, imgsz=args.imgsz)
        results.append(res_onnx)
    except Exception as e:
        print(f"ONNX Runtime benchmark error: {e}")

    # Display comparison table
    print("\n" + "=" * 85)
    print("INFERENCE SPEED BENCHMARK SUMMARY (200 FRAMES, 640x640)")
    print("=" * 85)
    header = f"{'Framework / Runtime':<30} | {'Device':<8} | {'Avg Latency':<12} | {'P95 Latency':<12} | {'Throughput (FPS)':<16}"
    print(header)
    print("-" * 85)
    for r in results:
        line = f"{r['framework']:<30} | {r['device']:<8} | {r['avg_latency_ms']:>6} ms    | {r['p95_latency_ms']:>6} ms    | {r['fps']:>7} FPS"
        print(line)
    print("=" * 85)

    # Save to disk
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "benchmark_fps.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Benchmark results saved to: {summary_path}")


if __name__ == "__main__":
    main()
