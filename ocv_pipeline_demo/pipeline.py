#!/usr/bin/env python3
"""
End-to-End Vision AI Pipeline on AMD Radeon with OpenCV 5 + ROCm/HIP

Pipeline:
  Video Input → GPU Preprocess (cv::cuda/HIP) → YOLO26x (MIGraphX)
  → NMS + ROI Crop → VLM (vLLM or llama.cpp) → Overlay + Output
"""

import argparse
import os
import sys
import time

# Ensure our HIP-enabled OpenCV and MIGraphX are found first
sys.path.insert(0, "/opt/opencv5/lib/python3.12/site-packages")
if "/opt/rocm/lib" not in sys.path:
    sys.path.insert(0, "/opt/rocm/lib")

# Default to a working GPU if not overridden by HIP_VISIBLE_DEVICES
if "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np

import config
from preprocess import preprocess_frame, preprocess_frame_cpu
from detector import YOLODetector
from vlm_client import AsyncVLMClient
from postprocess import draw_detections, draw_scene_panel, draw_stats


def check_gpu():
    """Check GPU availability and print info."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            print(f"[pipeline] AMD GPU via HIP: {count} device(s) available")
            cv2.cuda.setDevice(config.GPU_DEVICE_ID)
            cv2.cuda.printCudaDeviceInfo(config.GPU_DEVICE_ID)
            return True
        else:
            print("[pipeline] No GPU devices found, falling back to CPU preprocessing")
            return False
    except Exception as e:
        print(f"[pipeline] GPU check failed ({e}), falling back to CPU preprocessing")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Vision AI Pipeline on AMD Radeon with OpenCV 5"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input video file, RTSP URL, or camera index (e.g., 0)",
    )
    parser.add_argument(
        "--output", "-o", default="output.mp4",
        help="Output video file path (default: output.mp4)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Display output in window (requires X11)",
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Skip VLM stage for faster processing",
    )
    parser.add_argument(
        "--vlm-backend", default=None, choices=["vllm", "llamacpp"],
        help="VLM inference backend: 'vllm' (default) or 'llamacpp'",
    )
    parser.add_argument(
        "--vlm-url", default=None,
        help="Override VLM server base URL (e.g. http://localhost:8199/v1)",
    )
    parser.add_argument(
        "--vlm-model", default=None,
        help="Override VLM model name (llama.cpp: use 'auto' to detect)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames to process (0 = all)",
    )
    parser.add_argument(
        "--vlm-interval", type=int, default=30,
        help="Run VLM every N frames (default: 30)",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device ID (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config.GPU_DEVICE_ID = args.device

    print("=" * 70)
    print("  End-to-End Vision AI Pipeline on AMD Radeon with OpenCV 5")
    _backend = (args.vlm_backend or config.VLM_BACKEND).lower() if not args.no_vlm else "disabled"
    _vlm_label = {"vllm": "Qwen3-VL + vLLM", "llamacpp": "VLM + llama.cpp", "disabled": "no VLM"}.get(_backend, _backend)
    print(f"  OpenCV + ROCm/HIP | YOLO26x + MIGraphX | {_vlm_label}")
    print("=" * 70)
    print()

    # --- Check GPU ---
    has_gpu = check_gpu()

    # --- Open video input ---
    try:
        src = int(args.input)
    except ValueError:
        src = args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[pipeline] ERROR: cannot open input: {args.input}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[pipeline] Input: {args.input} ({width}x{height} @ {fps_in:.1f}fps, {total_frames} frames)")

    # --- Init video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))
    print(f"[pipeline] Output: {args.output}")

    # --- Init YOLO detector ---
    print()
    detector = YOLODetector(device_id=args.device)

    # --- Init VLM client (async — never blocks the main loop) ---
    vlm = None
    if not args.no_vlm:
        vlm = AsyncVLMClient(
            backend=args.vlm_backend,
            base_url=args.vlm_url,
            model_name=args.vlm_model,
        )
        if vlm.health_check():
            print(f"[pipeline] VLM connected (async, {vlm.backend_name}): {vlm.base_url}")
        else:
            print(f"[pipeline] WARNING: VLM not available at {vlm.base_url}, running without VLM")
            vlm = None

    print()
    print("[pipeline] Starting pipeline...")
    print("-" * 70)

    preprocess_fn = preprocess_frame if has_gpu else preprocess_frame_cpu

    # --- Timing accumulators ---
    t_preprocess = 0.0
    t_detect = 0.0
    t_vlm = 0.0
    t_postprocess = 0.0
    frame_count = 0
    vlm_descriptions = []  # persist VLM results across frames

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if args.max_frames > 0 and frame_count > args.max_frames:
            break

        # --- Stage 1: GPU Preprocessing ---
        t0 = time.time()
        blob, scale, pad_w, pad_h = preprocess_fn(frame)
        t1 = time.time()
        t_preprocess += t1 - t0

        # --- Stage 2: YOLO Detection ---
        detections = detector.detect_and_parse(blob, scale, pad_w, pad_h, frame.shape)
        t2 = time.time()
        t_detect += t2 - t1

        # --- Stage 3: VLM (async, periodic) ---
        if vlm and detections and (frame_count % args.vlm_interval == 0 or frame_count == 1):
            vlm.submit_rois(frame.copy(), detections)
        if vlm:
            vlm_descriptions = vlm.get_latest()
        t3 = time.time()
        t_vlm += t3 - t2

        # --- Stage 4: Overlay ---
        draw_detections(frame, detections, vlm_descriptions if vlm_descriptions else None)
        if vlm_descriptions:
            draw_scene_panel(frame, vlm_descriptions)

        elapsed = time.time() - t_start
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        stats = {
            "FPS": f"{current_fps:.1f}",
            "Frame": f"{frame_count}/{total_frames}",
            "Detections": str(len(detections)),
            "Preprocess": f"{(t1-t0)*1000:.1f}ms",
            "Detect": f"{(t2-t1)*1000:.1f}ms",
        }
        if vlm:
            lat = vlm.last_latency
            stats["VLM"] = f"{lat*1000:.0f}ms" if lat > 0 else "pending"
        draw_stats(frame, stats)

        t4 = time.time()
        t_postprocess += t4 - t3

        writer.write(frame)

        if args.display:
            cv2.imshow("AMD Radeon Vision AI Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Dets: {len(detections)}")

    # --- Cleanup ---
    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start

    print()
    print("=" * 70)
    print(f"  Pipeline Complete")
    print(f"  Frames: {frame_count} | Total: {total_time:.2f}s | Avg FPS: {frame_count/total_time:.1f}")
    print(f"  Avg Preprocess:  {t_preprocess/frame_count*1000:.2f}ms/frame {'(GPU/HIP)' if has_gpu else '(CPU)'}")
    print(f"  Avg Detection:   {t_detect/frame_count*1000:.2f}ms/frame ({detector.backend})")
    if vlm:
        vlm_calls = frame_count // args.vlm_interval + 1
        print(f"  Avg VLM:         {t_vlm/vlm_calls*1000:.0f}ms/call ({vlm.backend_name}, every {args.vlm_interval} frames)")
    print(f"  Avg Postprocess: {t_postprocess/frame_count*1000:.2f}ms/frame")
    print(f"  Output saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
