"""GPU-accelerated preprocessing using cv::cuda (HIP backend on AMD GPUs)."""

import cv2
import numpy as np

import config


def letterbox_gpu(gpu_frame, target_size):
    """Resize with aspect ratio preservation and padding on GPU using warpAffine.

    Uses a single warpAffine call to combine resize + translate (padding) into one
    GPU kernel — more efficient than separate resize + copyMakeBorder.

    Returns (gpu_padded, scale, pad_w, pad_h).
    """
    h, w = gpu_frame.size()[::-1]  # GpuMat.size() returns (cols, rows)
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w, pad_h = (tw - new_w) // 2, (th - new_h) // 2

    # Affine matrix: scale + translate to center
    M = np.array([
        [scale, 0, pad_w],
        [0, scale, pad_h],
    ], dtype=np.float64)

    gpu_padded = cv2.cuda.warpAffine(
        gpu_frame, M, (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )

    return gpu_padded, scale, pad_w, pad_h


def preprocess_frame(frame, stream=None):
    """Full GPU preprocessing pipeline for YOLO input.

    Args:
        frame: BGR numpy array (H, W, 3) uint8

    Returns:
        blob: numpy array (1, 3, 640, 640) float32, ready for MIGraphX
        scale, pad_w, pad_h: letterbox parameters for coordinate mapping
    """
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    gpu_padded, scale, pad_w, pad_h = letterbox_gpu(gpu_frame, config.INPUT_SIZE)

    gpu_rgb = cv2.cuda.cvtColor(gpu_padded, cv2.COLOR_BGR2RGB)

    gpu_float = cv2.cuda_GpuMat(gpu_rgb.size(), cv2.CV_32FC3)
    gpu_rgb.convertTo(cv2.CV_32FC3, gpu_float, alpha=1.0 / 255.0)

    blob_hwc = gpu_float.download()
    blob = blob_hwc.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    return blob.astype(np.float32), scale, pad_w, pad_h


def preprocess_frame_cpu(frame):
    """CPU fallback preprocessing (for comparison/debugging)."""
    h, w = frame.shape[:2]
    tw, th = config.INPUT_SIZE
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w, pad_h = (tw - new_w) // 2, (th - new_h) // 2

    M = np.array([
        [scale, 0, pad_w],
        [0, scale, pad_h],
    ], dtype=np.float64)

    padded = cv2.warpAffine(
        frame, M, (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    return blob, scale, pad_w, pad_h
