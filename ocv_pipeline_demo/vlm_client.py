"""Qwen3-VL scene understanding via vLLM OpenAI-compatible API."""

import base64
import threading
import time

import cv2
from openai import OpenAI

import config


class VLMClient:
    def __init__(self, base_url=None, model_name=None):
        self.base_url = base_url or config.VLLM_BASE_URL
        self.model_name = model_name or config.VLLM_MODEL_NAME
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")

    def describe_roi(self, roi_image, prompt=None):
        _, buf = cv2.imencode(".jpg", roi_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt or config.VLM_PROMPT,
                            },
                        ],
                    }
                ],
                max_tokens=config.VLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[VLM error: {e}]"

    def describe_rois(self, frame, detections, top_k=None):
        top_k = top_k or config.VLM_TOP_K_ROIS
        sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)[:top_k]

        results = []
        for det in sorted_dets:
            x1, y1, x2, y2 = [int(v) for v in det[:4]]
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            desc = self.describe_roi(roi)
            results.append((det, desc))

        return results

    def health_check(self):
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False


class AsyncVLMClient:
    """Non-blocking wrapper — runs VLM in a background thread so the main
    pipeline never stalls.  Latest results are cached and reused across frames
    until a new VLM call completes."""

    def __init__(self, base_url=None, model_name=None):
        self._sync = VLMClient(base_url, model_name)
        self._lock = threading.Lock()
        self._latest = []
        self._thread = None
        self._last_latency = 0.0

    def health_check(self):
        return self._sync.health_check()

    @property
    def base_url(self):
        return self._sync.base_url

    @property
    def last_latency(self):
        return self._last_latency

    def submit_rois(self, frame, detections, top_k=None):
        """Fire-and-forget: starts VLM inference in background.
        Skips if a previous call is still running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, args=(frame, detections, top_k), daemon=True,
        )
        self._thread.start()

    def _run(self, frame, detections, top_k):
        t0 = time.time()
        results = self._sync.describe_rois(frame, detections, top_k)
        self._last_latency = time.time() - t0
        with self._lock:
            self._latest = results

    def get_latest(self):
        """Return the most recent VLM results (non-blocking)."""
        with self._lock:
            return list(self._latest)
