"""VLM scene understanding — supports vLLM and llama.cpp backends.

Both backends expose an OpenAI-compatible /v1/chat/completions endpoint,
so the request format is identical. The only differences are:
  - Default base URL and model name
  - llama.cpp "auto" model resolution (queries /v1/models at startup)
"""

import base64
import threading
import time

import cv2
from openai import OpenAI

import config


# ---------------------------------------------------------------------------
# Synchronous clients
# ---------------------------------------------------------------------------

class VLMClient:
    """vLLM backend (default)."""

    def __init__(self, base_url=None, model_name=None):
        self.base_url = base_url or config.VLLM_BASE_URL
        self.model_name = model_name or config.VLLM_MODEL_NAME
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")

    def describe_roi(self, roi_image, prompt=None):
        return _chat_with_image(self.client, self.model_name, roi_image, prompt)

    def describe_rois(self, frame, detections, top_k=None):
        return _describe_rois(self, frame, detections, top_k)

    def health_check(self):
        return _health_check(self.client)


class LlamaCppVLMClient:
    """llama.cpp server backend.

    llama.cpp's server (built with ROCm/HIP support) exposes the same
    OpenAI-compatible API as vLLM, so the chat payload is identical.
    The only quirk is that the model name in llama.cpp is the filename of
    the GGUF, which may not be known in advance — setting LLAMACPP_MODEL_NAME
    to "auto" causes this client to query /v1/models at startup.
    """

    def __init__(self, base_url=None, model_name=None):
        self.base_url = base_url or config.LLAMACPP_BASE_URL
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        self.model_name = self._resolve_model(model_name or config.LLAMACPP_MODEL_NAME)

    def _resolve_model(self, name):
        if name != "auto":
            return name
        try:
            models = self.client.models.list()
            if models.data:
                resolved = models.data[0].id
                print(f"[llama.cpp] auto-resolved model: {resolved}")
                return resolved
        except Exception as e:
            print(f"[llama.cpp] WARNING: could not resolve model name ({e}), using 'default'")
        return "default"

    def describe_roi(self, roi_image, prompt=None):
        return _chat_with_image(self.client, self.model_name, roi_image, prompt)

    def describe_rois(self, frame, detections, top_k=None):
        return _describe_rois(self, frame, detections, top_k)

    def health_check(self):
        return _health_check(self.client)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chat_with_image(client, model_name, roi_image, prompt):
    """Send a single ROI image to the chat endpoint and return the reply."""
    _, buf = cv2.imencode(".jpg", roi_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
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


def _describe_rois(vlm_client, frame, detections, top_k):
    top_k = top_k or config.VLM_TOP_K_ROIS
    sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)[:top_k]
    results = []
    for det in sorted_dets:
        x1, y1, x2, y2 = [int(v) for v in det[:4]]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        desc = vlm_client.describe_roi(roi)
        results.append((det, desc))
    return results


def _health_check(client):
    try:
        models = client.models.list()
        return len(models.data) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_vlm_client(backend=None, base_url=None, model_name=None):
    """Return the appropriate synchronous VLM client for the given backend.

    Args:
        backend: "vllm" or "llamacpp" (defaults to config.VLM_BACKEND)
        base_url: override server URL
        model_name: override model name
    """
    backend = (backend or config.VLM_BACKEND).lower()
    if backend == "llamacpp":
        return LlamaCppVLMClient(base_url=base_url, model_name=model_name)
    return VLMClient(base_url=base_url, model_name=model_name)


# ---------------------------------------------------------------------------
# Async wrapper (backend-agnostic)
# ---------------------------------------------------------------------------

class AsyncVLMClient:
    """Non-blocking wrapper around any synchronous VLM client.

    Runs inference in a background thread so the main pipeline loop never
    stalls. The most recent result is cached and reused across frames until
    a new VLM call completes.
    """

    def __init__(self, backend=None, base_url=None, model_name=None):
        self._sync = make_vlm_client(backend=backend, base_url=base_url, model_name=model_name)
        self._lock = threading.Lock()
        self._latest = []
        self._thread = None
        self._last_latency = 0.0

    @property
    def base_url(self):
        return self._sync.base_url

    @property
    def backend_name(self):
        return type(self._sync).__name__

    @property
    def last_latency(self):
        return self._last_latency

    def health_check(self):
        return self._sync.health_check()

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
