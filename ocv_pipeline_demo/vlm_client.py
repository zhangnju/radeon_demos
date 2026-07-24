"""VLM scene understanding — supports vLLM and llama.cpp backends.

Both backends expose an OpenAI-compatible /v1/chat/completions endpoint,
so the request format is identical. The only differences are:
  - Default base URL and model name
  - llama.cpp "auto" model resolution (queries /v1/models at startup)
"""

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests
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


class LlamaCppIpcVLMClient:
    """llama.cpp backend with zero-copy GPU image input via HIP IPC.

    Instead of JPEG-encoding each ROI and sending pixels over HTTP, this client
    keeps the ROI on the GPU, runs the Qwen3-VL preprocess on the GPU, shares the
    resulting normalized f32 pixels via a HIP IPC handle, and posts only the
    64-byte handle (+ dims) to the server's /completions endpoint. The server
    maps the same VRAM and feeds the vision encoder directly — no JPEG, no host
    round-trip, no server-side CPU preprocess.

    Requires a llama-server built with the preprocessed-IPC support and running
    on the SAME physical GPU as the pipeline.
    """

    def __init__(self, base_url=None, model_name=None):
        # server-common uses /completions (not /v1); base_url points at the root
        self.base_url = (base_url or config.LLAMACPP_BASE_URL).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model_name = model_name or config.LLAMACPP_MODEL_NAME
        self._marker = None

    def _media_marker(self):
        if self._marker is None:
            try:
                r = requests.get(f"{self.base_url}/props", timeout=10)
                self._marker = r.json().get("media_marker", "<__media__>")
            except Exception:
                self._marker = "<__media__>"
        return self._marker

    def _prep_roi_gpu(self, roi_rgb_gpu):
        """GPU preprocess + IPC export for one ROI. Returns an IpcBuffer + dims.
        Kept OFF the thread pool: cv::cuda / hipMalloc on a shared GPU context are
        not safe to call concurrently from multiple threads."""
        import gpu_ipc
        nx, ny, pix = gpu_ipc.preprocess_roi_gpu(roi_rgb_gpu)
        buf = gpu_ipc.IpcBuffer(pix)  # dedicated VRAM + IPC handle; keep alive over the request
        return buf, nx, ny

    def _post_ipc(self, buf, nx, ny, prompt=None):
        """HTTP request for one already-prepared IPC buffer. Thread-safe (pure
        network I/O), so several of these can run concurrently."""
        entry = {
            "ipc_handle": base64.b64encode(buf.handle).decode("utf-8"),
            "width": nx, "height": ny, "preprocessed": True,
        }
        payload = {
            "prompt": {
                "prompt_string": self._media_marker() + (prompt or config.VLM_PROMPT),
                "multimodal_data": [entry],
            },
            "n_predict": config.VLM_MAX_TOKENS,
            "temperature": 0.0,
            "cache_prompt": False,
        }
        try:
            resp = requests.post(f"{self.base_url}/completions", json=payload, timeout=120)
            if resp.status_code != 200:
                return f"[VLM error: HTTP {resp.status_code} {resp.text[:200]}]"
            return resp.json().get("content", "").strip()
        except Exception as e:
            return f"[VLM error: {e}]"

    def describe_roi_gpu(self, roi_rgb_gpu, prompt=None):
        """roi_rgb_gpu: torch.Tensor (H,W,3) uint8 CUDA, RGB. Single-ROI helper."""
        buf, nx, ny = self._prep_roi_gpu(roi_rgb_gpu)
        try:
            return self._post_ipc(buf, nx, ny, prompt)
        finally:
            buf.free()

    def describe_rois_gpu(self, rgb_gpu, detections, top_k=None):
        """rgb_gpu: full-frame torch.Tensor (H,W,3) uint8 CUDA, RGB.
        GPU preprocess + IPC export run serially (GPU context not thread-safe);
        the HTTP requests are then fired concurrently so the server can batch
        them across its parallel slots."""
        top_k = top_k or config.VLM_TOP_K_ROIS
        H, W = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
        sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)[:top_k]

        prepared = []  # (det, buf, nx, ny)
        for det in sorted_dets:
            x1, y1, x2, y2 = [int(v) for v in det[:4]]
            x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
            roi = rgb_gpu[y1:y2, x1:x2, :].contiguous()
            if roi.numel() == 0:
                continue
            buf, nx, ny = self._prep_roi_gpu(roi)
            prepared.append((det, buf, nx, ny))

        try:
            if len(prepared) <= 1:
                return [(det, self._post_ipc(buf, nx, ny)) for det, buf, nx, ny in prepared]
            with ThreadPoolExecutor(max_workers=len(prepared)) as ex:
                descs = list(ex.map(lambda p: self._post_ipc(p[1], p[2], p[3]), prepared))
            return [(prepared[i][0], descs[i]) for i in range(len(prepared))]
        finally:
            for _, buf, _, _ in prepared:
                buf.free()

    def health_check(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False


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
    # crop first (cheap CPU slice), then run the JPEG-encode + HTTP requests
    # concurrently so the server can batch them across its parallel slots
    items = []  # (det, roi)
    for det in sorted_dets:
        x1, y1, x2, y2 = [int(v) for v in det[:4]]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        items.append((det, roi))
    if len(items) <= 1:
        return [(det, vlm_client.describe_roi(roi)) for det, roi in items]
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        descs = list(ex.map(lambda it: vlm_client.describe_roi(it[1]), items))
    return [(items[i][0], descs[i]) for i in range(len(items))]


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
    if backend in ("llamacpp-ipc", "llamacpp_ipc", "ipc"):
        return LlamaCppIpcVLMClient(base_url=base_url, model_name=model_name)
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

    @property
    def is_gpu_ipc(self):
        return isinstance(self._sync, LlamaCppIpcVLMClient)

    def submit_rois(self, frame, detections, top_k=None):
        """Fire-and-forget: starts VLM inference in background.
        Skips if a previous call is still running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, args=(frame, detections, top_k), daemon=True,
        )
        self._thread.start()

    def submit_rois_gpu(self, rgb_gpu, detections, top_k=None):
        """GPU zero-copy variant: rgb_gpu is a full-frame CUDA tensor (H,W,3)
        uint8 RGB. ROIs are cropped and preprocessed on the GPU. The tensor must
        stay valid for the duration of the async call, so the caller passes a
        clone (see pipeline)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_gpu, args=(rgb_gpu, detections, top_k), daemon=True,
        )
        self._thread.start()

    def _run(self, frame, detections, top_k):
        t0 = time.time()
        results = self._sync.describe_rois(frame, detections, top_k)
        self._last_latency = time.time() - t0
        with self._lock:
            self._latest = results

    def _run_gpu(self, rgb_gpu, detections, top_k):
        t0 = time.time()
        results = self._sync.describe_rois_gpu(rgb_gpu, detections, top_k)
        self._last_latency = time.time() - t0
        with self._lock:
            self._latest = results

    def get_latest(self):
        """Return the most recent VLM results (non-blocking)."""
        with self._lock:
            return list(self._latest)
