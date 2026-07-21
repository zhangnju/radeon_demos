# Building an End-to-End Vision AI Pipeline on AMD Radeon with OpenCV 5

A fully GPU-accelerated vision AI pipeline running on AMD Radeon GPUs via ROCm/HIP, demonstrating:

- **OpenCV 5.x cv::cuda/HIP** — GPU-accelerated image preprocessing
- **MIGraphX FP16** — YOLO26x object detection on GPU (zero-copy)
- **Qwen3-VL + vLLM** — Vision-Language Model scene understanding on GPU

```
Video Input
    │
    ▼
OpenCV 5 (cv::cuda / HIP)
  • warpAffine (letterbox)
  • cvtColor (BGR→RGB)
  • convertTo (normalize)
    │
    ▼
YOLO26x (MIGraphX FP16, zero-copy)
  • 300 detections × [x1,y1,x2,y2,score,class]
    │
    ▼
NMS + ROI Crop (OpenCV)
    │
    ▼
Qwen3-VL (ROCm vLLM)
  • Scene/object description per ROI
    │
    ▼
Overlay + Output Video
```

## Hardware Tested

| | Radeon Pro W7900 | Radeon RX (R9700) |
|---|---|---|
| GPU Arch | gfx1100 (RDNA3) | gfx1201 (RDNA4) |
| VRAM | 48 GB | 32 GB |
| ROCm | 7.2.1 | 7.2.1 |
| Container | `rocm/vllm-dev:rocm7.2.1_navi` | `rocm/vllm-dev:rocm7.2.1_navi` |

## Prerequisites

- ROCm 7.2.x installed on host
- Docker with GPU passthrough (`--device=/dev/kfd --device=/dev/dri`)
- Container: `rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0`
- Models:
  - YOLO26x ONNX model at `/home/yolo26x.onnx` (213 MB, input [1,3,640,640], output [1,300,6])
  - Qwen3-VL-8B-Instruct at `/models/Qwen3-VL-8B-Instruct`

## Setup

### Step 1: Install System Dependencies

```bash
apt-get update && apt-get install -y ffmpeg libavcodec-dev libavformat-dev \
    libavutil-dev libswscale-dev pkg-config libpng-dev libjpeg-dev
pip install onnxruntime
```

### Step 2: Clone OpenCV 5.x with HIP Support

The `5.x-hip` branches contain Jeff Daily's HIP core patches (PR [#29285](https://github.com/opencv/opencv/pull/29285)) cherry-picked onto OpenCV 5.x. The `opencv_contrib` `5.x-hip` branch already includes all 5.x compatibility fixes (namespace doubling, `DataType<uint>` guard, 64-bit integer `VecTraits`/`MakeVec`/`saturate_cast`, geometry API include, and `stereo` module removal) — no manual patches needed.

```bash
cd /home
git clone -b 5.x-hip https://github.com/zhangnju/opencv.git
git clone -b 5.x-hip https://github.com/zhangnju/opencv_contrib.git
```

<details>
<summary>What the 5.x-hip branch patches (for reference)</summary>

The upstream `moat-port` branch targets OpenCV 4.x. The `5.x-hip` branch applies these fixes on top:

**a. Fix namespace doubling** — HIP compiler resolves `cv::cuda::` as `cv::cv::cuda::` inside `namespace cv {}`:

Files: `modules/cudev/include/opencv2/cudev/util/{vec_math,vec_traits,detail/type_traits}.hpp`

```diff
- using cv::cuda::device::compat::double4;
+ using ::cv::cuda::device::compat::double4;
```

**b. Guard DataType\<uint\> redefinition** — OpenCV 5.x core already defines `DataType<unsigned>`:

File: `modules/cudev/include/opencv2/cudev/util/vec_traits.hpp`

```cpp
#include "opencv2/core/version.hpp"
// ...
#if !defined(CV_VERSION_MAJOR) || CV_VERSION_MAJOR < 5
template <> class DataType<uint> { ... };
#endif
```

**c. Add VecTraits/MakeVec for 64-bit integers** — OpenCV 5.x uses `long`/`unsigned long` in GPU convertTo:

File: `modules/cudev/include/opencv2/cudev/util/vec_traits.hpp`

```cpp
// After CV_CUDEV_VEC_TRAITS_INST(double)
template <> struct VecTraits<long> { typedef long elem_type; enum {cn=1}; ... };
template <> struct VecTraits<unsigned long> { ... };

// After MakeVec<bool, 4>
template<> struct MakeVec<long, 1> { typedef long type; };
template<> struct MakeVec<long, 2> { typedef long type; };
template<> struct MakeVec<long, 3> { typedef long type; };
template<> struct MakeVec<long, 4> { typedef long type; };
// same for unsigned long
```

**d. Add saturate_cast base templates** for `long`/`unsigned long`:

File: `modules/cudev/include/opencv2/cudev/util/saturate_cast.hpp`

```cpp
// After saturate_cast(double v)
template <typename T> __device__ __forceinline__ T saturate_cast(long v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(unsigned long v) { return T(v); }
```

**e. Fix 5.x API moves**:

File: `modules/cudawarping/src/warp.cpp` — add `#include "opencv2/geometry/2d.hpp"` (invertAffineTransform moved to geometry module in 5.x)

**f. Remove duplicate stereo module** (moved to opencv core in 5.x)

</details>

### Step 3: Build OpenCV 5.x with HIP

Replace `gfx1100` with your GPU architecture (`gfx1201` for RDNA4, etc.):

```bash
cd /home/opencv && mkdir build && cd build
cmake -DWITH_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DOPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib/modules \
      -DWITH_CUDA=OFF -DWITH_OPENCL=ON -DWITH_FFMPEG=ON \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_hfs=OFF \
      -DBUILD_opencv_bioinspired=OFF \
      -DBUILD_opencv_dnn_superres=OFF \
      -DBUILD_opencv_wechat_qrcode=OFF \
      -DBUILD_opencv_img_hash=OFF \
      -DBUILD_opencv_tracking=OFF \
      -DBUILD_opencv_cudalegacy=OFF \
      -DBUILD_opencv_xobjdetect=OFF \
      -DBUILD_opencv_cudaobjdetect=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/opencv5 \
      -DCMAKE_BUILD_TYPE=Release ..

cmake --build . -j$(nproc)
cmake --install .
```

Built modules include: `core cudaarithm cudabgsegm cudacodec cudafilters cudaimgproc cudawarping cudev dnn python3` and more.

### Step 4: Verify OpenCV HIP

```bash
PYTHONPATH=/opt/opencv5/lib/python3.12/site-packages python3 -c "
import cv2; print(cv2.__version__)
print('GPU devices:', cv2.cuda.getCudaEnabledDeviceCount())
"
```

Expected output:
```
5.1.0-dev
GPU devices: 1
```

### Step 5a: Start vLLM Server (default backend)

```bash
vllm serve /models/Qwen3-VL-8B-Instruct \
    --port 8198 --gpu-memory-utilization 0.8 \
    --max-model-len 4096 --tensor-parallel-size 1 \
    --trust-remote-code --dtype auto
```

Wait ~2 minutes for model loading. Verify with:

```bash
curl http://localhost:8198/v1/models
```

### Step 5b: Start llama.cpp Server (alternative backend)

llama.cpp supports AMD RDNA3/RDNA4 GPUs natively via ROCm/HIP and is a
lightweight alternative to vLLM — lower VRAM overhead, no Python runtime,
useful when running the full pipeline on a single GPU.

**Build llama.cpp with ROCm:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# RDNA3 (gfx1100, e.g. W7900 / RX 7900 XTX)
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
# RDNA4 (gfx1201, e.g. RX 9700)
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j$(nproc)
sudo cmake --install build
```

Or use the pre-built Docker image (already includes ROCm support):

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-rocm
```

**Download a vision GGUF model:**

```bash
# Qwen3-VL 8B Q8_0 — works on RDNA3 and RDNA4
huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF \
    Qwen3-VL-8B-Instruct-Q8_0.gguf \
    mmproj-F16.gguf \
    --local-dir /models
```

**Start the server:**

```bash
bash start_llamacpp.sh \
    /models/Qwen3-VL-8B-Instruct-Q8_0.gguf \
    /models/mmproj-F16.gguf \
    8199
```

Verify with:

```bash
curl http://localhost:8199/v1/models
```

## Sample Videos

Download free test videos from [Pexels](https://www.pexels.com/) (CC0 license, no registration needed):

```bash
cd /home

# Street scene — pedestrians + passing cars (1080p, 25fps)
wget -O sidewalk.mp4 "https://videos.pexels.com/video-files/854100/854100-hd_1920_1080_25fps.mp4"

# Crossroad — people crossing + vehicles (1080p, 25fps)
wget -O crossroad.mp4 "https://videos.pexels.com/video-files/853743/853743-hd_1920_1080_25fps.mp4"

# Busy street — dense pedestrian traffic (1080p, 30fps)
wget -O street.mp4 "https://videos.pexels.com/video-files/3552510/3552510-hd_1920_1080_30fps.mp4"
```

Or generate a test video from the classic YOLO test image:

```bash
wget -O zidane.jpg "https://ultralytics.com/images/zidane.jpg"
ffmpeg -loop 1 -i zidane.jpg -t 2 -pix_fmt yuv420p -r 30 zidane_60f.mp4
```

## Running the Pipeline

```bash
cd /home/ocv_pipeline_demo
PYTHONPATH=/opt/opencv5/lib/python3.12/site-packages:/opt/rocm/lib \
    python3 pipeline.py --input /home/sidewalk.mp4 --output output.mp4
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | (required) | Video file, RTSP URL, or camera index |
| `--output`, `-o` | `output.mp4` | Output video path |
| `--no-vlm` | off | Skip VLM stage entirely |
| `--vlm-backend` | `vllm` | VLM backend: `vllm` or `llamacpp` |
| `--vlm-url` | (from config) | Override VLM server base URL |
| `--vlm-model` | (from config) | Override VLM model name (`auto` for llama.cpp auto-detect) |
| `--max-frames` | 0 (all) | Limit frames to process |
| `--vlm-interval` | 30 | Run VLM every N frames |
| `--device` | 0 | GPU device index |
| `--display` | off | Show cv2.imshow window (needs X11) |

**Examples:**

```bash
# Default: vLLM backend
python3 pipeline.py --input sidewalk.mp4 --output out.mp4

# llama.cpp backend (server already started on port 8199)
python3 pipeline.py --input sidewalk.mp4 --output out.mp4 \
    --vlm-backend llamacpp

# llama.cpp with custom URL and explicit model name
python3 pipeline.py --input sidewalk.mp4 --output out.mp4 \
    --vlm-backend llamacpp \
    --vlm-url http://localhost:8199/v1 \
    --vlm-model Qwen3-VL-8B-Instruct-Q8_0.gguf
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_VISIBLE_DEVICES` | all | Select GPU (e.g., `3` for 4th GPU) |
| `YOLO_BACKEND` | `auto` | Force `migraphx` or `ort` |

## Demo

### sidewalk.mp4 — Pedestrians + USPS Truck

| Input | Output (YOLO26x + Qwen3-VL) |
|:-----:|:---------------------------:|
| ![input](assets/input_sidewalk.gif) | ![output](assets/output_sidewalk_vlm.gif) |

### crossroad.mp4 — People Crossing + Vehicles

| Input | Output (YOLO26x + Qwen3-VL) |
|:-----:|:---------------------------:|
| ![input](assets/input_crossroad.gif) | ![output](assets/output_crossroad_vlm.gif) |

### street.mp4 — Dense Pedestrian Traffic

| Input | Output (YOLO26x + Qwen3-VL) |
|:-----:|:---------------------------:|
| ![input](assets/input_street.gif) | ![output](assets/output_street_vlm.gif) |

## Test Results

### Test Videos

Three free [Pexels](https://www.pexels.com/) videos (CC0 license, 1920x1080) covering different street scenes:

| Video | Duration | FPS | Content | Typical Detections |
|-------|----------|-----|---------|--------------------|
| `sidewalk.mp4` | 15.7s (393 frames) | 25 | Pedestrians + USPS truck on city sidewalk | person, truck, car |
| `crossroad.mp4` | 24.5s (612 frames) | 25 | People crossing road + vehicles | person, car |
| `street.mp4` | 23.8s (712 frames) | 30 | Dense pedestrian traffic on busy street | 11-15 persons per frame |

### Performance: Without VLM (`--no-vlm`)

Tested on W7900 with all three sample videos:

| Video | Preprocess (GPU/HIP) | Detection (MIGraphX) | Postprocess | **FPS** |
|-------|---------------------|---------------------|-------------|---------|
| `sidewalk.mp4` | 2.21 ms | 6.85 ms | 0.43 ms | **60.5** |
| `crossroad.mp4` | 2.18 ms | 6.83 ms | 0.35 ms | **67.6** |
| `street.mp4` | 2.25 ms | 6.85 ms | 0.52 ms | **57.7** |

### Performance: Full Pipeline with VLM

Tested on W7900 with Qwen3-VL via vLLM (`--vlm-interval 30`):

| Video | Preprocess (GPU/HIP) | Detection (MIGraphX) | VLM (per call) | **FPS** |
|-------|---------------------|---------------------|----------------|---------|
| `sidewalk.mp4` | 1.71 ms | 7.50 ms | 5130 ms | **5.0** |
| `crossroad.mp4` | 1.62 ms | 7.44 ms | 3498 ms | **7.4** |
| `street.mp4` | 1.65 ms | 7.57 ms | 4503 ms | **5.9** |

### Performance: Single-GPU (pipeline + VLM on one card)

Entire pipeline (YOLO/MIGraphX + VLM server) pinned to a **single GPU**, `street.mp4`
(712 frames), Qwen3-VL-8B, GPU/HIP OpenCV, `--vlm-interval 30`. Because the VLM stage is
asynchronous (fire-and-forget in a background thread), it does not block the main loop — the
pipeline sustains real-time FPS while VLM inference runs concurrently on the same GPU.

| GPU | VLM backend | Preprocess (GPU/HIP) | Detection (MIGraphX) | Postprocess | **FPS** |
|-----|-------------|---------------------|----------------------|-------------|---------|
| **R9700 (gfx1201)** | llama.cpp Q8_0 | 1.08 ms | 7.36 ms | 1.54 ms | **53.4** |
| **W7900 (gfx1100)** | llama.cpp Q8_0 | 3.05 ms | 15.09 ms | 1.39 ms | **37.9** |
| **W7900 (gfx1100)** | vLLM BF16 | 1.31 ms | 11.12 ms | 0.84 ms | **49.0** |

On R9700, CPU-fallback preprocessing (no HIP OpenCV) gives 50.4 FPS vs 53.4 FPS with
GPU/HIP preprocessing. All configs exceed the 30 fps source rate (real-time).

> The two W7900 rows share the same hardware but differ in how the co-located VLM server
> contends for the GPU. With the llama.cpp server actively running on the same card, YOLO
> detection is throttled to ~15 ms/frame; with vLLM (which idles between the sparse async
> calls) detection runs at ~11 ms/frame. The measured pipeline FPS therefore reflects
> **GPU contention from the co-located VLM server**, not raw YOLO throughput (see the
> no-VLM Cross-GPU Comparison below for contention-free detection numbers).

> GPU preprocessing requires the HIP-enabled OpenCV 5.x build on the Python path
> (`cv2.cuda.getCudaEnabledDeviceCount()` must return ≥ 1); otherwise the pipeline
> automatically falls back to CPU preprocessing.

### VLM Inference Latency (R9700, single GPU)

Real per-request VLM latency (the pipeline's async stats report only submit time, **not** the
actual inference latency). Measured against the llama.cpp server (Qwen3-VL-8B **Q8_0**), single
ROI image, `max_tokens=100`:

| Measurement | Latency | Notes |
|-------------|---------|-------|
| **Client end-to-end** (JPEG encode + base64 + HTTP + inference) | **582 ms** avg | median 570 ms, range 502–704 ms (8 runs) |
| Server-side total | 507–687 ms | from llama.cpp `print_timing` |
| ├─ Prompt eval (incl. image encode) | ~100–150 ms | 280–1150 tok/s |
| └─ Generation | ~400–540 ms | **~58 tok/s** (25–31 tokens) |

> The pipeline sends `VLM_TOP_K_ROIS=3` ROIs **serially** per VLM trigger, so one full VLM
> cycle takes ≈ 3 × 582 ≈ **1.7 s** of wall-clock — but it runs asynchronously every 30 frames,
> so the main loop still holds 53 FPS.

### VLM Backend Comparison (single-concurrency, single ROI)

Same model, single ROI, `max_tokens=100`, per-request latency (client end-to-end):

| GPU | Precision | Backend | Avg latency | Throughput |
|-----|-----------|---------|-------------|------------|
| **R9700 (gfx1201)** | Q8_0 | llama.cpp | **584 ms** | **46.5 tok/s** |
|                     | BF16 | llama.cpp | 1094 ms | 29.1 tok/s |
|                     | BF16 | vLLM | 1279–1322 ms | ~21 tok/s |
| **W7900 (gfx1100)** | Q8_0 | llama.cpp | **531 ms** | ~48 tok/s |
|                     | BF16 | llama.cpp | 721 ms | 35.7 tok/s |
|                     | BF16 | vLLM | 1397 ms | 20.9 tok/s |

> Single-concurrency, low-latency scenario (the demo fires one async VLM call every 30 frames).
> llama.cpp leads here on both GPUs; vLLM's continuous-batching advantage shows at higher
> concurrency. Output quality is equivalent across backends and precisions. Notably, W7900's
> larger VRAM (48 GB) and mature RDNA3 llama.cpp path give it slightly lower VLM latency than
> R9700, even though R9700 is faster on the CV/YOLO pipeline stages.

### VLM Scene Understanding Examples

Qwen3-VL generates natural language descriptions for the top-3 detected ROIs every 30 frames. Example output from `sidewalk.mp4` frame 1:

| Detection | VLM Description |
|-----------|----------------|
| person 0.96 | *"A blurred image captures the lower half of a person walking on a city sidewalk."* |
| truck 0.94 | *"A white United States Postal Service delivery truck with the slogan 'We Deliver For You' is captured in motion on a city street."* |
| person 0.93 | *"A person in a black shirt and dark jeans walks past a fire hydrant on a city sidewalk."* |

### Cross-GPU Comparison

| Stage | W7900 (gfx1100) | R9700 (gfx1201) | Backend |
|-------|-----------------|-----------------|---------|
| GPU Preprocess | **1.93 ms** | 2.45 ms | OpenCV 5.x cv::cuda/HIP |
| YOLO Detection | **6.82 ms** | 5.34 ms | MIGraphX FP16 zero-copy |
| Postprocess | 0.34 ms | 0.66 ms | CPU (NMS + draw) |
| **Total FPS (no VLM)** | **91.9** | **91.1** | |

### MIGraphX vs PyTorch GPU (W7900)

YOLO26x inference benchmark (100 runs, input [1,3,640,640]):

| Backend | Latency | FPS | Speedup |
|---------|---------|-----|---------|
| PyTorch GPU FP32 | 18.49 ms | 54 | 1x |
| PyTorch GPU FP16 | 11.52 ms | 87 | 1.6x |
| MIGraphX FP32 (zero-copy) | 17.33 ms | 58 | 1.1x |
| MIGraphX FP16 (zero-copy) | **6.30 ms** | **159** | **2.9x** |

## Key Technical Details

### MIGraphX Zero-Copy Inference

Standard MIGraphX `to_gpu()` calls `hipHostRegister` which may fail on some GPU configurations. This pipeline uses the GPU-resident zero-copy pattern from the [AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/gpu-resident-yolo26/README.html):

```python
# Compile with offload_copy=False — no automatic host copies
model.compile(migraphx.get_target("gpu"), offload_copy=False)

# Pre-allocate output tensor on GPU via PyTorch
output_tensor = torch.empty(output_shape.lens(), dtype=torch.float32, device="cuda")
mgx_output = migraphx.argument_from_pointer(output_shape, output_tensor.data_ptr())

# Wrap PyTorch GPU input pointer directly — no hipHostRegister
mgx_input = migraphx.argument_from_pointer(input_shape, input_tensor.data_ptr())

# Execute on HIP stream — fully GPU-resident
model.run_async({...: mgx_input, ...: mgx_output}, stream.cuda_stream, "ihipStream_t")
```

### OpenCV 5.x + HIP Compatibility

Porting the cv::cuda HIP backend from 4.x to 5.x required fixing:

1. **Namespace resolution** — `cv::cuda::` → `::cv::cuda::` in device headers (HIP compiler is stricter than NVCC)
2. **Type redefinitions** — `DataType<uint>` guarded for 5.x which already defines `DataType<unsigned>`
3. **64-bit integer support** — Added `VecTraits`, `MakeVec`, `saturate_cast` for `long`/`unsigned long`
4. **Transform policy** — Force `shift=1` for 64-bit types to avoid vectorized path (no `long4` in HIP/CUDA)
5. **API moves** — `invertAffineTransform` moved to `opencv2/geometry/2d.hpp` in 5.x

## Project Structure

```
ocv_pipeline_demo/
├── pipeline.py          # Main orchestrator — frame loop
├── preprocess.py        # GPU preprocessing (cv::cuda warpAffine/cvtColor/convertTo)
├── detector.py          # YOLO26x detection (MIGraphX GPU or ONNX Runtime fallback)
├── vlm_client.py        # VLM clients: VLMClient (vLLM), LlamaCppVLMClient, AsyncVLMClient
├── postprocess.py       # NMS, bounding box overlay, VLM text rendering
├── config.py            # Paths, thresholds, model parameters, VLM_BACKEND selection
├── start_vllm.sh        # Launch vLLM server (Qwen3-VL)
├── start_llamacpp.sh    # Launch llama.cpp server (GGUF vision models, ROCm/HIP)
├── setup_env.sh         # One-time environment setup
└── README.md            # This file
```

## Source Code

- OpenCV 5.x + HIP: [zhangnju/opencv](https://github.com/zhangnju/opencv) branch `5.x-hip`
- opencv_contrib HIP: [zhangnju/opencv_contrib](https://github.com/zhangnju/opencv_contrib) branch `5.x-hip` (based on jeffdaily/opencv_contrib `moat-port` + 5.x compat patches)
- HIP core PR: [opencv/opencv#29285](https://github.com/opencv/opencv/pull/29285)
- HIP cuda modules PR: [opencv/opencv_contrib#4147](https://github.com/opencv/opencv_contrib/pull/4147)
