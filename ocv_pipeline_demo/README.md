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

### Step 5: Start vLLM Server

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

## Running the Pipeline

```bash
cd /home/ocv_pipeline_demo
PYTHONPATH=/opt/opencv5/lib/python3.12/site-packages:/opt/rocm/lib \
    python3 pipeline.py --input video.mp4 --output output.mp4
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | (required) | Video file, RTSP URL, or camera index |
| `--output`, `-o` | `output.mp4` | Output video path |
| `--no-vlm` | off | Skip Qwen3-VL stage |
| `--max-frames` | 0 (all) | Limit frames to process |
| `--vlm-interval` | 30 | Run VLM every N frames |
| `--device` | 0 | GPU device index |
| `--display` | off | Show cv2.imshow window (needs X11) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_VISIBLE_DEVICES` | all | Select GPU (e.g., `3` for 4th GPU) |
| `YOLO_BACKEND` | `auto` | Force `migraphx` or `ort` |

## Test Results

### Test Setup

- Input: Zidane test image (1280x720) repeated as 60-frame video
- YOLO26x detects 4 objects: 2 persons (97%, 96%), 2 ties (87%, 81%)
- VLM describes top-3 detected ROIs per interval

### Performance: Without VLM

| Stage | W7900 (gfx1100) | R9700 (gfx1201) | Backend |
|-------|-----------------|-----------------|---------|
| GPU Preprocess | **1.93 ms** | 2.45 ms | OpenCV 5.x cv::cuda/HIP |
| YOLO Detection | **6.82 ms** | 5.34 ms | MIGraphX FP16 zero-copy |
| Postprocess | 0.34 ms | 0.66 ms | CPU (NMS + draw) |
| **Total FPS** | **91.9** | **91.1** | |

### Performance: Full Pipeline with VLM

| Stage | W7900 (gfx1100) | R9700 (gfx1201) | Backend |
|-------|-----------------|-----------------|---------|
| GPU Preprocess | 4.76 ms | 5.44 ms | OpenCV 5.x cv::cuda/HIP |
| YOLO Detection | 9.63 ms | 5.20 ms | MIGraphX FP16 zero-copy |
| VLM (per call) | 5482 ms | 4002 ms | Qwen3-VL via vLLM |
| Postprocess | 2.08 ms | 2.41 ms | CPU |
| **Total FPS** | **0.7** | **0.9** | (VLM-bound) |

### MIGraphX vs ONNX Runtime

| Backend | Latency | Speedup |
|---------|---------|---------|
| ONNX Runtime CPU | ~80 ms/frame | 1x |
| MIGraphX GPU FP16 | ~5-7 ms/frame | **12-15x** |

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
├── vlm_client.py        # Qwen3-VL via vLLM OpenAI-compatible API
├── postprocess.py       # NMS, bounding box overlay, VLM text rendering
├── config.py            # Paths, thresholds, model parameters
├── start_vllm.sh        # Launch vLLM server
├── setup_env.sh         # One-time environment setup
└── README.md            # This file
```

## Source Code

- OpenCV 5.x + HIP: [zhangnju/opencv](https://github.com/zhangnju/opencv) branch `5.x-hip`
- opencv_contrib HIP: [zhangnju/opencv_contrib](https://github.com/zhangnju/opencv_contrib) branch `5.x-hip` (based on jeffdaily/opencv_contrib `moat-port` + 5.x compat patches)
- HIP core PR: [opencv/opencv#29285](https://github.com/opencv/opencv/pull/29285)
- HIP cuda modules PR: [opencv/opencv_contrib#4147](https://github.com/opencv/opencv_contrib/pull/4147)
