# 在 AMD Radeon 上用 OpenCV 5 构建端到端视觉 AI Pipeline

[English](README.md) | 中文

一条完全 GPU 加速、通过 ROCm/HIP 运行在 AMD Radeon GPU 上的视觉 AI pipeline,演示:

- **OpenCV 5.x cv::cuda/HIP** —— GPU 加速的图像预处理
- **MIGraphX FP16** —— YOLO26x 目标检测,GPU 上运行(零拷贝)
- **Qwen3-VL + vLLM / llama.cpp** —— GPU 上的视觉语言模型场景理解

```
视频输入
    │
    ▼
OpenCV 5 (cv::cuda / HIP)
  • warpAffine(letterbox 缩放)
  • cvtColor(BGR→RGB)
  • convertTo(归一化)
    │
    ▼
YOLO26x (MIGraphX FP16,零拷贝)
  • 300 个检测框 × [x1,y1,x2,y2,score,class]
    │
    ▼
NMS (cv::cuda::nms,GPU) + ROI 裁剪(JPEG 路径在 CPU / llamacpp-ipc 路径在 GPU)
    │
    ▼
Qwen3-VL (vLLM 或 llama.cpp,ROCm/HIP)
  • 为每个 ROI 生成场景/物体描述
    │
    ▼
叠加绘制 + 输出视频
```

## 测试硬件

| | Radeon Pro W7900 | Radeon RX (R9700) |
|---|---|---|
| GPU 架构 | gfx1100 (RDNA3) | gfx1201 (RDNA4) |
| 显存 | 48 GB | 32 GB |
| ROCm | 7.2.1 | 7.2.1 |
| 容器 | `rocm/vllm-dev:rocm7.2.1_navi` | `rocm/vllm-dev:rocm7.2.1_navi` |

## 前置条件

- 宿主机安装 ROCm 7.2.x
- Docker 开启 GPU 直通(`--device=/dev/kfd --device=/dev/dri`)
- 容器:`rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0`
- 模型:
  - YOLO26x ONNX 模型,位于 `/home/yolo26x.onnx`(213 MB,输入 [1,3,640,640],输出 [1,300,6])
  - VLM 权重,取决于后端:
    - **vLLM**:Qwen3-VL-8B-Instruct(HF 格式),位于 `/models/Qwen3-VL-8B-Instruct`
    - **llama.cpp**:GGUF 权重 + mmproj,位于 `/models/`(如 `Qwen3-VL-8B-Instruct-Q8_0.gguf` + `mmproj-F16.gguf`;见步骤 5b)

## 安装配置

### 步骤 1:安装系统依赖

```bash
apt-get update && apt-get install -y ffmpeg libavcodec-dev libavformat-dev \
    libavutil-dev libswscale-dev pkg-config libpng-dev libjpeg-dev
pip install onnxruntime
```

### 步骤 2:克隆带 HIP 支持的 OpenCV 5.x

`5.x-hip` 分支把 Jeff Daily 的 HIP core 补丁(PR [#29285](https://github.com/opencv/opencv/pull/29285))cherry-pick 到了 OpenCV 5.x 上,并附带 5.x 兼容性修复、`cv::cuda::nms`,以及(contrib 侧)零拷贝 VLM 路径所需的 `align_corners` resize 选项。无需手动打补丁。

`opencv_contrib` 这边 demo 用 **`5.x-hip-zerocopy`** 分支 —— 它就是 `5.x-hip` 加上 `align_corners` commit([opencv_contrib#4181](https://github.com/opencv/opencv_contrib/pull/4181));`align_corners` 默认关闭,所以除了零拷贝 `llamacpp-ipc` 路径外,行为与 `5.x-hip` 完全一致。(在上游这是两个独立 PR:[#4178](https://github.com/opencv/opencv_contrib/pull/4178) HIP 移植 + `cv::cuda::nms`,以及 [#4181](https://github.com/opencv/opencv_contrib/pull/4181) align_corners。)

```bash
cd /home
git clone -b 5.x-hip https://github.com/zhangnju/opencv.git
git clone -b 5.x-hip-zerocopy https://github.com/zhangnju/opencv_contrib.git
```

<details>
<summary>5.x-hip 分支打了哪些补丁(供参考)</summary>

上游 `moat-port` 分支面向 OpenCV 4.x。`5.x-hip` 分支在其之上应用了以下修复:

**a. 修复命名空间重复** —— HIP 编译器在 `namespace cv {}` 内会把 `cv::cuda::` 解析成 `cv::cv::cuda::`:

文件:`modules/cudev/include/opencv2/cudev/util/{vec_math,vec_traits,detail/type_traits}.hpp`

```diff
- using cv::cuda::device::compat::double4;
+ using ::cv::cuda::device::compat::double4;
```

**b. 保护 DataType\<uint\> 重定义** —— OpenCV 5.x core 已经定义了 `DataType<unsigned>`:

文件:`modules/cudev/include/opencv2/cudev/util/vec_traits.hpp`

```cpp
#include "opencv2/core/version.hpp"
// ...
#if !defined(CV_VERSION_MAJOR) || CV_VERSION_MAJOR < 5
template <> class DataType<uint> { ... };
#endif
```

**c. 为 64 位整数添加 VecTraits/MakeVec** —— OpenCV 5.x 在 GPU convertTo 里用了 `long`/`unsigned long`:

文件:`modules/cudev/include/opencv2/cudev/util/vec_traits.hpp`

```cpp
// 在 CV_CUDEV_VEC_TRAITS_INST(double) 之后
template <> struct VecTraits<long> { typedef long elem_type; enum {cn=1}; ... };
template <> struct VecTraits<unsigned long> { ... };

// 在 MakeVec<bool, 4> 之后
template<> struct MakeVec<long, 1> { typedef long type; };
template<> struct MakeVec<long, 2> { typedef long type; };
template<> struct MakeVec<long, 3> { typedef long type; };
template<> struct MakeVec<long, 4> { typedef long type; };
// unsigned long 同理
```

**d. 为 `long`/`unsigned long` 添加 saturate_cast 基础模板**:

文件:`modules/cudev/include/opencv2/cudev/util/saturate_cast.hpp`

```cpp
// 在 saturate_cast(double v) 之后
template <typename T> __device__ __forceinline__ T saturate_cast(long v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(unsigned long v) { return T(v); }
```

**e. 修复 5.x 的 API 迁移**:

文件:`modules/cudawarping/src/warp.cpp` —— 添加 `#include "opencv2/geometry/2d.hpp"`(5.x 中 invertAffineTransform 移到了 geometry 模块)

**f. 移除重复的 stereo 模块**(5.x 中已并入 opencv core)

**g. 在移植之上新增的 GPU 原语 / 选项(本项目的 contrib 工作):**

- `cv::cuda::nms` —— GPU 非极大值抑制(类别感知的 IoU 位掩码 kernel),支持 GPU 常驻的检测后处理。在 `5.x-hip` / [opencv_contrib#4178](https://github.com/opencv/opencv_contrib/pull/4178)。
- 修复了会导致 5.x-hip 全新构建失败的重复 `long`/`ulong` `VecTraits`/`MakeVec`/`saturate_cast` 定义,以及缺失的 `CV_32U` `#endif`。在 `5.x-hip` / #4178。
- 为 `cv::cuda::resize` 增加 `align_corners` 选项(截断型 linear kernel,与 PyTorch/参考 bilinear 逐位一致)—— 零拷贝 `llamacpp-ipc` VLM 路径所需。独立 PR [opencv_contrib#4181](https://github.com/opencv/opencv_contrib/pull/4181);已并入上文使用的 `5.x-hip-zerocopy` 分支。

> Core 侧(opencv 主仓库,[#29527](https://github.com/opencv/opencv/pull/29527)):5.x HIP 兼容性修复和 `GpuMat.fromDevicePointer`(对外部 GPU 内存做零拷贝包装)位于步骤 2 克隆的 `zhangnju/opencv` `5.x-hip` 分支。

</details>

### 步骤 3:编译带 HIP 的 OpenCV 5.x

把 `gfx1100` 换成你的 GPU 架构(RDNA4 用 `gfx1201`,以此类推):

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

编译出的模块包括:`core cudaarithm cudabgsegm cudacodec cudafilters cudaimgproc cudawarping cudev dnn python3` 等。

### 步骤 4:验证 OpenCV HIP

```bash
PYTHONPATH=/opt/opencv5/lib/python3.12/site-packages python3 -c "
import cv2; print(cv2.__version__)
print('GPU devices:', cv2.cuda.getCudaEnabledDeviceCount())
"
```

预期输出:
```
5.1.0-dev
GPU devices: 1
```

### 步骤 5a:启动 vLLM 服务(默认后端)

```bash
vllm serve /models/Qwen3-VL-8B-Instruct \
    --port 8198 --gpu-memory-utilization 0.8 \
    --max-model-len 4096 --tensor-parallel-size 1 \
    --trust-remote-code --dtype auto
```

等待约 2 分钟加载模型。验证:

```bash
curl http://localhost:8198/v1/models
```

### 步骤 5b:启动 llama.cpp 服务(备选后端)

llama.cpp 通过 ROCm/HIP 原生支持 AMD RDNA3/RDNA4 GPU,是 vLLM 的轻量替代 —— 显存开销更低、无需 Python 运行时,在单 GPU 上跑完整 pipeline 时很有用。

**用 ROCm 编译 llama.cpp:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# RDNA3(gfx1100,如 W7900 / RX 7900 XTX)
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
# RDNA4(gfx1201,如 RX 9700)
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j$(nproc)
sudo cmake --install build
```

或使用预构建的 Docker 镜像(已含 ROCm 支持):

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-rocm
```

**下载一个 vision GGUF 模型:**

```bash
# Qwen3-VL 8B Q8_0 —— RDNA3 和 RDNA4 都可用
huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF \
    Qwen3-VL-8B-Instruct-Q8_0.gguf \
    mmproj-F16.gguf \
    --local-dir /models
```

**启动服务:**

```bash
bash start_llamacpp.sh \
    /models/Qwen3-VL-8B-Instruct-Q8_0.gguf \
    /models/mmproj-F16.gguf \
    8199
```

验证:

```bash
curl http://localhost:8199/v1/models
```

> **多 GPU 提示:** `llama-server` 不认 `HIP_VISIBLE_DEVICES` —— 它总是枚举所有卡并默认用 device 0。要指定某张 GPU,请用它自带的参数:先 `llama-server --list-devices` 看 `ROCm0..N` 编号,再加 `--device ROCmN`。注意 llama.cpp 的 `ROCmN` 顺序可能和 `rocm-smi` 的 `GPU[N]` 以及 pipeline 的 `HIP_VISIBLE_DEVICES` 编号不同,所以要把 pipeline 和 VLM 服务放到*同一张*物理卡上,需先确认映射关系(例如在各进程加载时观察 `rocm-smi --showmemuse`)。

## 示例视频

从 [Pexels](https://www.pexels.com/) 下载免费测试视频(CC0 许可,无需注册):

```bash
cd /home

# 街景 —— 行人 + 过往车辆(1080p,25fps)
wget -O sidewalk.mp4 "https://videos.pexels.com/video-files/854100/854100-hd_1920_1080_25fps.mp4"

# 十字路口 —— 过马路的人 + 车辆(1080p,25fps)
wget -O crossroad.mp4 "https://videos.pexels.com/video-files/853743/853743-hd_1920_1080_25fps.mp4"

# 繁忙街道 —— 密集人流(1080p,30fps)
wget -O street.mp4 "https://videos.pexels.com/video-files/3552510/3552510-hd_1920_1080_30fps.mp4"
```

或用经典的 YOLO 测试图生成一段测试视频:

```bash
wget -O zidane.jpg "https://ultralytics.com/images/zidane.jpg"
ffmpeg -loop 1 -i zidane.jpg -t 2 -pix_fmt yuv420p -r 30 zidane_60f.mp4
```

## 运行 Pipeline

```bash
cd /home/ocv_pipeline_demo
PYTHONPATH=/opt/opencv5/lib/python3.12/site-packages:/opt/rocm/lib \
    python3 pipeline.py --input /home/sidewalk.mp4 --output output.mp4
```

### 命令行选项

| 参数 | 默认值 | 说明 |
|------|---------|-------------|
| `--input`, `-i` | (必填) | 视频文件、RTSP URL 或摄像头编号 |
| `--output`, `-o` | `output.mp4` | 输出视频路径 |
| `--no-vlm` | 关 | 完全跳过 VLM 阶段 |
| `--vlm-backend` | (来自 config) | VLM 后端:`vllm`、`llamacpp` 或 `llamacpp-ipc`(通过 HIP IPC 的零拷贝 GPU 图像输入;缺省回落到 `config.VLM_BACKEND`,当前为 `vllm`) |
| `--vlm-url` | (来自 config) | 覆盖 VLM 服务的 base URL |
| `--vlm-model` | (来自 config) | 覆盖 VLM 模型名(llama.cpp 用 `auto` 自动检测) |
| `--max-frames` | 0(全部) | 限制处理帧数 |
| `--vlm-interval` | 30 | 每 N 帧运行一次 VLM |
| `--device` | 0 | GPU 设备编号 |
| `--display` | 关 | 显示 cv2.imshow 窗口(需 X11) |

**示例:**

```bash
# 默认:vLLM 后端
python3 pipeline.py --input sidewalk.mp4 --output out.mp4

# llama.cpp 后端(服务已在 8199 端口启动)
python3 pipeline.py --input sidewalk.mp4 --output out.mp4 \
    --vlm-backend llamacpp

# llama.cpp,自定义 URL 和显式模型名
python3 pipeline.py --input sidewalk.mp4 --output out.mp4 \
    --vlm-backend llamacpp \
    --vlm-url http://localhost:8199/v1 \
    --vlm-model Qwen3-VL-8B-Instruct-Q8_0.gguf
```

### 环境变量

| 变量 | 默认值 | 说明 |
|----------|---------|-------------|
| `HIP_VISIBLE_DEVICES` | 全部 | 为 pipeline 选择 GPU(如 `3` 表示第 4 张)。注意:`llama-server` 不认此变量 —— 改用它的 `--device ROCmN` 参数(见步骤 5b)。 |
| `YOLO_BACKEND` | `auto` | 强制 `migraphx` 或 `ort` |

## 演示

### sidewalk.mp4 —— 行人 + USPS 邮政卡车

| 输入 | 输出(YOLO26x + Qwen3-VL) |
|:-----:|:---------------------------:|
| ![input](assets/input_sidewalk.gif) | ![output](assets/output_sidewalk_vlm.gif) |

### crossroad.mp4 —— 过马路的人 + 车辆

| 输入 | 输出(YOLO26x + Qwen3-VL) |
|:-----:|:---------------------------:|
| ![input](assets/input_crossroad.gif) | ![output](assets/output_crossroad_vlm.gif) |

### street.mp4 —— 密集人流

| 输入 | 输出(YOLO26x + Qwen3-VL) |
|:-----:|:---------------------------:|
| ![input](assets/input_street.gif) | ![output](assets/output_street_vlm.gif) |

## 测试结果

### 测试视频

三段免费的 [Pexels](https://www.pexels.com/) 视频(CC0 许可,1920x1080),覆盖不同街景:

| 视频 | 时长 | FPS | 内容 | 典型检测 |
|-------|----------|-----|---------|--------------------|
| `sidewalk.mp4` | 15.7s(393 帧) | 25 | 城市人行道上的行人 + USPS 卡车 | person, truck, car |
| `crossroad.mp4` | 24.5s(612 帧) | 25 | 过马路的人 + 车辆 | person, car |
| `street.mp4` | 23.8s(712 帧) | 30 | 繁忙街道的密集人流 | 每帧 11-15 个 person |

### 性能:不带 VLM(`--no-vlm`)

在 W7900 和 R9700 上用全部三段样本视频测试。**FPS 为端到端墙钟时间。**

**全 GPU 常驻路径**(rocDecode 硬件解码 → 零拷贝 `cv::cuda` 预处理 → MIGraphX 检测 → `cv::cuda::nms` 后处理 → VA-API 硬件编码)。帧在热路径上始终不离开 GPU,因此视频 I/O 和后处理开销基本消失。

**W7900 (gfx1100):**

| 视频 | 预处理 (GPU/HIP) | 检测 (MIGraphX) | 后处理 (GPU NMS) | **FPS(端到端)** |
|-------|---------------------|---------------------|-------------|---------|
| `sidewalk.mp4` | 0.38 ms | 7.58 ms | 0.26 ms | **78.2** |
| `crossroad.mp4` | 0.34 ms | 7.35 ms | 0.16 ms | **82.4** |
| `street.mp4` | 0.34 ms | 7.49 ms | 0.31 ms | **79.0** |

**R9700 (gfx1201)** —— RDNA4 更快的 MIGraphX 检测(~5.7 ms vs ~7.5 ms)在独占卡上把端到端速率推到接近 100 fps:

| 视频 | 预处理 (GPU/HIP) | 检测 (MIGraphX) | 后处理 (GPU NMS) | **FPS(端到端)** |
|-------|---------------------|---------------------|-------------|---------|
| `sidewalk.mp4` | 0.39 ms | 5.83 ms | 0.31 ms | **92.1** |
| `crossroad.mp4` | 0.35 ms | 5.65 ms | 0.18 ms | **98.7** |
| `street.mp4` | 0.36 ms | 5.76 ms | 0.37 ms | **94.8** |

**CPU 视频 I/O 路径**(`--video-decode cpu --video-encode cpu`,OpenCV FFmpeg + CPU NMS)—— 作对比。差距来自 `VideoCapture.read()` / `VideoWriter.write()` 的开销,以及预处理中的主机上传:

| 视频 | **FPS(端到端)** | 相对 GPU 路径 |
|-------|---------|-------------|
| `sidewalk.mp4` | 65.5 | GPU I/O 快 +21% |
| `crossroad.mp4` | 71.6 | GPU I/O 快 +14% |
| `street.mp4` | 62.9 | GPU I/O 快 +27% |

> GPU 常驻路径把预处理从 ~1.7 ms 降到 ~0.35 ms(在解码后的 surface 上用零拷贝 `GpuMat`,无主机上传),并把 NMS 保持在 GPU 上(~0.2–0.3 ms)。端到端提速的主要来源是把视频解码/编码搬到 VCN 引擎上。

### 性能:带 VLM 的完整 Pipeline(单 GPU)

整条 pipeline(YOLO/MIGraphX + VLM 服务)固定在**单张 GPU** 上,Qwen3-VL-8B,GPU/HIP OpenCV,`--vlm-interval 30`。VLM 阶段是**异步的**(后台线程发射后不管),因此不阻塞主循环 —— pipeline 在同卡并发跑 VLM 推理的同时保持实时 FPS。下表的每帧 VLM 时间反映的是异步**提交**开销(≈0–1 ms),*不是*真实推理延迟(那个见 [VLM 推理延迟](#vlm-推理延迟r9700单-gpu))。

**W7900 (gfx1100),全 GPU 常驻路径**(rocDecode + `cv::cuda` + MIGraphX + `cv::cuda::nms` + VA-API,异步 llama.cpp Q8_0 VLM)。即便 VLM 服务共用这张卡,主循环仍远高于 30 fps 的源帧率:

| 视频 | **FPS** |
|-------|---------|
| `sidewalk.mp4`  | **41.3** |
| `crossroad.mp4` | **50.0** |
| `street.mp4`    | **45.6** |

**R9700 (gfx1201),全 GPU 常驻路径**(rocDecode + `cv::cuda` + MIGraphX + `cv::cuda::nms` + VA-API,异步 vLLM BF16 Qwen3-VL-8B 共卡)。异步 VLM 从不阻塞主循环;GPU 争用把检测抬到 ~11 ms,但 pipeline 仍保持 ~60 fps:

| 视频 | 预处理 (GPU/HIP) | 检测 (MIGraphX) | 后处理 | **FPS** |
|-------|---------------------|---------------------|-------------|---------|
| `sidewalk.mp4`  | 0.64 ms | 11.97 ms | 0.58 ms | **57.7** |
| `crossroad.mp4` | 0.52 ms | 10.98 ms | 1.42 ms | **60.2** |
| `street.mp4`    | 0.54 ms | 11.08 ms | 1.54 ms | **58.8** |

下面按后端 / 按阶段的表早于 GPU 视频 I/O 和 GPU-NMS 的工作(它们用 CPU 视频解码,预处理/后处理时间偏高);它们对于比较 VLM 后端和两张 GPU 仍有参考价值。

**W7900 (gfx1100):**

| 视频 | VLM 后端 | 预处理 (GPU/HIP) | 检测 (MIGraphX) | 后处理 | **FPS** |
|-------|-------------|---------------------|---------------------|-------------|---------|
| `sidewalk.mp4`  | vLLM BF16      | 1.69 ms | 10.13 ms | 0.29 ms | **53.0** |
| `crossroad.mp4` | vLLM BF16      | 1.62 ms |  9.71 ms | 0.69 ms | **57.3** |
| `street.mp4`    | vLLM BF16      | 1.54 ms | 11.80 ms | 1.27 ms | **46.2** |
| `sidewalk.mp4`  | llama.cpp Q8_0 | 2.94 ms | 13.74 ms | 1.10 ms | **41.6** |
| `crossroad.mp4` | llama.cpp Q8_0 | 2.79 ms | 10.26 ms | 1.19 ms | **51.1** |
| `street.mp4`    | llama.cpp Q8_0 | 3.13 ms | 12.19 ms | 1.35 ms | **42.2** |
| `sidewalk.mp4`  | llama.cpp BF16 | 3.19 ms | 14.83 ms | 1.01 ms | **39.5** |
| `crossroad.mp4` | llama.cpp BF16 | 3.29 ms | 11.85 ms | 1.11 ms | **46.6** |
| `street.mp4`    | llama.cpp BF16 | 3.66 ms | 14.14 ms | 1.31 ms | **38.5** |

**R9700 (gfx1201):**

| 视频 | VLM 后端 | 预处理 (GPU/HIP) | 检测 (MIGraphX) | 后处理 | **FPS** |
|-------|-------------|---------------------|---------------------|-------------|---------|
| `sidewalk.mp4`  | vLLM BF16      | 1.34 ms | 11.87 ms | 0.74 ms | **46.8** |
| `crossroad.mp4` | vLLM BF16      | 1.23 ms | 10.48 ms | 1.17 ms | **52.4** |
| `street.mp4`    | vLLM BF16      | 1.33 ms | 10.64 ms | 1.33 ms | **47.1** |
| `sidewalk.mp4`  | llama.cpp Q8_0 | 2.22 ms | 12.13 ms | 1.09 ms | **44.0** |
| `crossroad.mp4` | llama.cpp Q8_0 | 2.19 ms | 11.58 ms | 1.25 ms | **47.6** |
| `street.mp4`    | llama.cpp Q8_0 | 2.54 ms | 12.50 ms | 1.47 ms | **40.8** |
| `sidewalk.mp4`  | llama.cpp BF16 | 2.34 ms | 13.86 ms | 0.92 ms | **40.7** |
| `crossroad.mp4` | llama.cpp BF16 | 2.57 ms | 13.99 ms | 1.20 ms | **42.1** |
| `street.mp4`    | llama.cpp BF16 | 2.91 ms | 16.15 ms | 1.43 ms | **35.2** |

所有配置在两张 GPU、三种后端上都超过 30 fps 源帧率(实时)。后端之间的 FPS 差异主要来自 **GPU 争用** —— 共卡的 VLM 服务与 YOLO/MIGraphX 争抢,越重的 VLM(BF16 > Q8_0;llama.cpp 比 vLLM 稀疏的异步调用让 GPU 更"热")越会拖慢检测。这反映的是真实单 GPU 部署,而非无争用的 YOLO 纯吞吐(见上文 no-VLM 数据)。

> GPU 预处理要求 Python 路径上有启用 HIP 的 OpenCV 5.x 构建(`cv2.cuda.getCudaEnabledDeviceCount()` 必须返回 ≥ 1);否则 pipeline 会自动回落到 CPU 预处理。

### VLM 推理延迟(R9700,单 GPU)

真实的单请求 VLM 延迟(pipeline 的异步统计只报提交时间,**不是**实际推理延迟)。针对 llama.cpp 服务(Qwen3-VL-8B **Q8_0**)、单张 ROI 图、`max_tokens=100` 测得:

| 测量项 | 延迟 | 备注 |
|-------------|---------|-------|
| **客户端端到端**(JPEG 编码 + base64 + HTTP + 推理) | 平均 **582 ms** | 中位 570 ms,范围 502–704 ms(8 次) |
| 服务端总计 | 507–687 ms | 来自 llama.cpp `print_timing` |
| ├─ Prompt eval(含图像编码) | ~100–150 ms | 280–1150 tok/s |
| └─ 生成 | ~400–540 ms | **~58 tok/s**(25–31 tokens) |

> pipeline 每次 VLM 触发发送 `VLM_TOP_K_ROIS=3` 个 ROI。这些请求是**并发**分发的(见下文[并发 ROI 分发](#并发-roi-分发)),所以一个完整 VLM 周期远低于单 ROI 延迟的 3 倍。整个 VLM 阶段又每 30 帧异步运行一次,因此主循环仍保持实时 FPS(此 Q8_0 后端在 R9700 上 40+;见上文完整 pipeline 表)。

### VLM 后端对比(单并发,单 ROI)

同模型,单 ROI,`max_tokens=100`,单请求延迟(客户端端到端):

| GPU | 精度 | 后端 | 平均延迟 | 吞吐 |
|-----|-----------|---------|-------------|------------|
| **R9700 (gfx1201)** | Q8_0 | llama.cpp | **584 ms** | **46.5 tok/s** |
|                     | BF16 | llama.cpp | 1094 ms | 29.1 tok/s |
|                     | BF16 | vLLM | 1279–1322 ms | ~21 tok/s |
| **W7900 (gfx1100)** | Q8_0 | llama.cpp | **531 ms** | ~48 tok/s |
|                     | BF16 | llama.cpp | 721 ms | 35.7 tok/s |
|                     | BF16 | vLLM | 1397 ms | 20.9 tok/s |

> 单并发、低延迟场景(demo 每 30 帧发一次异步 VLM 调用)。这里 llama.cpp 在两张 GPU 上都领先;vLLM 的连续批处理优势在高并发下才体现。各后端和精度的输出质量相当。值得注意的是,W7900 更大的显存(48 GB)和成熟的 RDNA3 llama.cpp 路径让它的 VLM 延迟略低于 R9700,尽管 R9700 在 CV/YOLO 各阶段更快。

### 并发 ROI 分发

每次 VLM 触发会描述得分最高的 `VLM_TOP_K_ROIS`(默认 3)个检测。客户端**并发**发送这些请求(用 `ThreadPoolExecutor` 包裹各 ROI 的 HTTP 调用),让 server 通过连续批处理在其并行 slot 上一起处理,而不是逐个串行。在 GPU-IPC 路径上,每个 ROI 的 GPU 预处理 + IPC 导出仍串行进行(cv::cuda / hipMalloc 在共享 context 上非线程安全);只有 HTTP 请求被并行化 —— 而 ~0.6 秒/ROI 的延迟正是在这一步。

3 个 ROI 的单次触发延迟(R9700,Q8_0,`max_tokens=100`,`llama-server --parallel 3`):

| 分发方式 | 单次触发延迟(3 个 ROI) | 加速 |
|----------|------------------------------|---------|
| 串行(3× 顺序 HTTP) | 2241 ms | 1× |
| **并发(3× 并行 HTTP)** | **1315 ms** | **1.70×** |

> 加速约 1.7×,而非 3×:server 和 pipeline 共用**一张 GPU**,3 个请求的视觉编码器 + 生成计算在设备上仍大体串行 —— 并发主要重叠了 prompt-eval 和调度。它缩短了每次场景描述的刷新,又因为 VLM 阶段是异步的,从不阻塞主循环 FPS。
>
> **`--parallel` 显存权衡:** 真正的并发需要 `--parallel N` ≥ ROI 数(否则多出的会排队),而每个 slot 都要预留自己的 KV cache。从 `--parallel 2 -c 8192` 提到 `--parallel 3 -c 12288` 会占用更多显存 —— 正是你为了给共卡的 MIGraphX 留空间而限制的那部分预算。要在 `--parallel` / `-c` 与共卡 pipeline 所需显存之间权衡;若 VLM 独占一张 GPU,则可自由调高两者。

### 零拷贝 VLM 图像输入(HIP IPC)

通常每个 ROI 都要 JPEG 编码、base64、经 HTTP 发给 VLM 服务,再在服务端解码并做 CPU 预处理(resize/normalize),然后才进视觉编码器。`llamacpp-ipc` 后端把这些全部省掉:ROI 全程留在 GPU,用 `cv::cuda` 在 GPU 上预处理(与 llama.cpp 的 CPU 预处理逐元素一致 —— 已验证),并通过 **HIP IPC handle** 与共卡的 llama-server 共享。只有 64 字节的 handle 过 HTTP;服务端映射同一块显存,直接喂进视觉编码器 —— 无 JPEG、无到主机的像素拷贝、无服务端预处理。

这个后端需要一个用 **[zhangnju/llama.cpp](https://github.com/zhangnju/llama.cpp/tree/vlm_zerocopy) fork 的 `vlm_zerocopy` 分支**构建的 llama-server,它给多模态服务端加了 HIP-IPC / 预处理图像的输入路径(`server-ipc.{cpp,h}` + 一个 `mtmd_bitmap_init_preprocessed` 旁路)。官方 `ggml-org/llama.cpp` 只支持上面的 JPEG `llamacpp` 后端。

```bash
# 编译打过补丁的服务(ROCm/HIP flags 与步骤 5b 的官方构建相同)
git clone -b vlm_zerocopy https://github.com/zhangnju/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release  # W7900 用 gfx1100
cmake --build build --target llama-server -j$(nproc)

# 启动(单 GPU:限制 KV cache,给 pipeline 的 MIGraphX 留出空间)
HIP_VISIBLE_DEVICES=2 ./build/bin/llama-server \
    -m /models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf \
    --mmproj /models/Qwen3-VL-8B-Instruct-GGUF/mmproj-F16.gguf \
    -ngl 99 -c 8192 --parallel 2 --port 8199 --host 0.0.0.0

# 在同一张物理 GPU 上运行 pipeline
python3 pipeline.py --input street.mp4 --vlm-backend llamacpp-ipc \
    --vlm-url http://localhost:8199
```

> pipeline 和 llama-server **必须共用同一张物理 GPU**(HIP IPC 在单个设备内映射显存)。与服务端逐位一致的预处理依赖 `cv::cuda::resize` 的 `align_corners` 选项([opencv_contrib#4181](https://github.com/opencv/opencv_contrib/pull/4181)),如果你在步骤 2 克隆的是 `5.x-hip-zerocopy` contrib 分支,它已经包含在内。

**客户端图像 ingestion 成本** —— 零拷贝真正改变的部分(把像素准备到可发送状态的每图开销)。JPEG 编码+base64 随分辨率增长;IPC 路径始终是一个 64 字节 handle:

| 图像尺寸 | JPEG 编码+base64 | IPC 导出 | 加速 | HTTP 传输量 |
|-----------|-------------------|-----------|---------|--------------|
| 320×240   | 0.17 ms | 0.09 ms | 1.8× | 415× 更小 |
| 640×480   | 0.73 ms | 0.27 ms | 2.7× | 1396× 更小 |
| 1280×720  | 2.02 ms | 0.41 ms | 4.9× | 3235× 更小 |
| 1920×1080 | 4.33 ms | 0.50 ms | **8.6×** | 6383× 更小 |
| 3840×2160 | 13.53 ms | 1.01 ms | **13.4×** | **14220× 更小** |

> JPEG 编码随分辨率近似线性增长(4K 要 13.5 ms),而 IPC 导出基本恒定(始终一个 88 字节 base64 handle)。图越大,优势越大;传输量缩小 3–4 个数量级(4K:1.2 MB base64 → 88 字节)。
>
> **何时有意义:** 在 demo 默认负载(interval-30、单个小 ROI)下,端到端收益可忽略,因为 VLM 自身的视觉编码器前向计算占主导。零拷贝路径在 **4K 大图、每请求多 ROI,或客户端 CPU / 带宽受限的多路部署**下才见效 —— 那些场景下逐图 JPEG 编码和 MB 级传输本会打满 CPU/网卡。

### VLM 场景理解示例

Qwen3-VL 每 30 帧为得分最高的 3 个 ROI 生成自然语言描述。以下是 `sidewalk.mp4` 第 1 帧的示例输出:

| 检测 | VLM 描述 |
|-----------|----------------|
| person 0.96 | *"A blurred image captures the lower half of a person walking on a city sidewalk."* |
| truck 0.94 | *"A white United States Postal Service delivery truck with the slogan 'We Deliver For You' is captured in motion on a city street."* |
| person 0.93 | *"A person in a black shirt and dark jeans walks past a fire hydrant on a city sidewalk."* |

### 跨 GPU 对比

**仅计算时间**的每阶段耗时(不含视频解码/编码 I/O),因此推算出的 FPS 高于上文端到端 `--no-vlm` 数据。用此表比较不同架构的原始 GPU 计算;用端到端表看真实吞吐。

| 阶段 | W7900 (gfx1100) | R9700 (gfx1201) | 后端 |
|-------|-----------------|-----------------|---------|
| GPU 预处理 | **1.93 ms** | 2.45 ms | OpenCV 5.x cv::cuda/HIP |
| YOLO 检测 | **6.82 ms** | 5.34 ms | MIGraphX FP16 零拷贝 |
| 后处理 | 0.34 ms | 0.66 ms | CPU(NMS + 绘制) |
| **总 FPS(仅计算,无 VLM)** | **91.9** | **91.1** | |

### MIGraphX vs PyTorch GPU(W7900)

YOLO26x 推理基准(100 次,输入 [1,3,640,640]):

| 后端 | 延迟 | FPS | 加速比 |
|---------|---------|-----|---------|
| PyTorch GPU FP32 | 18.49 ms | 54 | 1x |
| PyTorch GPU FP16 | 11.52 ms | 87 | 1.6x |
| MIGraphX FP32(零拷贝) | 17.33 ms | 58 | 1.1x |
| MIGraphX FP16(零拷贝) | **6.30 ms** | **159** | **2.9x** |

## 关键技术细节

### MIGraphX 零拷贝推理

标准 MIGraphX `to_gpu()` 会调用 `hipHostRegister`,在某些 GPU 配置下可能失败。本 pipeline 采用 [AMD 博客](https://rocm.blogs.amd.com/artificial-intelligence/gpu-resident-yolo26/README.html) 中的 GPU 常驻零拷贝模式:

```python
# 以 offload_copy=False 编译 —— 不做自动的主机拷贝
model.compile(migraphx.get_target("gpu"), offload_copy=False)

# 通过 PyTorch 在 GPU 上预分配输出张量
output_tensor = torch.empty(output_shape.lens(), dtype=torch.float32, device="cuda")
mgx_output = migraphx.argument_from_pointer(output_shape, output_tensor.data_ptr())

# 直接包装 PyTorch 的 GPU 输入指针 —— 无需 hipHostRegister
mgx_input = migraphx.argument_from_pointer(input_shape, input_tensor.data_ptr())

# 在 HIP stream 上执行 —— 全程 GPU 常驻
model.run_async({...: mgx_input, ...: mgx_output}, stream.cuda_stream, "ihipStream_t")
```

### OpenCV 5.x + HIP 兼容性

把 cv::cuda HIP 后端从 4.x 移植到 5.x 需要修复:

1. **命名空间解析** —— 设备头文件里 `cv::cuda::` → `::cv::cuda::`(HIP 编译器比 NVCC 更严格)
2. **类型重定义** —— 为 5.x 保护 `DataType<uint>`(5.x 已定义 `DataType<unsigned>`)
3. **64 位整数支持** —— 为 `long`/`unsigned long` 添加 `VecTraits`、`MakeVec`、`saturate_cast`
4. **Transform 策略** —— 对 64 位类型强制 `shift=1` 以避开向量化路径(HIP/CUDA 无 `long4`)
5. **API 迁移** —— 5.x 中 `invertAffineTransform` 移到了 `opencv2/geometry/2d.hpp`

## 项目结构

```
ocv_pipeline_demo/
├── pipeline.py          # 主编排器 —— 帧循环
├── preprocess.py        # GPU 预处理(cv::cuda warpAffine/cvtColor/convertTo)
├── detector.py          # YOLO26x 检测(MIGraphX GPU 或 ONNX Runtime 回退)
├── vlm_client.py        # VLM 客户端:VLMClient (vLLM)、LlamaCppVLMClient、AsyncVLMClient
├── postprocess.py       # NMS、边界框叠加、VLM 文本渲染
├── config.py            # 路径、阈值、模型参数、VLM_BACKEND 选择
├── start_vllm.sh        # 启动 vLLM 服务(Qwen3-VL)
├── start_llamacpp.sh    # 启动 llama.cpp 服务(GGUF vision 模型,ROCm/HIP)
├── setup_env.sh         # 一次性环境配置
└── README.md            # 本文件(英文)/ README_CN.md(中文)
```

## 源代码

- **OpenCV 5.x + HIP(core):** [zhangnju/opencv](https://github.com/zhangnju/opencv) 分支 `5.x-hip` —— 5.x HIP 兼容 + `GpuMat.fromDevicePointer`
- **opencv_contrib(HIP):** [zhangnju/opencv_contrib](https://github.com/zhangnju/opencv_contrib) 分支 `5.x-hip-zerocopy` —— `5.x-hip`(HIP 移植 + `cv::cuda::nms` + 5.x 修复)**加上** `align_corners`;demo 用这个分支
- **llama.cpp(零拷贝 VLM 服务):** [zhangnju/llama.cpp](https://github.com/zhangnju/llama.cpp/tree/vlm_zerocopy) 分支 `vlm_zerocopy` —— 仅 `--vlm-backend llamacpp-ipc` 路径需要

**上游 Pull Request:**

- OpenCV core HIP:[opencv/opencv#29527](https://github.com/opencv/opencv/pull/29527)(基础 HIP 移植 PR [#29285](https://github.com/opencv/opencv/pull/29285))
- opencv_contrib HIP 移植 + `cv::cuda::nms`:[opencv/opencv_contrib#4178](https://github.com/opencv/opencv_contrib/pull/4178)(基础 [#4147](https://github.com/opencv/opencv_contrib/pull/4147))
- opencv_contrib `align_corners` resize:[opencv/opencv_contrib#4181](https://github.com/opencv/opencv_contrib/pull/4181)
