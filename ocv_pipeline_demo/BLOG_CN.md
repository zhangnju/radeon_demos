# 在单张 AMD Radeon GPU 上构建端到端视觉 AI Pipeline

现代计算机视觉应用很少只依赖单个模型。一个真正可用的"视频理解"系统,需要*解码*视频帧、对其*预处理*、*检测*其中的目标,并且越来越多地还要用自然语言*描述*它所看到的内容。传统上,这些阶段各自运行在独立的运行时里——往往还要各自占用一块加速卡。

本文将介绍一个可运行的 demo,它借助 ROCm/HIP 软件栈,把这整条链路**端到端地收敛到一张 AMD Radeon GPU 上**。VCN 引擎上的硬件视频解码、基于 HIP 加速的 OpenCV 预处理、通过 MIGraphX 运行的 YOLO26x 目标检测、用于场景理解的 Qwen3-VL 视觉语言模型,以及硬件视频编码,全部跑在同一张卡上——不需要第二块 GPU,重负载阶段也不必回落到 CPU。从压缩码流输入,到标注后的码流输出,帧数据全程无需离开 GPU。

这个 demo 的核心在于**在单个设备上做到功能完整**:它证明了一条完整的"先检测、再描述"的视觉 pipeline *能够装进并运行在*一张 Radeon GPU 上。这里我们刻意不去突出性能数字;重点是架构本身,以及"所有组件共存于同一块芯片上"这一事实。

![Pipeline 架构图](assets/pipeline_architecture.jpg)

## Pipeline 总览

```
视频输入(文件 · RTSP · 摄像头)
        │
        ▼
视频解码                     rocDecode · VCN
  • H.264 / HEVC 码流 → 显存中的 RGB surface
        │
        ▼
OpenCV 5.x 预处理             cv::cuda / HIP
  • warpAffine(letterbox 缩放)、归一化 —— 在解码 surface 上零拷贝进行
        │
        ▼
YOLO26x 检测                  MIGraphX FP16(零拷贝)
  • 300 个候选框 → [x1, y1, x2, y2, score, class]
        │
        ▼
NMS + ROI 裁剪               CPU · OpenCV(轻量)
        │
        ▼
Qwen3-VL 场景理解            vLLM  |  llama.cpp
  • 为每个 ROI 生成一句自然语言描述(异步)
        │
        ▼
叠加绘制                     CPU · OpenCV(轻量)
        │
        ▼
视频编码                     VA-API · VCN
  • 标注后的帧 → H.264 码流
```

重负载阶段——解码、预处理、检测、描述、编码——都通过 ROCm/HIP 运行在**同一张** AMD Radeon GPU 上。只有两个轻量步骤(NMS/ROI 的簿记和叠加绘制)留在 CPU。

## 逐阶段解析

### 1. 用 rocDecode 做硬件视频解码

在任何 AI 运行之前,压缩视频得先被解码。demo 没有用 CPU 上的 FFmpeg 软解,而是通过 **rocDecode** 及其 `rocPyDecode` Python 绑定,驱动 GPU 上专用的 **VCN**(Video Core Next)引擎。解码器对 H.264/HEVC 码流做解复用,并把解码后的帧**直接产生在显存里**。

关键细节在于帧如何交给流水线的后续部分。rocDecode 把每个解码后的 surface 暴露为一个 **DLPack** capsule,可以零拷贝地包装成 PyTorch GPU 张量:

```python
from pyRocVideoDecode.types import OUT_SURFACE_MEM_DEV_COPIED
decoder = dec.decoder(codec_id, mem_type=OUT_SURFACE_MEM_DEV_COPIED, b_force_zero_latency=True)
# ...
decoder.GetFrameRgb(packet, rgb_format=3)
rgb_gpu = torch.from_dlpack(packet.ext_buf[0])   # (H, W, 3) uint8,在 cuda 上
```

所以流水线拿到的第一个东西,就是一帧已经驻留在 GPU 上的 RGB 图像——压缩字节直接从磁盘进入 VCN 解码器再到显存,全程没碰过 NumPy 数组。如果 rocDecode 不可用,reader 会透明回落到 `cv2.VideoCapture`(CPU FFmpeg)。

### 2. 基于 HIP 的 OpenCV 5.x 预处理 —— 在解码 surface 上零拷贝

经典的 YOLO 预处理——letterbox 缩放到 640×640,并归一化成 `float32` 的 NCHW blob——运行在带 **HIP 后端**构建的 OpenCV 5.x 上,使得 `cv::cuda::warpAffine`、`convertTo` 直接在 Radeon GPU 上执行。

为了让整条链路保持零拷贝,我们需要让 OpenCV 的 `cv::cuda::GpuMat` **直接包装解码器已有的 GPU 内存**,而不是从主机上传。OpenCV 的 C++ 侧本就有一个基于外部内存的构造函数 `GpuMat(rows, cols, type, void* data, step)`,但它没有暴露给 Python。我们给 `5.x-hip` 分支加了一个小补丁,新增 `GpuMat.fromDevicePointer(...)` 绑定,接受一个裸设备地址:

```python
gpu_rgb = cv2.cuda_GpuMat.fromDevicePointer(
    rgb_gpu.data_ptr(), h, w, cv2.CV_8UC3, w * 3)   # 零拷贝,不拥有该内存
gpu_padded, scale, pad_w, pad_h = letterbox_gpu(gpu_rgb, (640, 640))
```

这样一来,rocDecode 的 surface、cv::cuda 预处理、以及检测器的输入,全都指向同一块 GPU 内存——从解码到推理之间,帧从不拷贝回主机。此前那些让 `cv::cuda` 在 HIP 上编译通过的可移植性修复(命名空间解析、64 位向量 traits、`invertAffineTransform` 的 API 迁移)也都在同一个公开的 `5.x-hip` 分支上。

如果 Python 路径上没有启用 HIP 的 OpenCV,pipeline 会检测到这一点(`cv2.cuda.getCudaEnabledDeviceCount()` 返回 0),并回落到 CPU 预处理——demo 依然能跑。

### 3. 通过 MIGraphX 运行 YOLO26x 检测——零拷贝

目标检测使用一个以 **MIGraphX** 编译、FP16 精度的 YOLO26x ONNX 模型。真正巧妙的地方在于数据是如何流动的。一种朴素的集成方式会在每一帧都把张量在主机与设备之间来回拷贝;而标准的 `to_gpu()` 路径甚至会调用 `hipHostRegister`,这在某些配置下会失败。

demo 采用的是**驻留 GPU、零拷贝**的模式:

```python
# 以 offload_copy=False 编译——不做任何自动的主机拷贝
model.compile(migraphx.get_target("gpu"), offload_copy=False)

# 通过 PyTorch 在 GPU 上预分配输出张量
output_tensor = torch.empty(output_shape.lens(), dtype=torch.float32, device="cuda")
mgx_output = migraphx.argument_from_pointer(output_shape, output_tensor.data_ptr())

# 直接包装 PyTorch 的 GPU 输入指针——无需 hipHostRegister
mgx_input = migraphx.argument_from_pointer(input_shape, input_tensor.data_ptr())

# 在当前 HIP stream 上执行——全程驻留 GPU
model.run_async({...: mgx_input, ...: mgx_output}, stream.cuda_stream, "ihipStream_t")
```

输入 blob 本身就是一个 PyTorch CUDA 张量;MIGraphX 直接针对它的裸设备指针运行,并把结果写入同一个 HIP stream 上预分配好的 GPU 输出张量。返回的是 300 个候选框,随后按置信度过滤,再经过 OpenCV 的 NMS。

如果 MIGraphX 或兼容的 GPU 运行时不可用,检测器会回落到 ONNX Runtime(ROCm 执行提供程序,或 CPU)——同样,pipeline 依旧能正常工作。

### 4. 用 Qwen3-VL 做场景理解——可插拔后端

对于每一个触发 VLM 阶段的帧,pipeline 会裁剪出得分最高的 top-K 个检测区域,并把每个 ROI 送入一个 **Qwen3-VL-8B** 视觉语言模型,由它返回一句简短的自然语言描述("一辆白色的美国邮政快递卡车……")。正是这一步,把一个只会画框的检测器,变成了真正能*描述场景*的系统。

关键在于,这个 VLM 是**可插拔后端**。两种方案都暴露相同的、兼容 OpenAI 的 `/v1/chat/completions` 接口,因此请求代码完全一致:

- **vLLM**——在 ROCm 上以 HF 格式部署 Qwen3-VL 模型。
- **llama.cpp**——以 GGUF 格式部署同一个模型,原生运行在 ROCm/HIP 上。占用更轻、无需 Python 运行时,并且会在启动时自动解析模型名。

切换只需改动一个参数:`--vlm-backend vllm` 或 `--vlm-backend llamacpp`。由于 VLM 服务与 pipeline 的其余部分共享**同一张 GPU**,这才是名副其实的单设备部署。

### 5. 异步 VLM——主循环永不阻塞

视觉语言模型远比目标检测更"重",因此如果每一帧都同步内联地跑它,就会拖垮整条 pipeline。demo 用一套**异步、发射后不管(fire-and-forget)**的设计解决了这个问题:

- VLM 仅每 *N* 帧触发一次(可配置,默认 30)。
- 触发时,推理在**后台线程**中运行;主循环立即继续解码、检测、绘制。
- 最近一次的 VLM 描述结果会被缓存,并叠加到后续帧上,直到有更新的结果到来。
- 如果下一次触发时上一次 VLM 调用仍在运行,则直接跳过——不会堆积任务队列。

最终效果是:检测循环按自己的节奏持续运行,与此同时场景描述不断出现并刷新——而这一切都在一张 GPU 上完成。

### 6. 叠加绘制与硬件编码

叠加绘制是一个轻量的 CPU 步骤,用 OpenCV 完成:检测框和类别标签、侧边面板里缓存的场景描述,以及实时统计信息。为此流水线会付出一次帧的设备到主机拷贝——绘制和 VLM 的 JPEG 编码都是 CPU 操作,所以这里是帧唯一一次离开 GPU 的地方。

但写出结果又回到了 GPU。标注后的帧通过管道送给 ffmpeg 的 **`h264_vaapi`** 编码器,它上传到一个 VA-API surface,并在做解码的**同一个 VCN 引擎**上编码——用硬件编码闭合了整个环路。和其它每个阶段一样,这里也有回退:如果 VA-API 不可用,writer 会降级到 `cv2.VideoWriter`(CPU FFmpeg)。

## 为什么"单 GPU"很重要

这个 demo 的看点不是某个跑分,而是**共址(co-location)**。一条横跨传统 CV(OpenCV)、编译型检测模型(MIGraphX)和大型视觉语言模型(Qwen3-VL)的完整视觉 pipeline,正是人们通常认为需要一整机架加速卡才能承载的那类工作负载。而在这里,它运行在一张 AMD Radeon 卡上:

- **一张 GPU,一套 ROCm/HIP 栈。** 硬件解码、预处理、检测、VLM 推理和硬件编码,都通过同一套驱动栈面向同一个设备。
- **帧数据驻留在 GPU 上。** rocDecode 把帧落在显存里,一个零拷贝的 `GpuMat` 把它包装给 cv::cuda 预处理,MIGraphX 就地读取结果——在热路径上直到 CPU 叠加步骤之前,都不经过主机往返。
- **每个阶段都有优雅回退。** 没有 rocDecode?回落 CPU FFmpeg 解码。没有 HIP OpenCV?回落 CPU 预处理。没有 MIGraphX?回落 ONNX Runtime。没有 VA-API?回落 CPU 编码。VLM 服务没起来?就只跑检测。这个 demo 足够健壮,能在各种各样的环境下*启动起来*。
- **可插拔、兼容 OpenAI 的 VLM。** vLLM 或 llama.cpp 任选,代码路径不变。

它已经在 **Radeon Pro W7900**(gfx1100,RDNA3)和 **Radeon RX 9700**(gfx1201,RDNA4)两款卡上跑通,使用的是 ROCm 7.2.x 的 `rocm/vllm-dev` 容器。

## 自己动手运行

需要准备的几个部分:

1. **带 HIP 的 OpenCV 5.x**——从 `5.x-hip` 分支构建(其中包含 `GpuMat.fromDevicePointer` 绑定)并安装到 `/opt/opencv5`。
2. **rocDecode + rocPyDecode**——用于 GPU 硬件解码(可选;可回落到 CPU FFmpeg)。**带 VA-API 的 ffmpeg**——用于 GPU 硬件编码(可选)。
3. **一个 VLM 服务**——要么
   `vllm serve /models/Qwen3-VL-8B-Instruct --port 8198 …`,要么在 8199 端口起一个带 GGUF 模型 + mmproj 的 llama.cpp `llama-server`。
4. **运行 pipeline**:

```bash
cd ocv_pipeline_demo
PYTHONPATH=/opt/opencv5/lib/python3.12/site-packages:/opt/rocm/lib \
    python3 pipeline.py --input sidewalk.mp4 --output out.mp4

# 用 llama.cpp 后端替代 vLLM
python3 pipeline.py --input sidewalk.mp4 --output out.mp4 --vlm-backend llamacpp

# 显式强制 GPU 硬件解码 + 编码
python3 pipeline.py --input sidewalk.mp4 --output out.mp4 \
    --video-decode rocdecode --video-encode vaapi
```

若干参数(`--no-vlm`、`--vlm-interval`、`--vlm-backend`、`--video-decode`、`--video-encode`、`--device`)可以用来调整运行方式——包括一个完全不需要 VLM 服务的"仅检测"模式。

> 多卡提示:pipeline 支持用 `HIP_VISIBLE_DEVICES` 选卡,但 `llama-server` 不认这个变量——要用它自带的 `--device ROCmN` 参数,并核对映射关系,确保 pipeline 与 VLM 落在*同一张*物理 GPU 上。

## 结语

借助 ROCm/HIP、rocDecode、OpenCV 5.x、MIGraphX、一个兼容 OpenAI 的 VLM 服务,以及 VA-API,一条完整的**解码 → 预处理 → 检测 → 描述 → 编码**视觉 pipeline,可以从容地装进一张 AMD Radeon GPU——而且帧数据从压缩输入到标注输出,始终驻留在这张 GPU 上。每个阶段都在关键处做了 GPU 加速,每个阶段都有合理的回退,而视觉语言后端更是可以在 vLLM 与 llama.cpp 之间即插即换。它为在 AMD 硬件上——在一张卡上——构建真实的视觉 AI 应用,提供了一个紧凑、自洽的蓝本。
