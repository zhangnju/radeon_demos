# Qwen Image Edit Demo 说明文档

## 项目简介

基于 **vLLM-Omni** 框架运行 **Qwen-Image-Edit** 模型，提供图像编辑能力。支持离线推理（本地直接调用）和在线服务（OpenAI 兼容 API）两种模式。

---

## 运行环境

| 项目 | 信息 |
|------|------|
| 服务器 | `nzhang@10.176.18.224` |
| Docker 容器 | `vllm_ning` |
| 容器镜像 | `rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0` |
| Python 版本 | 3.12.3 |
| vllm-omni 版本 | `0.16.0+rocm` |
| ROCm 版本 | 7.2.1 |
| GPU 架构 | gfx1100 (RDNA3 / RX 7900 系列) |

---

## 模型信息

| 模型 | 路径 | 说明 |
|------|------|------|
| `Qwen/Qwen-Image-Edit` | `/models/Qwen-Image-Edit/` | 单图编辑，主力版本 |
| `Qwen/Qwen-Image-Edit-2509` | HuggingFace | 多图输入版本（2509） |
| `Qwen/Qwen-Image-Edit-2511` | HuggingFace | 多图输入版本（2511） |

---

## 项目结构

```
/app/vllm-omni/examples/
├── offline_inference/image_to_image/
│   ├── image_edit.py               # 命令行离线推理主脚本
│   ├── image_edit_demo.py          # Gradio 离线 Demo（直接加载模型）
│   ├── run_qwen_image_edit_2511.sh # 2511 模型 + cache_dit 加速示例
│   ├── image_to_image.md           # 离线推理文档
│   └── qwen-bear.png               # 示例图片
└── online_serving/image_to_image/
    ├── run_server.sh               # vLLM 服务启动脚本
    ├── openai_chat_client.py       # Python OpenAI 兼容客户端
    ├── run_curl_image_edit.sh      # curl 调用示例
    ├── gradio_demo.py              # Gradio 在线 Demo（连接服务端）
    └── README.md                   # 在线服务文档

/app/Qwen-Image-Edit/               # 模型文件（容器内软链）
/models/Qwen-Image-Edit/            # 模型实际存放路径
```

---

## 方式一：离线推理（直接加载模型）

### 单图编辑

```bash
python image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

### 多图合并编辑（需用 2509/2511 模型）

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2509 \
  --image img1.png img2.png \
  --prompt "Combine these images into a single scene" \
  --output output_image_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --guidance-scale 1.0
```

### 使用 cache_dit 加速（2511 模型）

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image qwen-bear.png \
  --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland." \
  --output output_image_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --cache-backend cache_dit
```

### 使用 tea_cache 加速

```bash
python image_edit.py \
  --image input.png \
  --prompt "Edit description" \
  --cache-backend tea_cache \
  --tea-cache-rel-l1-thresh 0.25
```

### Gradio 离线 Demo

```bash
python image_edit_demo.py --model /models/Qwen-Image-Edit/ --port 7860
```

访问 `http://10.176.18.224:7860`

---

## 方式二：在线服务（OpenAI 兼容 API）

### 启动 vLLM 服务

```bash
# 默认端口 8092
vllm serve Qwen/Qwen-Image-Edit --omni --port 8092

# 或使用脚本
bash run_server.sh

# 多图编辑版本
MODEL=Qwen/Qwen-Image-Edit-2509 bash run_server.sh
```

### 调用方式 1：curl

```bash
bash run_curl_image_edit.sh input.png "Convert this image to watercolor style"
```

手动构造请求：

```bash
IMG_B64=$(base64 -w0 input.png)
curl -s http://10.176.18.224:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Convert to watercolor style\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG_B64}\"}}
      ]
    }],
    \"extra_body\": {\"num_inference_steps\": 50, \"guidance_scale\": 1, \"seed\": 42}
  }" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2 | base64 -d > output.png
```

### 调用方式 2：Python 客户端

```bash
python openai_chat_client.py \
  --input input.png \
  --prompt "Convert to oil painting style" \
  --output output.png \
  --server http://10.176.18.224:8092

# 多图
python openai_chat_client.py \
  --input input1.png input2.png \
  --prompt "Combine these images" \
  --output output.png
```

### 调用方式 3：Gradio 在线 Demo

```bash
python gradio_demo.py --server http://localhost:8092 --port 7861
```

访问 `http://10.176.18.224:7861`

---

## API 参数说明

### 请求格式（extra_body）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `height` | int | None | 输出图像高度（像素） |
| `width` | int | None | 输出图像宽度（像素） |
| `num_inference_steps` | int | 50 | 去噪步数，步数越多质量越高 |
| `guidance_scale` | float | 7.5 | CFG 引导强度 |
| `seed` | int | None | 随机种子（可复现） |
| `negative_prompt` | str | None | 负向提示词 |
| `num_outputs_per_prompt` | int | 1 | 每个 prompt 生成图片数 |

### 命令行参数（image_edit.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen-Image-Edit` | 模型名称或本地路径 |
| `--image` | 必填 | 输入图片路径（可多个） |
| `--prompt` | 必填 | 编辑描述文本 |
| `--cfg-scale` | 4.0 | 分类器自由引导强度 |
| `--guidance-scale` | 1.0 | 引导蒸馏模型参数 |
| `--num-inference-steps` | 50 | 去噪步数 |
| `--output` | `output_image_edit.png` | 输出路径 |
| `--cache-backend` | None | 加速后端：`cache_dit` / `tea_cache` |
| `--enforce-eager` | False | 禁用 torch.compile |
| `--vae-use-slicing` | False | VAE 分片（省显存） |
| `--vae-use-tiling` | False | VAE 分块（省显存） |
| `--cfg-parallel-size` | 1 | CFG 并行 GPU 数（1 或 2） |

---

## 常用编辑指令示例

| 指令 | 效果 |
|------|------|
| `Convert this image to watercolor style` | 水彩风格转换 |
| `Convert the image to black and white` | 黑白化 |
| `Enhance the color saturation` | 色彩增强 |
| `Convert to cartoon style` | 卡通化 |
| `Add vintage filter effect` | 复古滤镜 |
| `Convert daytime scene to nighttime` | 日夜转换 |
| `Convert to oil painting style` | 油画风格 |
| `Let this mascot dance under the moon` | 场景创意编辑 |

---

## OOM 解决方案

遇到显存不足（OOM）时，启用以下选项：

```bash
# 离线推理
python image_edit.py --image input.png --prompt "..." \
  --vae-use-slicing --vae-use-tiling

# 在线服务
vllm serve Qwen/Qwen-Image-Edit --omni --port 8092 \
  --vae-use-slicing --vae-use-tiling
```

---

## 响应格式

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": [{
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,..."
        }
      }]
    }
  }]
}
```
