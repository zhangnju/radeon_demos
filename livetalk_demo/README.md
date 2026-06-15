# LiveTalk Demo 说明文档

## 项目简介

**LiveTalk** 是一个实时多模态交互式数字人视频生成系统，通过改进的 On-Policy 蒸馏方法，将双向扩散模型蒸馏为因果少步自回归模型，实现超过 **20× 加速**，支持实时交互体验。

- **论文**：[arXiv:2512.23576](http://arxiv.org/abs/2512.23576)
- **模型**：[GAIR/LiveTalk-1.3B-V0.1 on HuggingFace](https://huggingface.co/GAIR/LiveTalk-1.3B-V0.1)

---

## 运行环境

| 项目 | 信息 |
|------|------|
| 服务器 | `nzhang@10.161.176.9` |
| Docker 容器 | `avatar_ning` |
| 容器镜像 | `rocm/vllm-dev:rocm7.1.1_navi_ubuntu22.04_py3.10_pytorch_2.8_vllm_0.10.2rc1` |
| 项目路径（容器内） | `/app/livetalk` |
| Python 版本 | 3.10 |
| GPU 环境 | ROCm 7.1.1 |

---

## 核心亮点

- **实时生成**：吞吐量 24.82 FPS，首帧延迟 0.33s
- **多模态输入**：支持文本、图像、音频联合驱动
- **高效推理**：4 步扩散蒸馏，推理时间从 ~83s 降至实时
- **多轮对话**：多轮交互基准上与 Veo3、Sora2 对比具有竞争力
- **端到端系统**：集成音频语言模型，支持对话式 AI

---

## 系统架构

```
用户输入（文本/音频）
        ↓
  Thinker/Talker (Audio LM)  ← 流式音频语言模型
        ↓ 流式音频
  Performer (4-step Diffusion)
    - 参考图像 + 音频 + 文本 prompt
    - Block-wise AR 生成（每块 3 帧 latent）
    - KV Cache 跨块复用，保持时序一致性
    - AHIS 锚点机制，长视频身份保持
        ↓
  并行 Denoising + VAE Decoding
        ↓
  实时输出视频流
```

**关键技术：**
- **AHIS（Anchor-Heavy Identity Sinks）**：将部分 KV 窗口作为固定身份锚点，用小滑动窗口做上下文，稳定长视频中的人物外观
- **并行流水线**：Diffusion 去噪与 VAE 解码并行，生成始终领先于播放

---

## 模型文件结构

```
/app/livetalk
├── OmniAvatar/                  # 多模态视频扩散基础模型
├── pretrained_checkpoints/
│   ├── LiveTalk-1.3B-V0.1/     # LiveTalk 主模型
│   ├── Wan2.1-T2V-1.3B/        # Wan2.1 基础模型（T5文本编码器 + VAE）
│   └── wav2vec2/               # 音频特征提取模型
├── configs/
│   └── causal_inference.yaml   # 推理配置文件
├── scripts/
│   ├── inference.sh            # 命令行推理脚本
│   └── inference_example.py   # 推理核心逻辑
├── examples/inference/
│   ├── example1.jpg            # 示例参考图像
│   └── example1.wav            # 示例音频
├── livetalk_demo.py            # Gradio Web Demo
├── inference.py                # 单次推理入口
└── requirements.txt
```

---

## 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dtype` | `bf16` | 模型精度 |
| `video_duration` | `5` | 视频时长（秒），建议取 3n+2：5,8,11,14... |
| `max_hw` | `720` | 分辨率：720→480p，1280→720p |
| `fps` | `16` | 输出帧率 |
| `num_steps` | `4` | 扩散去噪步数 |
| `local_attn_size` | `15` | 局部注意力窗口大小 |
| `num_frame_per_block` | `3` | 每个 AR 块的帧数 |
| `denoising_step_list` | `[1000,750,500,250]` | 去噪时间步列表 |

---

## 运行方式

### 方式一：命令行推理

```bash
# 进入容器
docker exec -it avatar_ning bash

# 执行推理
cd /app/livetalk
bash ./scripts/inference.sh
```

### 方式二：Gradio Web Demo

```bash
# 进入容器启动 Demo（默认端口 8888）
docker exec -it avatar_ning bash -c "cd /app/livetalk && python livetalk_demo.py --port 8888"
```

访问 `http://10.161.176.9:8888` 使用 Web 界面。

**Web Demo 功能：**
- 上传参考人像图片（或使用内置示例）
- 上传语音音频（WAV，16kHz）
- 编辑文本 Prompt
- 点击 Generate 生成口型同步视频

---

## 默认 Prompt

```
A realistic video of a person speaking directly to the camera.
The individual maintains steady eye contact with clear, expressive facial features.
Their facial expressions are naturally animated and emotionally engaging,
with precise lip movements perfectly synchronized to their speech.
```

---

## GPU 显存要求

推理约需 **20GB GPU 显存**（已在 AMD R9700/w7900 测试）。

---

## 引用

```bibtex
@article{chern2025livetalk,
  title={LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation},
  author={Chern, Ethan and Hu, Zhulin and Tang, Bohao and Su, Jiadi and Chern, Steffi and Deng, Zhijie and Liu, Pengfei},
  journal={arXiv preprint arXiv:2512.23576},
  year={2025}
}
```
