# HuggingFace BF16 可运行模型清单

生成时间：2026-06-15 08:21 UTC

## 筛选条件

| 条件 | 值 |
|------|----|
| GPU | 2× AMD W7900 48GB（共 96GB）|
| 格式 | BF16 |
| 框架 | HuggingFace Transformers + ROCm |
| 显存上限 | 86.4 GB（参数量 ≤ 46.4B）|
| 发布时间 | 2025-01-01 之后 |
| 排序 | trendingScore 降序 |

## 统计

- 已排除（黑名单 / 量化 / 小机构 finetune）：694
- 非 Transformers / ROCm 兼容：195
- 发布时间过早：181
- 显存不足：8
- 参数量未知：49
- **可运行模型：74**

## 模型列表

| # | 模型 | 参数量 | BF16显存 | Trending | 任务类型 | 发布日期 |
|---|------|--------|----------|----------|----------|----------|
| 1 | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 26.0B | 48.4GB | 773.00 | image-text-to-text | 2026-06-09 |
| 2 | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 12.0B | 22.4GB | 246.00 | any-to-any | 2026-05-23 |
| 3 | [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) | 12.0B | 22.4GB | 101.00 | any-to-any | 2026-05-23 |
| 4 | [sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) | 1.0B | 1.9GB | 49.00 | text-generation | 2026-05-17 |
| 5 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.0B | 16.8GB | 25.00 | image-text-to-text | 2026-02-27 |
| 6 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4.0B | 7.5GB | 21.00 | image-text-to-text | 2026-02-27 |
| 7 | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1.0B | 1.9GB | 21.00 | text-generation | 2026-05-21 |
| 8 | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | 3.2GB | 20.00 | automatic-speech-recognition | 2026-01-28 |
| 9 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 600M | 1.1GB | 18.00 | text-generation | 2025-04-27 |
| 10 | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30.0B | 55.9GB | 15.00 | text-generation | 2025-07-31 |
| 11 | [google/gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant) | 12.0B | 22.4GB | 14.00 | any-to-any | 2026-05-23 |
| 12 | [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) | 12.0B | 22.4GB | 14.00 | any-to-any | 2026-06-04 |
| 13 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.0B | 14.9GB | 12.00 | text-generation | 2025-04-27 |
| 14 | [google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B) | 31.0B | 57.7GB | 11.00 | image-text-to-text | 2026-03-12 |
| 15 | [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) | 31.0B | 57.7GB | 11.00 | image-text-to-text | 2026-04-28 |
| 16 | [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 26.0B | 48.4GB | 11.00 | image-text-to-text | 2026-03-12 |
| 17 | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 12.0B | 22.4GB | 11.00 | image-text-to-text | 2025-03-01 |
| 18 | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | 4.0B | 7.5GB | 10.00 | image-text-to-text | 2026-01-07 |
| 19 | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | 2.0B | 3.7GB | 10.00 | automatic-speech-recognition | 2026-04-16 |
| 20 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 800M | 1.5GB | 10.00 | image-text-to-text | 2026-02-28 |
| 21 | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.0B | 14.9GB | 9.00 | image-text-to-text | 2025-10-11 |
| 22 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2.0B | 3.7GB | 9.00 | image-text-to-text | 2026-02-28 |
| 23 | [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized) | 26.0B | 48.4GB | 8.00 | image-text-to-text | 2026-04-29 |
| 24 | [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) | 1.8B | 3.4GB | 8.00 | translation | 2026-05-11 |
| 25 | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4.0B | 7.5GB | 7.00 | image-text-to-text | 2025-02-20 |
| 26 | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | 1.9GB | 7.00 | text-generation | 2025-03-10 |
| 27 | [moonshotai/MoonViT-SO-400M](https://huggingface.co/moonshotai/MoonViT-SO-400M) | 400M | 0.7GB | 7.00 | image-feature-extraction | 2025-04-10 |
| 28 | [microsoft/UserLM-8b](https://huggingface.co/microsoft/UserLM-8b) | 8.0B | 14.9GB | 6.00 | text-generation | 2025-09-30 |
| 29 | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | 1.5B | 2.8GB | 6.00 | text-to-speech | 2025-08-25 |
| 30 | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 600M | 1.1GB | 6.00 | text-ranking | 2025-05-29 |
| 31 | [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | 258M | 0.5GB | 6.00 | image-text-to-text | 2025-05-19 |
| 32 | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 35.0B | 65.2GB | 0.00 | image-text-to-text | 2026-02-24 |
| 33 | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32.0B | 59.6GB | 0.00 | text-generation | 2025-04-27 |
| 34 | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | 32.0B | 59.6GB | 0.00 | image-text-to-text | 2025-10-19 |
| 35 | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 30.0B | 55.9GB | 0.00 | any-to-any | 2025-09-20 |
| 36 | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30.0B | 55.9GB | 0.00 | text-generation | 2025-04-27 |
| 37 | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 30.0B | 55.9GB | 0.00 | text-generation | 2025-07-28 |
| 38 | [Alibaba-NLP/Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) | 30.0B | 55.9GB | 0.00 | text-generation | 2025-09-16 |
| 39 | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27.0B | 50.3GB | 0.00 | image-text-to-text | 2025-03-01 |
| 40 | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27.0B | 50.3GB | 0.00 | image-text-to-text | 2026-02-24 |
| 41 | [baidu/ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) | 21.0B | 39.1GB | 0.00 | text-generation | 2025-09-08 |
| 42 | [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 17.0B | 31.7GB | 0.00 | image-text-to-text | 2025-04-02 |
| 43 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14.0B | 26.1GB | 0.00 | text-generation | 2025-04-27 |
| 44 | [zai-org/GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) | 9.0B | 16.8GB | 0.00 | image-text-to-text | 2025-06-28 |
| 45 | [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) | 8.0B | 14.9GB | 0.00 | text-generation | 2025-05-29 |
| 46 | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 8.0B | 14.9GB | 0.00 | text-generation | 2025-01-20 |
| 47 | [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | 8.0B | 14.9GB | 0.00 | text-ranking | 2025-05-29 |
| 48 | [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | 7.0B | 13.0GB | 0.00 | any-to-any | 2025-01-26 |
| 49 | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | 7.0B | 13.0GB | 0.00 | any-to-any | 2025-03-22 |
| 50 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.0B | 13.0GB | 0.00 | text-generation | 2025-01-20 |
| 51 | [tencent/Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B) | 7.0B | 13.0GB | 0.00 | translation | 2025-08-28 |
| 52 | [allenai/olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) | 7.0B | 13.0GB | 0.00 | image-text-to-text | 2025-01-15 |
| 53 | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | 4.0B | 7.5GB | 0.00 | image-text-to-text | 2025-05-19 |
| 54 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | 7.5GB | 0.00 | text-generation | 2025-08-05 |
| 55 | [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) | 4.0B | 7.5GB | 0.00 | image-text-to-text | 2026-01-12 |
| 56 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | 7.5GB | 0.00 | text-generation | 2025-04-27 |
| 57 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4.0B | 7.5GB | 0.00 | image-text-to-text | 2025-10-11 |
| 58 | [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4.0B | 7.5GB | 0.00 | text-ranking | 2025-06-03 |
| 59 | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.0B | 5.6GB | 0.00 | text-generation | 2025-07-08 |
| 60 | [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3.0B | 5.6GB | 0.00 | text-to-speech | 2025-07-01 |
| 61 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3.0B | 5.6GB | 0.00 | image-text-to-text | 2025-01-26 |
| 62 | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 3.0B | 5.6GB | 0.00 | any-to-any | 2025-04-30 |
| 63 | [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | 2.0B | 3.7GB | 0.00 | text-generation | 2025-04-15 |
| 64 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2.0B | 3.7GB | 0.00 | image-text-to-text | 2025-10-19 |
| 65 | [tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B) | 1.8B | 3.4GB | 0.00 | translation | 2025-12-25 |
| 66 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | 3.2GB | 0.00 | text-generation | 2025-04-27 |
| 67 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | 2.8GB | 0.00 | text-generation | 2025-01-20 |
| 68 | [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) | 1.0B | 1.9GB | 0.00 | text-to-speech | 2025-03-06 |
| 69 | [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1.0B | 1.9GB | 0.00 | image-text-to-text | 2026-01-16 |
| 70 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 600M | 1.1GB | 0.00 | automatic-speech-recognition | 2026-01-28 |
| 71 | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | 500M | 0.9GB | 0.00 | text-to-speech | 2025-12-04 |
| 72 | [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | 270M | 0.5GB | 0.00 | text-generation | 2025-08-05 |
| 73 | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) | 270M | 0.5GB | 0.00 | text-generation | 2025-10-08 |
| 74 | [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 256M | 0.5GB | 0.00 | image-text-to-text | 2025-01-17 |
