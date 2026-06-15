# HuggingFace BF16 可运行模型清单

生成时间：2026-06-15 08:38 UTC

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

- 已跳过（核心模型 / 量化 / 小机构 finetune）：694
- 非 Transformers / ROCm 兼容：195
- 发布时间过早：181
- 显存不足：8
- 参数量未知：49
- **可运行模型：75**

## 模型列表

| # | 模型 | 参数量 | BF16显存 | Trending | Likes | Downloads | 任务类型 | 发布日期 |
|---|------|--------|----------|----------|-------|-----------|----------|----------|
| 1 | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 26.0B | 48.4GB | 774.00 | 818 | 198,912 | image-text-to-text | 2026-06-09 |
| 2 | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 12.0B | 22.4GB | 246.00 | 1,016 | 1,084,405 | any-to-any | 2026-05-23 |
| 3 | [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) | 12.0B | 22.4GB | 101.00 | 547 | 213,502 | any-to-any | 2026-05-23 |
| 4 | [sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) | 1.0B | 1.9GB | 49.00 | 771 | 125,482 | text-generation | 2026-05-17 |
| 5 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.0B | 16.8GB | 25.00 | 1,564 | 5,782,976 | image-text-to-text | 2026-02-27 |
| 6 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4.0B | 7.5GB | 21.00 | 644 | 6,395,081 | image-text-to-text | 2026-02-27 |
| 7 | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1.0B | 1.9GB | 21.00 | 800 | 99,342 | text-generation | 2026-05-21 |
| 8 | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | 3.2GB | 20.00 | 883 | 1,163,984 | automatic-speech-recognition | 2026-01-28 |
| 9 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 600M | 1.1GB | 18.00 | 1,326 | 18,337,841 | text-generation | 2025-04-27 |
| 10 | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30.0B | 55.9GB | 15.00 | 1,111 | 1,348,446 | text-generation | 2025-07-31 |
| 11 | [google/gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant) | 12.0B | 22.4GB | 14.00 | 84 | 31,736 | any-to-any | 2026-05-23 |
| 12 | [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) | 12.0B | 22.4GB | 14.00 | 46 | 19,991 | any-to-any | 2026-06-04 |
| 13 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.0B | 14.9GB | 12.00 | 1,140 | 8,709,119 | text-generation | 2025-04-27 |
| 14 | [google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B) | 31.0B | 57.7GB | 11.00 | 416 | 340,212 | image-text-to-text | 2026-03-12 |
| 15 | [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) | 31.0B | 57.7GB | 11.00 | 25 | 7,605 | image-text-to-text | 2026-04-28 |
| 16 | [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 26.0B | 48.4GB | 11.00 | 311 | 97,391 | image-text-to-text | 2026-03-12 |
| 17 | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 12.0B | 22.4GB | 11.00 | 754 | 1,644,704 | image-text-to-text | 2025-03-01 |
| 18 | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | 4.0B | 7.5GB | 10.00 | 681 | 289,329 | image-text-to-text | 2026-01-07 |
| 19 | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | 2.0B | 3.7GB | 10.00 | 138 | 432,070 | automatic-speech-recognition | 2026-04-16 |
| 20 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 800M | 1.5GB | 10.00 | 572 | 1,696,378 | image-text-to-text | 2026-02-28 |
| 21 | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.0B | 14.9GB | 9.00 | 956 | 5,636,266 | image-text-to-text | 2025-10-11 |
| 22 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2.0B | 3.7GB | 9.00 | 306 | 1,164,402 | image-text-to-text | 2026-02-28 |
| 23 | [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized) | 26.0B | 48.4GB | 8.00 | 25 | 4,897 | image-text-to-text | 2026-04-29 |
| 24 | [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) | 1.8B | 3.4GB | 8.00 | 1,109 | 22,860 | translation | 2026-05-11 |
| 25 | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4.0B | 7.5GB | 7.00 | 1,367 | 1,136,729 | image-text-to-text | 2025-02-20 |
| 26 | [microsoft/FastContext-1.0-4B-SFT](https://huggingface.co/microsoft/FastContext-1.0-4B-SFT) | 4.0B | 7.5GB | 7.00 | 7 | 0 | text-generation | 2026-06-14 |
| 27 | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | 1.9GB | 7.00 | 1,009 | 1,549,005 | text-generation | 2025-03-10 |
| 28 | [moonshotai/MoonViT-SO-400M](https://huggingface.co/moonshotai/MoonViT-SO-400M) | 400M | 0.7GB | 7.00 | 78 | 1,765 | image-feature-extraction | 2025-04-10 |
| 29 | [microsoft/UserLM-8b](https://huggingface.co/microsoft/UserLM-8b) | 8.0B | 14.9GB | 6.00 | 374 | 547 | text-generation | 2025-09-30 |
| 30 | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | 1.5B | 2.8GB | 6.00 | 2,395 | 153,711 | text-to-speech | 2025-08-25 |
| 31 | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 600M | 1.1GB | 6.00 | 366 | 954,702 | text-ranking | 2025-05-29 |
| 32 | [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | 258M | 0.5GB | 6.00 | 1,193 | 173,576 | image-text-to-text | 2025-05-19 |
| 33 | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 35.0B | 65.2GB | 0.00 | 1,446 | 1,673,265 | image-text-to-text | 2026-02-24 |
| 34 | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32.0B | 59.6GB | 0.00 | 702 | 2,359,120 | text-generation | 2025-04-27 |
| 35 | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | 32.0B | 59.6GB | 0.00 | 204 | 2,236,870 | image-text-to-text | 2025-10-19 |
| 36 | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 30.0B | 55.9GB | 0.00 | 939 | 1,430,987 | any-to-any | 2025-09-20 |
| 37 | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30.0B | 55.9GB | 0.00 | 900 | 1,475,704 | text-generation | 2025-04-27 |
| 38 | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 30.0B | 55.9GB | 0.00 | 816 | 582,620 | text-generation | 2025-07-28 |
| 39 | [Alibaba-NLP/Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) | 30.0B | 55.9GB | 0.00 | 814 | 47,551 | text-generation | 2025-09-16 |
| 40 | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27.0B | 50.3GB | 0.00 | 1,978 | 971,122 | image-text-to-text | 2025-03-01 |
| 41 | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27.0B | 50.3GB | 0.00 | 984 | 1,763,667 | image-text-to-text | 2026-02-24 |
| 42 | [baidu/ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) | 21.0B | 39.1GB | 0.00 | 786 | 7,984 | text-generation | 2025-09-08 |
| 43 | [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 17.0B | 31.7GB | 0.00 | 1,304 | 400,100 | image-text-to-text | 2025-04-02 |
| 44 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14.0B | 26.1GB | 0.00 | 407 | 1,346,492 | text-generation | 2025-04-27 |
| 45 | [zai-org/GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) | 9.0B | 16.8GB | 0.00 | 776 | 347,177 | image-text-to-text | 2025-06-28 |
| 46 | [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) | 8.0B | 14.9GB | 0.00 | 1,075 | 229,752 | text-generation | 2025-05-29 |
| 47 | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 8.0B | 14.9GB | 0.00 | 865 | 370,253 | text-generation | 2025-01-20 |
| 48 | [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | 8.0B | 14.9GB | 0.00 | 238 | 852,208 | text-ranking | 2025-05-29 |
| 49 | [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | 7.0B | 13.0GB | 0.00 | 3,615 | 6,550 | any-to-any | 2025-01-26 |
| 50 | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | 7.0B | 13.0GB | 0.00 | 1,905 | 501,912 | any-to-any | 2025-03-22 |
| 51 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.0B | 13.0GB | 0.00 | 842 | 293,261 | text-generation | 2025-01-20 |
| 52 | [tencent/Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B) | 7.0B | 13.0GB | 0.00 | 731 | 11,487 | translation | 2025-08-28 |
| 53 | [allenai/olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) | 7.0B | 13.0GB | 0.00 | 708 | 20,870 | image-text-to-text | 2025-01-15 |
| 54 | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | 4.0B | 7.5GB | 0.00 | 983 | 175,694 | image-text-to-text | 2025-05-19 |
| 55 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | 7.5GB | 0.00 | 876 | 3,387,032 | text-generation | 2025-08-05 |
| 56 | [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) | 4.0B | 7.5GB | 0.00 | 788 | 69,133 | image-text-to-text | 2026-01-12 |
| 57 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | 7.5GB | 0.00 | 636 | 9,903,273 | text-generation | 2025-04-27 |
| 58 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4.0B | 7.5GB | 0.00 | 395 | 2,735,070 | image-text-to-text | 2025-10-11 |
| 59 | [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4.0B | 7.5GB | 0.00 | 143 | 889,704 | text-ranking | 2025-06-03 |
| 60 | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.0B | 5.6GB | 0.00 | 971 | 379,952 | text-generation | 2025-07-08 |
| 61 | [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3.0B | 5.6GB | 0.00 | 682 | 126,164 | text-to-speech | 2025-07-01 |
| 62 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3.0B | 5.6GB | 0.00 | 656 | 3,285,741 | image-text-to-text | 2025-01-26 |
| 63 | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 3.0B | 5.6GB | 0.00 | 336 | 1,151,732 | any-to-any | 2025-04-30 |
| 64 | [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | 2.0B | 3.7GB | 0.00 | 1,458 | 6,959 | text-generation | 2025-04-15 |
| 65 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2.0B | 3.7GB | 0.00 | 423 | 1,399,930 | image-text-to-text | 2025-10-19 |
| 66 | [tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B) | 1.8B | 3.4GB | 0.00 | 1,181 | 12,666 | translation | 2025-12-25 |
| 67 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | 3.2GB | 0.00 | 485 | 3,787,523 | text-generation | 2025-04-27 |
| 68 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | 2.8GB | 0.00 | 1,523 | 480,611 | text-generation | 2025-01-20 |
| 69 | [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) | 1.0B | 1.9GB | 0.00 | 2,392 | 194,193 | text-to-speech | 2025-03-06 |
| 70 | [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1.0B | 1.9GB | 0.00 | 694 | 196,007 | image-text-to-text | 2026-01-16 |
| 71 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 600M | 1.1GB | 0.00 | 302 | 612,912 | automatic-speech-recognition | 2026-01-28 |
| 72 | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | 500M | 0.9GB | 0.00 | 1,232 | 509,462 | text-to-speech | 2025-12-04 |
| 73 | [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | 270M | 0.5GB | 0.00 | 1,034 | 5,467,893 | text-generation | 2025-08-05 |
| 74 | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) | 270M | 0.5GB | 0.00 | 1,013 | 99,016 | text-generation | 2025-10-08 |
| 75 | [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 256M | 0.5GB | 0.00 | 366 | 667,186 | image-text-to-text | 2025-01-17 |

---

## 核心模型

以下模型已纳入核心部署，不在新增候选列表中。

| # | 模型 | HuggingFace 链接 |
|---|------|------------------|
| 1 | `BAAI/bge-m3` | https://huggingface.co/BAAI/bge-m3 |
| 2 | `OpenGVLab/InternVL3-78B` | https://huggingface.co/OpenGVLab/InternVL3-78B |
| 3 | `Qwen/QwQ-32B` | https://huggingface.co/Qwen/QwQ-32B |
| 4 | `Qwen/Qwen2.5-VL-7B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct |
| 5 | `Qwen/Qwen3.6-27B` | https://huggingface.co/Qwen/Qwen3.6-27B |
| 6 | `Qwen/Qwen3.6-35B-A3B` | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| 7 | `Tongyi-MAI/Z-Image-Turbo` | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| 8 | `Wan-AI/Wan2.2-T2V-A14B` | https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B |
| 9 | `black-forest-labs/FLUX.1-dev` | https://huggingface.co/black-forest-labs/FLUX.1-dev |
| 10 | `black-forest-labs/FLUX.2-dev` | https://huggingface.co/black-forest-labs/FLUX.2-dev |
| 11 | `black-forest-labs/FLUX.2-klein-9B` | https://huggingface.co/black-forest-labs/FLUX.2-klein-9B |
| 12 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |
| 13 | `google/gemma-4-26B-A4B-it` | https://huggingface.co/google/gemma-4-26B-A4B-it |
| 14 | `google/gemma-4-31B-it` | https://huggingface.co/google/gemma-4-31B-it |
| 15 | `google/gemma-4-E4B-it` | https://huggingface.co/google/gemma-4-E4B-it |
| 16 | `ibm-granite/granite-4.1-8b` | https://huggingface.co/ibm-granite/granite-4.1-8b |
| 17 | `meta-llama/Llama-3.1-8B-Instruct` | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| 18 | `meta-llama/Llama-3.3-70B-Instruct` | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| 19 | `openai/gpt-oss-120b` | https://huggingface.co/openai/gpt-oss-120b |
| 20 | `openai/gpt-oss-20b` | https://huggingface.co/openai/gpt-oss-20b |
| 21 | `openai/whisper-large-v3-turbo` | https://huggingface.co/openai/whisper-large-v3-turbo |
| 22 | `poolside/Laguna-XS.2` | https://huggingface.co/poolside/Laguna-XS.2 |
| 23 | `zai-org/GLM-4.7-Flash` | https://huggingface.co/zai-org/GLM-4.7-Flash |
