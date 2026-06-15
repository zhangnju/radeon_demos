# HuggingFace BF16 可运行模型清单

生成时间：2026-06-15 12:25 UTC

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

- 已跳过（核心模型 / 量化 / 小机构 finetune）：691
- 非 Transformers / ROCm 兼容：197
- 发布时间过早：180
- 显存不足：21
- 参数量未知：8
- **可运行模型：102**

## 模型列表

| # | 模型 | 参数量 | BF16显存 | Trending | Likes | Downloads | 任务类型 | 发布日期 |
|---|------|--------|----------|----------|-------|-----------|----------|----------|
| 1 | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 26.0B | 48.4GB | 791.00 | 836 | 311,788 | image-text-to-text | 2026-06-09 |
| 2 | [CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) | 30.0B | 55.9GB | 323.00 | 375 | 11,145 | text-generation | 2026-06-05 |
| 3 | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 12.0B | 22.4GB | 243.00 | 1,018 | 1,160,435 | any-to-any | 2026-05-23 |
| 4 | [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) | 12.0B | 22.4GB | 97.00 | 548 | 250,498 | any-to-any | 2026-05-23 |
| 5 | [sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) | 1.0B | 1.9GB | 49.00 | 772 | 126,571 | text-generation | 2026-05-17 |
| 6 | [microsoft/FastContext-1.0-4B-SFT](https://huggingface.co/microsoft/FastContext-1.0-4B-SFT) | 4.0B | 7.5GB | 39.00 | 39 | 13 | text-generation | 2026-06-14 |
| 7 | [facebook/sam3](https://huggingface.co/facebook/sam3) | 1.7B | 3.2GB | 39.00 | 2,222 | 1,303,880 | mask-generation | 2025-11-07 |
| 8 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.0B | 16.8GB | 26.00 | 1,565 | 5,782,976 | image-text-to-text | 2026-02-27 |
| 9 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4.0B | 7.5GB | 21.00 | 644 | 6,395,081 | image-text-to-text | 2026-02-27 |
| 10 | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 2.0B | 3.7GB | 20.00 | 734 | 1,630,916 | any-to-any | 2026-03-02 |
| 11 | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | 3.2GB | 20.00 | 883 | 1,163,984 | automatic-speech-recognition | 2026-01-28 |
| 12 | [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) | 1.0B | 1.9GB | 20.00 | 1,643 | 205,808 | token-classification | 2026-04-17 |
| 13 | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1.0B | 1.9GB | 19.00 | 800 | 112,754 | text-generation | 2026-05-21 |
| 14 | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 1.0B | 1.9GB | 17.00 | 996 | 394,722 | automatic-speech-recognition | 2026-03-24 |
| 15 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 600M | 1.1GB | 16.00 | 1,326 | 18,337,841 | text-generation | 2025-04-27 |
| 16 | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30.0B | 55.9GB | 15.00 | 1,111 | 1,348,446 | text-generation | 2025-07-31 |
| 17 | [google/gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant) | 12.0B | 22.4GB | 14.00 | 84 | 35,793 | any-to-any | 2026-05-23 |
| 18 | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | 1.3B | 2.4GB | 14.00 | 1,115 | 538,417 | image-text-to-text | 2026-04-13 |
| 19 | [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) | 12.0B | 22.4GB | 13.00 | 47 | 21,364 | any-to-any | 2026-06-04 |
| 20 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.0B | 14.9GB | 12.00 | 1,140 | 8,709,119 | text-generation | 2025-04-27 |
| 21 | [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) | 4.0B | 7.5GB | 12.00 | 317 | 525,223 | any-to-any | 2026-03-02 |
| 22 | [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) | 2.0B | 3.7GB | 12.00 | 336 | 488,047 | any-to-any | 2026-03-02 |
| 23 | [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) | 31.0B | 57.7GB | 11.00 | 25 | 8,082 | image-text-to-text | 2026-04-28 |
| 24 | [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 26.0B | 48.4GB | 11.00 | 311 | 100,477 | image-text-to-text | 2026-03-12 |
| 25 | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 12.0B | 22.4GB | 11.00 | 754 | 1,644,704 | image-text-to-text | 2025-03-01 |
| 26 | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.0B | 14.9GB | 11.00 | 958 | 5,636,266 | image-text-to-text | 2025-10-11 |
| 27 | [google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B) | 31.0B | 57.7GB | 10.00 | 416 | 325,600 | image-text-to-text | 2026-03-12 |
| 28 | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | 4.0B | 7.5GB | 10.00 | 681 | 289,329 | image-text-to-text | 2026-01-07 |
| 29 | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | 2.0B | 3.7GB | 10.00 | 138 | 434,900 | automatic-speech-recognition | 2026-04-16 |
| 30 | [baidu/Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) | 1.0B | 1.9GB | 10.00 | 1,188 | 169,834 | image-text-to-text | 2026-03-18 |
| 31 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 800M | 1.5GB | 10.00 | 572 | 1,696,378 | image-text-to-text | 2026-02-28 |
| 32 | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 9.0B | 16.8GB | 9.00 | 1,825 | 2,577,979 | image-text-to-text | 2026-01-30 |
| 33 | [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) | 1.8B | 3.4GB | 9.00 | 1,110 | 23,389 | translation | 2026-05-11 |
| 34 | [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 1.0B | 1.9GB | 9.00 | 987 | 1,380,897 | image-text-to-text | 2026-01-27 |
| 35 | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | 300M | 0.6GB | 9.00 | 1,177 | 465,055 | automatic-speech-recognition | 2026-01-21 |
| 36 | [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized) | 26.0B | 48.4GB | 8.00 | 25 | 6,081 | image-text-to-text | 2026-04-29 |
| 37 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2.0B | 3.7GB | 8.00 | 306 | 1,164,402 | image-text-to-text | 2026-02-28 |
| 38 | [microsoft/UserLM-8b](https://huggingface.co/microsoft/UserLM-8b) | 8.0B | 14.9GB | 7.00 | 375 | 547 | text-generation | 2025-09-30 |
| 39 | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4.0B | 7.5GB | 7.00 | 1,367 | 1,136,729 | image-text-to-text | 2025-02-20 |
| 40 | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | 1.9GB | 7.00 | 1,010 | 1,549,005 | text-generation | 2025-03-10 |
| 41 | [moonshotai/MoonViT-SO-400M](https://huggingface.co/moonshotai/MoonViT-SO-400M) | 400M | 0.7GB | 7.00 | 78 | 1,765 | image-feature-extraction | 2025-04-10 |
| 42 | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | 1.5B | 2.8GB | 6.00 | 2,395 | 153,711 | text-to-speech | 2025-08-25 |
| 43 | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 600M | 1.1GB | 6.00 | 366 | 954,702 | text-ranking | 2025-05-29 |
| 44 | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 35.0B | 65.2GB | 0.00 | 1,446 | 1,673,265 | image-text-to-text | 2026-02-24 |
| 45 | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32.0B | 59.6GB | 0.00 | 702 | 2,359,120 | text-generation | 2025-04-27 |
| 46 | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | 32.0B | 59.6GB | 0.00 | 204 | 2,236,870 | image-text-to-text | 2025-10-19 |
| 47 | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 30.0B | 55.9GB | 0.00 | 939 | 1,430,987 | any-to-any | 2025-09-20 |
| 48 | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30.0B | 55.9GB | 0.00 | 900 | 1,475,704 | text-generation | 2025-04-27 |
| 49 | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 30.0B | 55.9GB | 0.00 | 816 | 582,620 | text-generation | 2025-07-28 |
| 50 | [Alibaba-NLP/Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) | 30.0B | 55.9GB | 0.00 | 814 | 47,551 | text-generation | 2025-09-16 |
| 51 | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27.0B | 50.3GB | 0.00 | 1,978 | 971,122 | image-text-to-text | 2025-03-01 |
| 52 | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27.0B | 50.3GB | 0.00 | 984 | 1,763,667 | image-text-to-text | 2026-02-24 |
| 53 | [baidu/ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) | 21.0B | 39.1GB | 0.00 | 786 | 7,984 | text-generation | 2025-09-08 |
| 54 | [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 17.0B | 31.7GB | 0.00 | 1,304 | 400,100 | image-text-to-text | 2025-04-02 |
| 55 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14.0B | 26.1GB | 0.00 | 407 | 1,346,492 | text-generation | 2025-04-27 |
| 56 | [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) | 9.0B | 16.8GB | 0.00 | 1,402 | 110,902 | text-generation | 2025-07-20 |
| 57 | [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | 9.0B | 16.8GB | 0.00 | 1,392 | 162,779 | any-to-any | 2026-02-03 |
| 58 | [zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6) | 9.0B | 16.8GB | 0.00 | 1,227 | 10,419 | text-generation | 2025-09-29 |
| 59 | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | 9.0B | 16.8GB | 0.00 | 1,093 | 63,399 | image-text-to-text | 2025-08-24 |
| 60 | [zai-org/GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) | 9.0B | 16.8GB | 0.00 | 776 | 347,177 | image-text-to-text | 2025-06-28 |
| 61 | [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V) | 9.0B | 16.8GB | 0.00 | 719 | 83,393 | image-text-to-text | 2025-08-10 |
| 62 | [openbmb/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) | 9.0B | 16.8GB | 0.00 | 682 | 4,384 | text-generation | 2026-02-11 |
| 63 | [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6) | 8.0B | 14.9GB | 0.00 | 1,292 | 288,011 | any-to-any | 2025-01-12 |
| 64 | [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) | 8.0B | 14.9GB | 0.00 | 1,075 | 229,752 | text-generation | 2025-05-29 |
| 65 | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) | 8.0B | 14.9GB | 0.00 | 917 | 12,780 | image-text-to-text | 2025-06-03 |
| 66 | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 8.0B | 14.9GB | 0.00 | 865 | 370,253 | text-generation | 2025-01-20 |
| 67 | [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | 8.0B | 14.9GB | 0.00 | 238 | 852,208 | text-ranking | 2025-05-29 |
| 68 | [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | 7.0B | 13.0GB | 0.00 | 3,615 | 6,550 | any-to-any | 2025-01-26 |
| 69 | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | 7.0B | 13.0GB | 0.00 | 1,905 | 501,912 | any-to-any | 2025-03-22 |
| 70 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.0B | 13.0GB | 0.00 | 843 | 293,261 | text-generation | 2025-01-20 |
| 71 | [tencent/Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B) | 7.0B | 13.0GB | 0.00 | 731 | 11,487 | translation | 2025-08-28 |
| 72 | [allenai/olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) | 7.0B | 13.0GB | 0.00 | 708 | 20,870 | image-text-to-text | 2025-01-15 |
| 73 | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 5.6B | 10.4GB | 0.00 | 1,603 | 350,119 | automatic-speech-recognition | 2025-02-24 |
| 74 | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | 4.0B | 7.5GB | 0.00 | 983 | 175,694 | image-text-to-text | 2025-05-19 |
| 75 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | 7.5GB | 0.00 | 876 | 3,387,032 | text-generation | 2025-08-05 |
| 76 | [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) | 4.0B | 7.5GB | 0.00 | 788 | 69,133 | image-text-to-text | 2026-01-12 |
| 77 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | 7.5GB | 0.00 | 636 | 9,903,273 | text-generation | 2025-04-27 |
| 78 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4.0B | 7.5GB | 0.00 | 395 | 2,735,070 | image-text-to-text | 2025-10-11 |
| 79 | [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4.0B | 7.5GB | 0.00 | 143 | 889,704 | text-ranking | 2025-06-03 |
| 80 | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B | 7.1GB | 0.00 | 768 | 888,853 | text-generation | 2025-02-19 |
| 81 | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.0B | 5.6GB | 0.00 | 971 | 379,952 | text-generation | 2025-07-08 |
| 82 | [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3.0B | 5.6GB | 0.00 | 682 | 126,164 | text-to-speech | 2025-07-01 |
| 83 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3.0B | 5.6GB | 0.00 | 656 | 3,285,741 | image-text-to-text | 2025-01-26 |
| 84 | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 3.0B | 5.6GB | 0.00 | 336 | 1,151,732 | any-to-any | 2025-04-30 |
| 85 | [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | 2.0B | 3.7GB | 0.00 | 1,458 | 6,959 | text-generation | 2025-04-15 |
| 86 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2.0B | 3.7GB | 0.00 | 423 | 1,399,930 | image-text-to-text | 2025-10-19 |
| 87 | [tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B) | 1.8B | 3.4GB | 0.00 | 1,181 | 12,666 | translation | 2025-12-25 |
| 88 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | 3.2GB | 0.00 | 485 | 3,787,523 | text-generation | 2025-04-27 |
| 89 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | 2.8GB | 0.00 | 1,523 | 480,611 | text-generation | 2025-01-20 |
| 90 | [jinaai/ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2) | 1.5B | 2.8GB | 0.00 | 794 | 173,180 | text-generation | 2025-01-13 |
| 91 | [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 1.0B | 1.9GB | 0.00 | 3,277 | 1,691,094 | image-text-to-text | 2025-10-17 |
| 92 | [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) | 1.0B | 1.9GB | 0.00 | 2,392 | 194,193 | text-to-speech | 2025-03-06 |
| 93 | [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1.0B | 1.9GB | 0.00 | 695 | 196,007 | image-text-to-text | 2026-01-16 |
| 94 | [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) | 1.0B | 1.9GB | 0.00 | 70 | 773,382 | zero-shot-image-classification | 2025-02-18 |
| 95 | [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) | 1.0B | 1.9GB | 0.00 | 4 | 592,935 | zero-shot-image-classification | 2025-02-17 |
| 96 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 600M | 1.1GB | 0.00 | 302 | 612,912 | automatic-speech-recognition | 2026-01-28 |
| 97 | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | 500M | 0.9GB | 0.00 | 1,232 | 509,462 | text-to-speech | 2025-12-04 |
| 98 | [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) | 400M | 0.7GB | 0.00 | 32 | 573,122 | zero-shot-image-classification | 2025-02-18 |
| 99 | [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | 270M | 0.5GB | 0.00 | 1,035 | 5,467,893 | text-generation | 2025-08-05 |
| 100 | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) | 270M | 0.5GB | 0.00 | 1,013 | 99,016 | text-generation | 2025-10-08 |
| 101 | [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | 258M | 0.5GB | 0.00 | 1,193 | 173,576 | image-text-to-text | 2025-05-19 |
| 102 | [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 256M | 0.5GB | 0.00 | 366 | 667,186 | image-text-to-text | 2025-01-17 |

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

---

## 参数量未知的模型

以下模型无法从 safetensors 元数据或模型名推断参数量，需人工核查。

| # | 模型 | HuggingFace 链接 |
|---|------|------------------|
| 1 | `google/gemma-4-E2B-it-qat-mobile-transformers` | https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers |
| 2 | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized |
| 3 | `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` | https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant |
| 4 | `google/gemma-4-E4B-it-qat-mobile-transformers` | https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers |
| 5 | `google/gemma-4-E4B-it-qat-q4_0-unquantized` | https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized |
| 6 | `microsoft/SchGen` | https://huggingface.co/microsoft/SchGen |
| 7 | `google/gemma-3n-E4B-it-litert-preview` | https://huggingface.co/google/gemma-3n-E4B-it-litert-preview |
| 8 | `microsoft/OmniParser-v2.0` | https://huggingface.co/microsoft/OmniParser-v2.0 |
