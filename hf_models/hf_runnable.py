#!/usr/bin/env python3
"""
从 HuggingFace API 三个榜单中筛选可在 2×48G（共 96G）显卡以 bf16 格式运行的模型。
排除已有的 23 个模型，结果按 trendingScore 降序排列。

数据来源:
  https://huggingface.co/api/models?sort=trendingScore
  https://huggingface.co/api/models?sort=likes
  https://huggingface.co/api/models?sort=downloads
"""

import requests
import json
import re
import csv
from datetime import datetime, timezone
from typing import Optional

# ── 显存配置 ──────────────────────────────────────────────────────────────────

TOTAL_VRAM_GB  = 96           # 2 × AMD W7900 48GB
VRAM_RATIO     = 0.90         # 预留 10% 给 KV cache / activations
MAX_VRAM_BYTES = int(TOTAL_VRAM_GB * (1024 ** 3) * VRAM_RATIO)   # ~86.4 GB
BF16_BYTES     = 2            # bf16: 2 bytes / param
MAX_PARAMS     = MAX_VRAM_BYTES // BF16_BYTES                     # ~46.4B 参数上限

# ── 黑名单（已有 / 无需重复） ──────────────────────────────────────────────────

EXCLUDED = {
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/QwQ-32B",
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E4B-it",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "zai-org/GLM-4.7-Flash",
    "ibm-granite/granite-4.1-8b",
    "poolside/Laguna-XS.2",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenGVLab/InternVL3-78B",
    "openai/whisper-large-v3-turbo",
    "black-forest-labs/FLUX.2-dev",
    "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.1-dev",
    "Tongyi-MAI/Z-Image-Turbo",
    "Wan-AI/Wan2.2-T2V-A14B",
    "BAAI/bge-m3",
}

# ── 数据源（加 limit=500 多捞一些） ───────────────────────────────────────────

SOURCES = [
    "https://huggingface.co/api/models?sort=trendingScore&limit=500&direction=-1",
    "https://huggingface.co/api/models?sort=likes&limit=500&direction=-1",
    "https://huggingface.co/api/models?sort=downloads&limit=500&direction=-1",
]

HEADERS = {"Accept": "application/json"}

# ── 参数量提取 ────────────────────────────────────────────────────────────────
# 对于 MoE 模型，用总参数量（非激活参数量）估算显存占用

_PATTERNS = [
    # 8x7B / 8X22B
    (re.compile(r'(\d+)[xX](\d+(?:\.\d+)?)B', re.I),
     lambda m: int(int(m.group(1)) * float(m.group(2)) * 1e9)),
    # 236B-A22B → 236B（取总参数）
    (re.compile(r'(\d+(?:\.\d+)?)B-A\d+(?:\.\d+)?B', re.I),
     lambda m: int(float(m.group(1)) * 1e9)),
    # 7B / 0.5B / 1.5B
    (re.compile(r'[_\-/](\d+(?:\.\d+)?)B(?:[_\-\s]|$)', re.I),
     lambda m: int(float(m.group(1)) * 1e9)),
    # 270M / 82M
    (re.compile(r'[_\-/](\d+(?:\.\d+)?)M(?:[_\-\s]|$)', re.I),
     lambda m: int(float(m.group(1)) * 1e6)),
]


# ── HuggingFace Transformers 兼容的 pipeline tag ────────────────────────────

TRANSFORMERS_TAGS = {
    "text-generation", "text2text-generation", "fill-mask",
    "question-answering", "summarization", "translation",
    "zero-shot-classification", "token-classification", "text-classification",
    "sentence-similarity", "feature-extraction",
    "image-classification", "image-segmentation", "object-detection",
    "image-to-text", "visual-question-answering", "document-question-answering",
    "image-text-to-text", "automatic-speech-recognition", "audio-classification",
    "depth-estimation", "video-classification",
}


# Tags 表示模型依赖 CUDA 专属运行时，ROCm 不支持
CUDA_ONLY_TAGS = {"tensorrt", "tensorrt-llm", "cuda-required"}


def is_transformers_compatible(model: dict) -> bool:
    lib = model.get("library_name", "")
    if lib == "transformers":
        return True
    if not lib and model.get("pipeline_tag", "") in TRANSFORMERS_TAGS:
        return True
    return False


def is_rocm_compatible(model: dict) -> bool:
    """排除依赖 CUDA 专属库（TensorRT 等）的模型。"""
    tags = {t.lower() for t in model.get("tags", [])}
    return not (tags & CUDA_ONLY_TAGS)


KEYWORD_BLACKLIST = {"abliterated", "uncensored", "heretic", "gguf"}

# 非 bf16 量化格式：模型名或 tag 中出现即排除
QUANT_BLACKLIST = {
    "fp8", "fp4", "nvfp4",
    "awq", "gptq",
    "w8a8", "w4a16", "w8a16", "w4a8",
    "int8", "int4",
    "bnb", "bitsandbytes",
    "exl2",
    "sq",          # SmoothQuant
}

# 这些组织的模型依赖 CUDA 专属库，AMD 上无法正常运行
ORG_BLACKLIST = {"nvidia"}

# 主流 AI 机构白名单 —— 这些机构发布的模型视为"原始模型"保留
TRUSTED_ORGS = {
    "meta-llama", "google", "mistralai", "qwen", "deepseek-ai",
    "microsoft", "cohere", "cohereforai", "ai21labs", "eleutherai",
    "tiiuae", "01-ai", "internlm", "baai", "openbmb", "moonshotai",
    "zai-org", "openai", "openai-community", "black-forest-labs",
    "wan-ai", "opengvlab", "allenai", "huggingfaceh4", "huggingfacetb",
    "ibm-granite", "poolside", "tongyi-mai", "hexgrad", "pyannote",
    "answerdotai", "sentence-transformers", "stabilityai", "coqui",
}

# 非白名单机构模型，likes 低于此阈值且带有 base_model tag 则视为小机构 finetune
MIN_LIKES_UNKNOWN_ORG = 500


def is_keyword_blocked(model_id: str) -> bool:
    name = model_id.lower()
    return any(kw in name for kw in KEYWORD_BLACKLIST)


def is_quant_blocked(model_id: str, model: dict) -> bool:
    """排除 fp8 / fp4 / nvfp4 格式（模型名或 tags 中出现均算）。"""
    name = model_id.lower()
    tags = {t.lower() for t in model.get("tags", [])}
    return bool(QUANT_BLACKLIST & (tags | {kw for kw in QUANT_BLACKLIST if kw in name}))


def is_org_blocked(model_id: str) -> bool:
    org = model_id.split("/")[0].lower()
    return org in ORG_BLACKLIST


def is_small_org_finetune(model_id: str, model: dict) -> bool:
    """非白名单机构 + 带 base_model tag（衍生模型）→ 小机构 finetune，过滤掉。"""
    org = model_id.split("/")[0].lower()
    if org in TRUSTED_ORGS:
        return False
    tags = model.get("tags", [])
    has_base_model_tag = any(t.startswith("base_model:") for t in tags)
    if has_base_model_tag:
        return True
    # 无 base_model tag 但 likes 极低，也视为无关紧要的小模型
    if model.get("likes", 0) < MIN_LIKES_UNKNOWN_ORG:
        return True
    return False


def get_params(model: dict) -> Optional[int]:
    """优先从 safetensors 元数据获取精确参数量，否则从模型名推断。"""
    st = model.get("safetensors")
    if isinstance(st, dict) and "total" in st:
        return st["total"]
    name = (model.get("id") or model.get("modelId", "")).split("/")[-1]
    for pat, calc in _PATTERNS:
        m = pat.search(name)
        if m:
            return calc(m)
    return None


def fmt_params(n: int) -> str:
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9:  return f"{n/1e9:.1f}B"
    if n >= 1e6:  return f"{n/1e6:.0f}M"
    return str(n)


def vram_bf16_gb(params: int) -> float:
    return params * BF16_BYTES / (1024 ** 3)


# ── 拉取数据 ──────────────────────────────────────────────────────────────────

print("=" * 70)
print("HuggingFace BF16 可运行模型筛选  (2×48G = 96G)")
print("=" * 70)

all_models: dict[str, dict] = {}
for url in SOURCES:
    print(f"Fetching {url} ...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        before = len(all_models)
        for m in r.json():
            mid = m.get("id") or m.get("modelId", "")
            if mid and mid not in all_models:
                all_models[mid] = m
        print(f"  → 新增 {len(all_models) - before} 个，累计 {len(all_models)} 个")
    except Exception as e:
        print(f"  [WARN] 获取失败: {e}")

print()

# ── 筛选 ─────────────────────────────────────────────────────────────────────

qualified    = []
cnt_excluded = 0
cnt_not_tf   = 0
cnt_too_old  = 0
cnt_large    = 0
cnt_unknown  = 0

for mid, m in all_models.items():
    if mid in EXCLUDED or is_keyword_blocked(mid) or is_org_blocked(mid) \
            or is_quant_blocked(mid, m) or is_small_org_finetune(mid, m):
        cnt_excluded += 1
        continue

    if not is_transformers_compatible(m) or not is_rocm_compatible(m):
        cnt_not_tf += 1
        continue

    created = (m.get("createdAt") or "")[:10]
    if created < "2025-01-01":
        cnt_too_old += 1
        continue

    params = get_params(m)
    if params is None:
        cnt_unknown += 1
        continue

    if params * BF16_BYTES > MAX_VRAM_BYTES:
        cnt_large += 1
        continue

    qualified.append({
        "id":            mid,
        "params":        params,
        "params_fmt":    fmt_params(params),
        "vram_gb":       round(vram_bf16_gb(params), 1),
        "pipeline_tag":  m.get("pipeline_tag", ""),
        "library_name":  m.get("library_name", ""),
        "likes":         m.get("likes", 0),
        "downloads":     m.get("downloads", 0),
        "trendingScore": m.get("trendingScore", 0.0),
        "createdAt":     (m.get("createdAt") or "")[:10],
    })

# 按 trendingScore 降序，同分时按参数量降序
qualified.sort(key=lambda x: (x["trendingScore"], x["params"]), reverse=True)
for i, m in enumerate(qualified, 1):
    m["rank"] = i

# ── 统计摘要 ─────────────────────────────────────────────────────────────────

print(f"参数量上限 (BF16, {VRAM_RATIO*100:.0f}% VRAM): {fmt_params(MAX_PARAMS)}"
      f"  ({MAX_VRAM_BYTES/(1024**3):.1f} GB)")
print(f"{'─'*50}")
print(f"  已排除（黑名单）:           {cnt_excluded}")
print(f"  非 Transformers 兼容:      {cnt_not_tf}")
print(f"  发布时间过早（<2025-01）:  {cnt_too_old}")
print(f"  显存不足（>96G BF16）:     {cnt_large}")
print(f"  参数量未知:                {cnt_unknown}")
print(f"  可运行模型:                {len(qualified)}")
print(f"{'─'*50}\n")

# ── 输出表格 ─────────────────────────────────────────────────────────────────

COL = f"{'#':>3}  {'模型 ID':<55} {'参数量':>8}  {'BF16显存':>8}  " \
      f"{'TrendScore':>10}  {'任务类型':<25}  {'发布日期'}"
print(COL)
print("─" * 125)
for m in qualified:
    print(
        f"{m['rank']:>3}. {m['id']:<55} {m['params_fmt']:>8}"
        f"  {m['vram_gb']:>6.1f}GB"
        f"  {m['trendingScore']:>10.2f}"
        f"  {m['pipeline_tag']:<25}"
        f"  {m['createdAt']}"
    )

# ── 保存 JSON ─────────────────────────────────────────────────────────────────

output = {
    "config": {
        "total_vram_gb":      TOTAL_VRAM_GB,
        "format":             "bf16",
        "gpu":                "2x AMD W7900 48GB",
        "vram_ratio":         VRAM_RATIO,
        "max_weight_vram_gb": round(MAX_VRAM_BYTES / (1024 ** 3), 1),
        "max_params":         MAX_PARAMS,
        "sort":               "trendingScore desc",
    },
    "qualified": qualified,
}

with open("hf_runnable.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── 保存 CSV ──────────────────────────────────────────────────────────────────

csv_fields = ["rank", "id", "params_fmt", "vram_gb", "trendingScore",
              "pipeline_tag", "likes", "downloads", "createdAt"]

with open("hf_runnable.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(qualified)

# ── 保存 Markdown ─────────────────────────────────────────────────────────────

generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

with open("hf_runnable.md", "w", encoding="utf-8") as f:
    f.write(f"# HuggingFace BF16 可运行模型清单\n\n")
    f.write(f"生成时间：{generated_at}\n\n")
    f.write(f"## 筛选条件\n\n")
    f.write(f"| 条件 | 值 |\n|------|----|\n")
    f.write(f"| GPU | 2× AMD W7900 48GB（共 96GB）|\n")
    f.write(f"| 格式 | BF16 |\n")
    f.write(f"| 框架 | HuggingFace Transformers + ROCm |\n")
    f.write(f"| 显存上限 | {MAX_VRAM_BYTES/(1024**3):.1f} GB（参数量 ≤ {fmt_params(MAX_PARAMS)}）|\n")
    f.write(f"| 发布时间 | 2025-01-01 之后 |\n")
    f.write(f"| 排序 | trendingScore 降序 |\n\n")
    f.write(f"## 统计\n\n")
    f.write(f"- 已排除（黑名单 / 量化 / 小机构 finetune）：{cnt_excluded}\n")
    f.write(f"- 非 Transformers / ROCm 兼容：{cnt_not_tf}\n")
    f.write(f"- 发布时间过早：{cnt_too_old}\n")
    f.write(f"- 显存不足：{cnt_large}\n")
    f.write(f"- 参数量未知：{cnt_unknown}\n")
    f.write(f"- **可运行模型：{len(qualified)}**\n\n")
    f.write(f"## 模型列表\n\n")
    f.write(f"| # | 模型 | 参数量 | BF16显存 | Trending | 任务类型 | 发布日期 |\n")
    f.write(f"|---|------|--------|----------|----------|----------|----------|\n")
    for m in qualified:
        hf_url = f"https://huggingface.co/{m['id']}"
        f.write(
            f"| {m['rank']} "
            f"| [{m['id']}]({hf_url}) "
            f"| {m['params_fmt']} "
            f"| {m['vram_gb']}GB "
            f"| {m['trendingScore']:.2f} "
            f"| {m['pipeline_tag']} "
            f"| {m['createdAt']} |\n"
        )

print(f"\n输出文件：")
print(f"  hf_runnable.json  — 完整数据（含配置）")
print(f"  hf_runnable.csv   — 表格，可用 Excel / Numbers 打开")
print(f"  hf_runnable.md    — Markdown 文档，含模型链接")
print(f"  共 {len(qualified)} 个模型")
