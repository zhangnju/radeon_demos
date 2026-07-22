#!/usr/bin/env python3
"""Render the end-to-end pipeline architecture diagram (horizontal) as a JPEG."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- palette (AMD-ish, brand-neutral, colorblind-safe) ---
INK      = "#1a1a1a"
MUTED    = "#5b6470"
CARD_BG  = "#ffffff"
GPU_BG   = "#eef4fb"
GPU_EDGE = "#c9d6e5"
STAGE    = "#111827"
ACCENT_DEC = "#1864ab"   # blue   – video decode (VCN)
ACCENT_PRE = "#0b7285"   # teal   – preprocess
ACCENT_DET = "#c92a2a"   # red    – detection
ACCENT_VLM = "#5f3dc4"   # violet – VLM
ACCENT_ENC = "#1864ab"   # blue   – video encode (VCN)
ACCENT_OUT = "#2b8a3e"   # green  – output
FAINT      = "#98a2b3"   # de-emphasized CPU steps
FAINT_BG   = "#f4f6f9"

fig, ax = plt.subplots(figsize=(17, 6.6), dpi=150)
ax.set_xlim(0, 17)
ax.set_ylim(0, 6.6)
ax.axis("off")
fig.patch.set_facecolor("#ffffff")

# ---------------------------------------------------------------- title
ax.text(8.5, 6.25, "End-to-End Vision AI Pipeline on a Single AMD Radeon GPU",
        ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
ax.text(8.5, 5.78,
        "Decode · preprocess · detect · describe · encode — the full chain on one GPU via ROCm/HIP",
        ha="center", va="center", fontsize=11, color=MUTED)

# ---------------------------------------------------------------- GPU boundary box
gpu = FancyBboxPatch((0.35, 1.15), 16.3, 4.05,
                     boxstyle="round,pad=0.02,rounding_size=0.15",
                     linewidth=2.0, edgecolor=GPU_EDGE, facecolor=GPU_BG, zorder=0)
ax.add_patch(gpu)
ax.text(0.62, 4.92, "Single AMD Radeon GPU  ·  ROCm / HIP",
        ha="left", va="center", fontsize=11.5, fontweight="bold", color="#37506b")

# card geometry
CY   = 3.15          # vertical center of the row of GPU cards
W    = 2.55          # heavy card width
H    = 1.75          # heavy card height
LW   = 2.2           # light step width
LH   = 1.15          # light step height

def heavy(cx, title, sub, tag, accent):
    x = cx - W / 2
    y = CY - H / 2
    card = FancyBboxPatch((x, y), W, H,
                          boxstyle="round,pad=0.02,rounding_size=0.10",
                          linewidth=1.8, edgecolor=accent, facecolor=CARD_BG, zorder=3)
    ax.add_patch(card)
    # accent bar on top
    bar = FancyBboxPatch((x, y + H - 0.16), W, 0.16,
                         boxstyle="round,pad=0,rounding_size=0.02",
                         linewidth=0, facecolor=accent, zorder=4)
    ax.add_patch(bar)
    ax.text(cx, y + H - 0.52, title, ha="center", va="center",
            fontsize=11.2, fontweight="bold", color=STAGE, zorder=5)
    ax.text(cx, y + H - 0.92, sub, ha="center", va="center",
            fontsize=8.3, color=MUTED, zorder=5, wrap=True)
    ax.text(cx, y + 0.24, tag, ha="center", va="center",
            fontsize=8.2, color=accent, fontweight="bold", zorder=5,
            fontfamily="monospace")

def light(cx, title, tag):
    x = cx - LW / 2
    y = CY - LH / 2
    card = FancyBboxPatch((x, y), LW, LH,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=1.0, edgecolor=FAINT, facecolor=FAINT_BG,
                          linestyle=(0, (4, 2)), zorder=3)
    ax.add_patch(card)
    ax.text(cx, CY + 0.14, title, ha="center", va="center",
            fontsize=8.7, color=MUTED, zorder=5)
    ax.text(cx, CY - 0.28, tag, ha="center", va="center",
            fontsize=7.6, color=FAINT, zorder=5, fontfamily="monospace")

def harrow(x0, x1, faint=False):
    ax.add_patch(FancyArrowPatch(
        (x0, CY), (x1, CY), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.3 if faint else 2.0,
        color=FAINT if faint else MUTED, zorder=2))

# x-centers across the row: 6 GPU cards + 2 light steps interleaved
# order: decode -> preprocess -> detect -> [NMS+ROI] -> VLM -> [overlay] -> encode
x_dec = 2.05
x_pre = 4.55
x_det = 7.05
x_nms = 9.15
x_vlm = 11.35
x_ovl = 13.5
x_enc = 15.35

harrow(x_dec + W/2, x_pre - W/2)
harrow(x_pre + W/2, x_det - W/2)
harrow(x_det + W/2, x_nms - LW/2, faint=True)
harrow(x_nms + LW/2, x_vlm - W/2, faint=True)
harrow(x_vlm + W/2, x_ovl - LW/2, faint=True)
harrow(x_ovl + LW/2, x_enc - W/2, faint=True)

heavy(x_dec, "Video Decode", "H.264 / HEVC\nbitstream", "rocDecode · VCN", ACCENT_DEC)
heavy(x_pre, "Preprocess", "letterbox · RGB\nnormalize", "cv::cuda / HIP", ACCENT_PRE)
heavy(x_det, "YOLO26x Detect", "300 boxes,\nzero-copy", "MIGraphX FP16", ACCENT_DET)
light(x_nms, "NMS + ROI crop", "CPU · OpenCV")
heavy(x_vlm, "Qwen3-VL", "per-ROI text\nasync", "vLLM | llama.cpp", ACCENT_VLM)
light(x_ovl, "Overlay draw", "CPU · OpenCV")
heavy(x_enc, "Video Encode", "annotated\nH.264 stream", "VA-API · VCN", ACCENT_ENC)

# input / output labels outside the box
ax.annotate("", xy=(x_dec - W/2, CY), xytext=(0.05, CY),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.0))
ax.text(0.05, CY + 0.42, "Video In", ha="left", va="center",
        fontsize=9.5, color="#37506b", style="italic")
ax.text(0.05, CY - 0.42, "file·RTSP·cam", ha="left", va="center",
        fontsize=8, color=MUTED, style="italic")

ax.annotate("", xy=(16.95, CY), xytext=(x_enc + W/2, CY),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT_OUT, lw=2.0))
ax.text(16.92, CY + 0.42, "Output", ha="right", va="center",
        fontsize=9.5, fontweight="bold", color=ACCENT_OUT)
ax.text(16.92, CY - 0.42, "annotated .mp4", ha="right", va="center",
        fontsize=8, color=MUTED, style="italic")

# async VLM note (below VLM card)
ax.annotate("async: fire-and-forget every N frames — main loop never stalls",
            xy=(x_vlm, CY - H/2), xytext=(x_vlm, 1.45),
            ha="center", va="center", fontsize=8, color=ACCENT_VLM,
            arrowprops=dict(arrowstyle="->", color=ACCENT_VLM, lw=1.1), zorder=6)

# legend
ax.text(0.62, 1.42,
        "Bold cards = GPU compute / VCN (ROCm·HIP)     Faint dashed = lightweight CPU steps",
        ha="left", va="center", fontsize=8.6, color=MUTED, style="italic")

# footer
ax.text(8.5, 0.55,
        "rocDecode hardware decode  ·  zero-copy GPU-resident inference  ·  "
        "VA-API hardware encode  ·  pluggable VLM backend",
        ha="center", va="center", fontsize=9, color=MUTED)

plt.tight_layout()
out = "/home/radeon_demos/ocv_pipeline_demo/assets/pipeline_architecture.jpg"
fig.savefig(out, format="jpeg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 92})
print("saved", out)
