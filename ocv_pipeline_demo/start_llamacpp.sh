#!/bin/bash
# Launch llama.cpp server for vision-language inference on AMD ROCm / RDNA3/RDNA4
#
# llama.cpp ROCm build supports RDNA3 (gfx1100) and RDNA4 (gfx1201) natively.
# The server exposes an OpenAI-compatible API on the configured port.
#
# Supported multimodal models (GGUF format):
#   Qwen3-VL, Qwen2-VL, LLaVA, MobileVLM, moondream2, InternVL, etc.
#
# Usage:
#   bash start_llamacpp.sh [model.gguf] [mmproj.gguf] [port]
#
# Examples:
#   bash start_llamacpp.sh /models/Qwen3-VL-8B-Instruct-Q8_0.gguf /models/mmproj-F16.gguf
#   bash start_llamacpp.sh /models/llava-v1.6-mistral-7b.Q4_K_M.gguf /models/llava-v1.6-mmproj-model-f16.gguf 8199

MODEL="${1:-/models/Qwen3-VL-8B-Instruct-Q8_0.gguf}"
MMPROJ="${2:-}"   # multimodal projector weights (required for vision models)
PORT="${3:-8199}"
GPU_LAYERS="${LLAMACPP_GPU_LAYERS:-99}"   # offload all layers to GPU by default
CTX_SIZE="${LLAMACPP_CTX:-4096}"

echo "[llama.cpp] ============================================"
echo "[llama.cpp] llama.cpp VLM server — AMD ROCm / HIP"
echo "[llama.cpp] Model    : ${MODEL}"
echo "[llama.cpp] Port     : ${PORT}"
echo "[llama.cpp] GPU layers: ${GPU_LAYERS}  (set LLAMACPP_GPU_LAYERS to override)"
echo "[llama.cpp] Context  : ${CTX_SIZE} tokens"
echo "[llama.cpp] ============================================"
echo ""

# --- Check if server is already running ---
if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[llama.cpp] Server already running on port ${PORT}"
    curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || \
        curl -s "http://localhost:${PORT}/v1/models"
    exit 0
fi

# --- Locate llama-server binary ---
LLAMA_SERVER=""
for candidate in \
    "$(which llama-server 2>/dev/null)" \
    "/opt/llama.cpp/bin/llama-server" \
    "/usr/local/bin/llama-server" \
    "./llama-server"; do
    if [ -x "$candidate" ]; then
        LLAMA_SERVER="$candidate"
        break
    fi
done

if [ -z "$LLAMA_SERVER" ]; then
    echo "[llama.cpp] ERROR: llama-server not found."
    echo ""
    echo "  Build llama.cpp with ROCm support:"
    echo "    git clone https://github.com/ggml-org/llama.cpp"
    echo "    cd llama.cpp"
    echo "    # For RDNA3 (gfx1100):"
    echo "    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release"
    echo "    # For RDNA4 (gfx1201):"
    echo "    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release"
    echo "    cmake --build build --target llama-server -j\$(nproc)"
    echo "    sudo cmake --install build"
    echo ""
    echo "  Or pull the pre-built Docker image:"
    echo "    docker pull ghcr.io/ggml-org/llama.cpp:server-rocm"
    exit 1
fi

echo "[llama.cpp] Using binary: ${LLAMA_SERVER}"

# --- Check model file ---
if [ ! -f "${MODEL}" ]; then
    echo "[llama.cpp] ERROR: model file not found: ${MODEL}"
    echo ""
    echo "  Download a vision GGUF model, for example:"
    echo "    # Qwen3-VL 8B (recommended — RDNA3/RDNA4 tested)"
    echo "    huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF \\"
    echo "        Qwen3-VL-8B-Instruct-Q8_0.gguf \\"
    echo "        mmproj-F16.gguf \\"
    echo "        --local-dir /models"
    exit 1
fi

# --- Build command ---
CMD="${LLAMA_SERVER}"
CMD="${CMD} --model ${MODEL}"
CMD="${CMD} --port ${PORT}"
CMD="${CMD} --n-gpu-layers ${GPU_LAYERS}"
CMD="${CMD} --ctx-size ${CTX_SIZE}"
CMD="${CMD} --host 0.0.0.0"

if [ -n "${MMPROJ}" ]; then
    if [ ! -f "${MMPROJ}" ]; then
        echo "[llama.cpp] WARNING: mmproj file not found: ${MMPROJ}"
        echo "[llama.cpp]          Vision features will be unavailable."
    else
        CMD="${CMD} --mmproj ${MMPROJ}"
        echo "[llama.cpp] Projector: ${MMPROJ}"
    fi
else
    echo "[llama.cpp] WARNING: no --mmproj specified; vision input will be disabled."
    echo "[llama.cpp]          Pass the mmproj GGUF as second argument."
fi

echo ""
echo "[llama.cpp] Running: ${CMD}"
echo ""

exec ${CMD} 2>&1 | tee /tmp/llamacpp_server.log
