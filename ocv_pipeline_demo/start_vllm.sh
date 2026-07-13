#!/bin/bash
# Launch vLLM server for Qwen3-VL-8B-Instruct

MODEL_PATH="/models/Qwen3-VL-8B-Instruct"
PORT=8198

echo "[vllm] Starting Qwen3-VL server on port ${PORT}..."
echo "[vllm] Model: ${MODEL_PATH}"
echo "[vllm] GPU memory utilization: 80%"
echo ""

# Check if already running
if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
    echo "[vllm] Server already running on port ${PORT}"
    curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool
    exit 0
fi

export PYTHONPATH=/opt/rocm/lib:${PYTHONPATH}

vllm serve ${MODEL_PATH} \
    --port ${PORT} \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype auto \
    2>&1 | tee /tmp/vllm_server.log
