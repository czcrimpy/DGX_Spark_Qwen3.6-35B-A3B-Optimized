#!/bin/bash
# Baseline config — Intel INT4 AutoRound only, no optimizations.
# Expected: ~64 tok/s on DGX Spark.
# Use this to benchmark what the raw model does before v2 optimizations.

PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

docker run -d --name vllm-qwen36b-baseline \
    --gpus all --net=host --ipc=host \
    --entrypoint vllm \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${PROJECT_DIR}/configs/chat_template.jinja:/opt/unsloth.jinja:ro" \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    vllm-node-tf5 \
    serve Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    --served-model-name qwen \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --attention-backend FLASHINFER \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --load-format fastsafetensors --trust-remote-code \
    --chat-template /opt/unsloth.jinja \
    -tp 1
