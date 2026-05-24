#!/bin/bash
# v2 config (RECOMMENDED) — INT8 LM Head + MTP-2 speculative + FlashInfer.
# Expected: ~113 tok/s average, ~125 tok/s peak on DGX Spark.
# No hybrid checkpoint needed — works with plain Intel INT4 AutoRound.

PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

docker run -d --name vllm-qwen36b \
    --gpus all --net=host --ipc=host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${PROJECT_DIR}/configs/chat_template.jinja:/opt/unsloth.jinja:ro" \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    vllm-qwen36b-v2 \
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
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    -tp 1
