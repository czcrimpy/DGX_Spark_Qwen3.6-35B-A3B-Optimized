#!/bin/bash
# v2-hybrid config (OPTIONAL) — all Phase 3 optimizations + Hybrid INT4+FP8.
# Expected: ~115 tok/s average, ~128 tok/s peak on DGX Spark.
#
# ONLY +1.8% over v2 (Phase 3) — on 35B the shared_expert layers are too small
# (intermediate_size=512) for CUTLASS FP8 block-128 kernels to shine. On 122B
# this optimization delivers +9% but on 35B it's barely measurable.
#
# Requires: hybrid checkpoint built via `install.sh --hybrid` at
#           ~/models/qwen36b-hybrid-int4fp8/

PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
HYBRID_MODEL_DIR="${HYBRID_MODEL_DIR:-${HOME}/models/qwen36b-hybrid-int4fp8}"

if [ ! -f "${HYBRID_MODEL_DIR}/model.safetensors.index.json" ]; then
    echo "Error: hybrid checkpoint not found at ${HYBRID_MODEL_DIR}"
    echo "Run: ./install.sh --hybrid"
    exit 1
fi

docker run -d --name vllm-qwen36b \
    --gpus all --net=host --ipc=host \
    -v "${HOME}/models:/models" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${PROJECT_DIR}/configs/chat_template.jinja:/opt/unsloth.jinja:ro" \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    vllm-qwen36b-v2 \
    serve /models/qwen36b-hybrid-int4fp8 \
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
