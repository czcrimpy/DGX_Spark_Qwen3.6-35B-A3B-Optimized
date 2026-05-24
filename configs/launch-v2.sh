#!/bin/bash
# v2 config (RECOMMENDED) — INT8 LM Head + MTP-2 speculative + FlashInfer.
# Expected: ~113 tok/s average, ~125 tok/s peak on DGX Spark.
# No hybrid checkpoint needed — works with plain Intel INT4 AutoRound.

docker stop vllm-qwen36b
docker rm vllm-qwen36b
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run -d --name vllm-qwen36b \
    --gpus all --net=host --ipc=host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    vllm-qwen36b-v2 \
    serve Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    --served-model-name qwen/qwen3.6-35b-a3b \
    --port 8000 \
    --host 0.0.0.0 \
    --attention-backend FLASHINFER \
    --load-format fastsafetensors \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 3 \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --override-generation-config '{"max_new_tokens":8192, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.00, "repetition_penalty": 1.05}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    -tp 1
