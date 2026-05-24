# DGX Spark · Qwen3.6-35B-A3B Optimized

> **Qwen3.6-35B-A3B on a single DGX Spark: 64 → 113-115 tok/s (+77%)**
>
> Port of [albond's v2 optimizations](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) from Qwen3.5-122B-A10B down to the smaller Qwen3.6-35B-A3B variant, with measurements for each optimization layer and one critical bug fix.

## Results

Benchmarked on a single NVIDIA DGX Spark (GB10, SM121, 128 GB LPDDR5x) running [albond's bench_qwen35.sh](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/blob/master/bench_qwen35.sh) (5 prompts × 2 runs, temperature=0.0):

| Configuration | Avg tok/s | Peak tok/s | Phase gain | vs. baseline |
|---|---|---|---|---|
| Baseline (Intel INT4 AutoRound + FlashInfer) | 64.1 | 66.1 | — | — |
| + INT8 LM Head v2 | 82.8 | 85.7 | +29.2% | +29% |
| + MTP-2 speculative decoding (**recommended**) | **112.8** | 125.3 | +36.2% | +76% |
| + Hybrid INT4+FP8 (optional, low-ROI) | 114.8 | **128.1** | +1.8% | +79% |

*Trade-off note:* these are speed benchmarks. Qwen3.6-35B-A3B has 3B active parameters and is aimed at memory-bandwidth-limited hardware. Simple code/chat/summary quality is strong; long complex synthesis (writing a full game from scratch, multi-file refactors) benefits from larger models. Use 35B for speed-sensitive interactive work; fall back to Qwen3.5-122B-A10B for complexity-heavy tasks.

## Quick Start

```bash
git clone https://github.com/czcrimpy/DGX_Spark_Qwen3.6-35B-A3B-Optimized.git
cd DGX_Spark_Qwen3.6-35B-A3B-Optimized
./install.sh --launch
```

Wait ~3-4 min for model load, then:
```bash
curl http://127.0.0.1:8000/v1/models
```

For the optional hybrid config (+1.8% speed, +25 min install, +35 GB disk):
```bash
./install.sh --hybrid --launch
```

## Hardware Requirements

- **System:** NVIDIA DGX Spark (ASUS Ascent GX10 and similar)
- **GPU:** NVIDIA GB10 (Blackwell, SM121)
- **Memory:** 128 GB unified CPU-GPU (LPDDR5x, 273 GB/s)
- **CUDA:** 13.0+
- **Disk:** ~40 GB free for default, ~80 GB free if building hybrid
- **Arch:** aarch64 (ARM Grace CPU)

## What This Ports

Three of albond's four optimization layers, adapted for 35B:

1. **FlashInfer attention backend** — SM121-tuned kernels, same flag as albond.
2. **INT8 LM Head v2** — runtime per-channel INT8 quantization of the 248K×2048 logits layer. Worked drop-in on 35B (patch is model-agnostic — activates for any `lm_head.dtype ∈ {bf16, fp16}` with `vocab_size > 100000`).
3. **MTP-2 speculative decoding** — Qwen3.6-35B ships with native 785-tensor MTP head same as 122B. No script changes needed; just add `--speculative-config`.
4. **Hybrid INT4+FP8 dense layers** *(optional, marginal gain)* — replaces BF16 shared-expert MLPs with FP8 from Qwen's official FP8 checkpoint. See below for why the gain is smaller on 35B.

## Key Findings

### Finding 1 — Three of four optimizations are architecture-independent

Neither `build-hybrid-checkpoint.py` nor `inc.py` nor `patch_int8_lmhead.py` hardcode layer counts or names. All three detect model structure dynamically. **The INT8 LM Head patch literally applied with zero code changes.**

### Finding 2 — Critical bug: hybrid build strips MTP weights

`build-hybrid-checkpoint.py` only processes files matching `model-*-of-*.safetensors`. It silently drops Intel AutoRound's `model_extra_tensors.safetensors` (which holds all 2329 MTP tensors for 35B), rebuilds the index without them, and vLLM then loads the MTP drafter **with garbage weights** (only the shared embedding/lm_head get actual values).

Symptom: catastrophic regression from 112 → 45 tok/s with no obvious error (the drafter "works" but accepts 0% of its own proposals, so speculative decoding actively hurts).

Fix: always run `add-mtp-weights.py` after `build-hybrid-checkpoint.py`. albond's install.sh does this automatically for 122B; we encode the same sequence in our `install.sh`.

### Finding 3 — Hybrid INT4+FP8 gain scales with model size

| Model | `shared_expert_intermediate_size` | Reported/measured hybrid gain |
|---|---|---|
| Qwen3.5-122B-A10B | larger (exact size not published) | **+9%** (albond) |
| Qwen3.6-35B | 512 | **+1.8%** (this work) |

CUTLASS FP8 block-128 kernels have fixed per-launch overhead. When shared-expert matrices are only 512-wide, the kernel is launched on 128×128 tiles with little to amortize. Marlin INT4 kernels — the fallback used for everything in Phase 3 — handle the small-matrix path better on GB10.

**Takeaway for deployers:** skip `--hybrid` on 35B. The extra 25 min of build time, 35 GB of disk for the Qwen FP8 download, and 20 GB for the output checkpoint buy you less than 2 tok/s.

### Finding 4 — MTP acceptance rate peaks on code, not prose

Measured during live generation (vLLM's `SpecDecoding metrics` log):

| Workload | Draft acceptance rate |
|---|---|
| Tetris-like code generation (highly patterned) | 92-98% |
| JSON / structured output | 92-95% |
| Math reasoning | 80-85% |
| Free-form prose | 78-90% |

This matches albond's 122B measurements. The 3B active-param draft model predicts 2 tokens ahead correctly ~90% of the time when the next tokens are syntactically constrained (closing brackets, repeated keywords, template fills).

## Repository Layout

```
.
├── install.sh                    — automated build pipeline
├── bench_qwen36b.sh              — benchmark script (5 scenarios × 2 runs)
├── configs/
│   ├── launch-baseline.sh        — ~64 tok/s, unoptimized
│   ├── launch-v2.sh              — ~113 tok/s (recommended)
│   └── launch-v2-hybrid.sh       — ~115 tok/s (optional hybrid)
├── docker/
│   ├── Dockerfile.v2             — thin layer over vllm-sm121:latest
│   └── entrypoint-v2.sh          — applies INT8 LM Head patch at startup
├── patches/
│   ├── 01-hybrid-int4-fp8/       — hybrid checkpoint builder + inc.py overlay
│   ├── 02-mtp-speculative/       — MTP weight reattach script
│   └── 03-int8-lm-head/          — runtime INT8 LM Head patch (unmodified albond)
├── benchmarks/                   — raw bench output per phase
├── research/                     — discovery notes for each phase
├── PROGRESS.md                   — phase-by-phase build log
└── LICENSE / NOTICE              — Apache 2.0 + attribution
```

## Configuration Notes

- **Prefix caching must stay OFF.** Qwen3.6's DeltaNet-style hybrid attention (3:1 linear:full) conflicts with prefix caching and crashes the engine. This affects both 122B and 35B; we verified 35B exhibits the same issue.
- **`--reasoning-parser qwen3`** separates the model's chain-of-thought from the final reply (accessible via the `reasoning` field in the response JSON). Some clients (Cline, some Open WebUI versions) don't know to read that field and will show an empty `content`. Drop the flag if your client is one of those — the thinking will then appear inline inside `<think>...</think>` tags.
- **`--tool-call-parser qwen3_xml`** is the correct parser for Qwen3.6's native XML-style tool call format. `hermes` (common online answer) does not work for Qwen3.6.
- Container uses ~125 GB of unified memory with `gpu-memory-utilization 0.90` at 256K context — most of that is KV cache pre-allocation, not model weights.

## Credits

- **[albond](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** — original 122B v2 optimization stack.
- **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** — SM121-compiled vLLM base image pipeline.
- **Intel** — AutoRound INT4 checkpoint.
- **Qwen** — base model and FP8 checkpoint.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
