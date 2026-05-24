# Progress Log

## Phase 0 — Discovery ✅

**Done 2026-04-14.** See [research/phase0-findings.md](research/phase0-findings.md).

- Intel INT4 AutoRound checkpoint of 35B-A3B exists on HuggingFace ✅
- Qwen FP8 checkpoint of 35B-A3B exists ✅
- 785 MTP tensors present in model (same count as 122B) ✅
- Architecture compatible: 40 layers, hidden=2048, 256 experts, same DeltaNet 3:1 hybrid attention ✅

**Decision:** All four optimization layers are technically portable. Proceed.

## Phase 1 — Baseline

**Status:** In progress.

### Step 1a — Recipe file ✅

Created `configs/recipes/qwen3.6-35b-a3b-int4-autoround.yaml` matching albond's 122B baseline methodology:
- Intel AutoRound INT4 checkpoint
- FlashInfer attention backend
- Prefix caching OFF (DeltaNet conflict)
- Solo mode, `-tp 1`

### Step 1b — Model download

**Status:** In progress. `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` (~18 GB).

### Step 1c — Baseline benchmark ✅

Ran `bench_qwen36b.sh "baseline-intel-int4-autoround"` on 2026-04-14. Results in [benchmarks/01-baseline-intel-int4-autoround.txt](benchmarks/01-baseline-intel-int4-autoround.txt).

### Step 1d — Results ✅

| Scenario | Run 1 | Run 2 |
|---|---|---|
| Q&A (256 tok) | 54.5 | 65.8 |
| Code (512 tok) | 66.1 | 66.1 |
| JSON (1024 tok) | 64.6 | 66.1 |
| Math (64 tok) | 64.0 | 62.1 |
| LongCode (2048 tok) | 66.0 | 66.0 |

**Averages:** Run 1 = 63.0, Run 2 = 65.2, **Overall = 64.1 tok/s**.

**Comparison to albond's 122B baseline (28.3 tok/s):** 2.27× faster with 3.33× fewer active parameters (10B → 3B). ~68% of theoretical max, rest lost to memory bandwidth saturation on LPDDR5x.

## Phase 2 — INT8 LM Head v2 Port ✅

**Status:** Complete 2026-04-14.

### Hypothesis (confirmed)

Patch is a runtime monkey-patch targeting vLLM's ubiquitous `logits_processor._get_logits()`. Activation criteria: `lm_head.weight.dtype in (bfloat16, float16) and shape[0] > 100000`. Since Qwen3.6-35B has `vocab_size=248320` and Intel AutoRound keeps `lm_head` in BF16, the patch applies drop-in.

### Implementation

- `patches/03-int8-lm-head/patch_int8_lmhead.py` — verbatim copy from albond (Apache 2.0)
- `docker/entrypoint-v2.sh` — applies patch, then exec's `vllm`
- `docker/Dockerfile.v2` — thin layer over `vllm-sm121:latest`

Zero code changes needed. At startup the patch logs:
```
DGX_SPARK_V2: LM Head -> INT8 Batched Triton ([248320, 2048], saved 970MB)
```

### Results

Benchmark `v2-int8-lmhead` vs baseline:

| Scenario | Baseline | +INT8 LM Head | Gain |
|---|---|---|---|
| Q&A | 60.2 | 78.0 | +30% |
| Code | 66.1 | 85.6 | +29% |
| JSON | 65.4 | 84.4 | +29% |
| Math | 63.1 | 81.0 | +28% |
| LongCode | 66.0 | 85.3 | +29% |
| **Average** | **64.1** | **82.8** | **+29.2%** |

albond reported +33% on 122B; we hit +29.2% on 35B (expected: LM head is a smaller share of total compute on 35B since hidden_size is 2048 vs 3072).

Raw data: [benchmarks/02-int8-lmhead.txt](benchmarks/02-int8-lmhead.txt).

## Phase 3 — MTP-2 Speculative Port ✅

**Status:** Complete 2026-04-14.

### Key finding: script not needed!

Unlike the 122B Intel AutoRound checkpoint (which embeds MTP weights but does NOT reference them in the index), the 35B Intel AutoRound checkpoint **already has MTP tensors mapped in `model.safetensors.index.json`**. 2329 MTP entries, all pointing to `model_extra_tensors.safetensors`.

This means **no script adaptation is required** — the existing `add-mtp-weights.py` solves a problem Intel/35B doesn't have. Just enable speculative decoding in vLLM:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

### Startup confirmation

vLLM log on launch:
```
Resolved architecture: Qwen3_5MoeMTP
Detected MTP model. Sharing target model embedding weights with the draft model.
Detected MTP model. Sharing target model lm_head weights with the draft model.
DGX_SPARK_V2: LM Head -> INT8 Batched Triton ([248320, 2048], saved 970MB)
```

### Results

Benchmark `v2-int8-lmhead+mtp`:

| Scenario | Baseline | +INT8 LM Head | +INT8+MTP | Cumulative gain |
|---|---|---|---|---|
| Q&A | 60.2 | 78.0 | 89.9 | +49% |
| Code | 66.1 | 85.6 | 121.2 | +83% |
| JSON | 65.4 | 84.4 | 118.2 | +81% |
| Math | 63.1 | 81.0 | 109.4 | +73% |
| LongCode | 66.0 | 85.3 | **125.3** | **+90%** |
| **Average** | **64.1** | **82.8** | **112.8** | **+76%** |

Phase-on-phase gain from adding MTP: **+36.2%** (albond reported +25% for 122B).

Why bigger on 35B? 35B is more tightly memory-bound per token, so each correctly-predicted speculative token amortizes the weight-read cost across more output — the MTP acceptance window is where the real win lives.

Raw data: [benchmarks/03-int8-lmhead-mtp.txt](benchmarks/03-int8-lmhead-mtp.txt).

## Phase 4 — Hybrid INT4+FP8 Port ✅

**Status:** Complete 2026-04-14.

### Key finding #1: scripts are architecture-independent

Neither `build-hybrid-checkpoint.py` nor `inc.py` hardcode layer counts, hidden sizes, or expert counts. Both detect structure dynamically at runtime.

### Key finding #2 (critical bug): MTP weights lost during hybrid build!

First run showed catastrophic regression: 112 → 45 tok/s. Root cause diagnosed after looking at tensor counts: **`build-hybrid-checkpoint.py` strips the `model_extra_tensors.safetensors` file** from the output, removing all 2329 MTP tensors from the index (95815 → 93606 tensors).

vLLM still loaded MTP drafter (shared weights pattern) but with garbage/zero MTP-specific weights. Speculative decoding proposed tokens the main model rejected 100% of the time, actively hurting performance.

**Fix:** Run `add-mtp-weights.py` AFTER hybrid build to restore MTP tensors:
```bash
python add-mtp-weights.py \
  --source <Intel_INT4_dir> \
  --target /path/to/hybrid-checkpoint
```

This is how albond's install.sh handles it (hybrid → add-mtp-weights as sequential steps).

### Key finding #3: Hybrid gain is marginal on 35B

After fixing MTP:

| Config | Avg tok/s | Phase gain |
|---|---|---|
| + INT8 LM Head + MTP (Phase 3) | 112.8 | — |
| + Hybrid INT4+FP8 (Phase 4 fixed) | 114.8 | **+1.8%** |

albond reported +9% on 122B. We see +1.8% on 35B — roughly 5× less.

**Reason:** 35B's `shared_expert_intermediate_size = 512` vs 122B's much larger dense layers. The BF16→FP8 swap saves less bytes and has less per-layer amortization benefit on small matrices. CUTLASS FP8 block-128 kernels have fixed launch overhead that dominates when matrices are small.

### Recommendation for users

Hybrid is **optional and low-ROI** on 35B: 25 min extra build time + 0.5 GB extra disk for <2% speedup. Unless you're chasing every last tok/s, stop at Phase 3 (INT4 AutoRound + MTP + INT8 LM Head) which gives 98% of the benefit.

Raw data: [benchmarks/04-hybrid-fixed.txt](benchmarks/04-hybrid-fixed.txt).

## Phase 5 — Packaging

**Status:** Not started.

Deliverables:
- `install.sh` (automated Steps 0-4 pipeline)
- `docker/Dockerfile.v2` (thin layer over vLLM SM121 base)
- `bench_qwen36b.sh` (adapted benchmark)
- `configs/launch-*.sh` (reference launch commands for each config)
- README with full results table
- GitHub Actions CI (optional)
