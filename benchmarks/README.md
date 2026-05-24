# Benchmark Results

All runs use [bench_qwen36b.sh](../bench_qwen36b.sh): 5 prompts (Q&A, Code, JSON, Math, LongCode) × 2 runs at temperature 0.0. Hardware: single NVIDIA DGX Spark (GB10, SM121, 128 GB LPDDR5x).

| File | Config | Avg tok/s | Peak tok/s |
|---|---|---|---|
| [01-baseline-intel-int4-autoround.txt](01-baseline-intel-int4-autoround.txt) | Intel INT4 + FlashInfer | 64.1 | 66.1 |
| [02-int8-lmhead.txt](02-int8-lmhead.txt) | + INT8 LM Head v2 | 82.8 | 85.7 |
| [03-int8-lmhead-mtp.txt](03-int8-lmhead-mtp.txt) | **+ MTP-2 (recommended)** | **112.8** | 125.3 |
| [04a-hybrid-broken-no-mtp.txt](04a-hybrid-broken-no-mtp.txt) | Hybrid without MTP weights restore (BUG) | 45.0 | 45.5 |
| [04b-hybrid-fixed.txt](04b-hybrid-fixed.txt) | + Hybrid INT4+FP8 (fixed) | 114.8 | **128.1** |

## Notes on 04a (bug case, kept for historical/educational value)

`build-hybrid-checkpoint.py` drops Intel's `model_extra_tensors.safetensors` during the rewrite, losing all 2329 MTP tensors. vLLM's MTP drafter then loads with only the shared embedding/lm_head weights and garbage values elsewhere — producing ~0% draft acceptance and an active regression from 112 → 45 tok/s.

**Fix:** run `add-mtp-weights.py` after the hybrid build (see [install.sh](../install.sh)).

## Build log

[hybrid-build.log](hybrid-build.log) is the full stdout of `build-hybrid-checkpoint.py`. The "344 unexpected unmatched FP8 tensor" warnings are MTP-related and expected — those are the tensors whose matching INT4 counterparts live in `model_extra_tensors.safetensors`, which the script doesn't process.
