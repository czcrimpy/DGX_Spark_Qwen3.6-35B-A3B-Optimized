# Phase 0 — Discovery Report

**Date:** 2026-04-14
**Goal:** Determine feasibility of porting albond's v2 optimizations from Qwen3.5-122B-A10B to Qwen3.6-35B-A3B.

## 1. Checkpoint Availability

| Checkpoint | Status | Notes |
|---|---|---|
| `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` | **EXISTS** | 23 likes, 27K downloads. 11 shards + `model_extra_tensors.safetensors` + `quantization_config.json` |
| `Qwen/Qwen3.6-35B-A3B-FP8` | **EXISTS** | 143 likes, 1.93M downloads. 14 shards |
| `Qwen/Qwen3.6-35B-A3B` (base BF16) | **EXISTS** | MTP tensors embedded inline |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | EXISTS | 39 likes |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | EXISTS | 17 likes |
| `Qwen/Qwen3.6-35B-A3B-GPTQ-Int4` | EXISTS (official) | 71 likes, 675K downloads |

**Conclusion:** Both checkpoints albond uses (Intel AutoRound + Qwen FP8) exist for 35B-A3B. Hybrid INT4+FP8 technique applies directly.

## 2. MTP (Multi-Token Prediction) Support

- Base `Qwen/Qwen3.6-35B-A3B` embeds MTP tensors **inline in main shards** (differs from 122B's separate file)
- Total 1811 tensors, **785 MTP tensors** (same count as 122B)
- Config has `mtp_num_hidden_layers: 1` and `mtp_use_dedicated_embeddings: false`
- Sample keys: `mtp.fc.weight`, `mtp.layers.0.self_attn.q_proj.weight`, `mtp.layers.0.mlp.experts.0.down_proj.weight`

**Intel INT4 repo** additionally ships `model_extra_tensors.safetensors` (AutoRound convention).
**FP8 repo** embeds MTP in shards (1560 tensors including FP8 `_scale_inv` pairs).

**Conclusion:** MTP-2 speculative decoding is portable. `add-mtp-weights.py` needs adaptation: tensor count is identical (785) but packaging differs between 122B and 35B.

## 3. Architecture Comparison

| Field | 35B-A3B | 122B-A10B | Ratio |
|---|---|---|---|
| num_hidden_layers | **40** | 62 | 0.65× |
| hidden_size | **2048** | 3072 | 0.67× |
| moe_intermediate_size | 512 | — | — |
| shared_expert_intermediate_size | 512 | — | — |
| num_experts | 256 | 256 | 1× |
| num_experts_per_tok | 8 | — | — |
| vocab_size | 248,320 | 248,320 | 1× |
| head_dim | 256 | — | — |
| num_attention_heads | 16 | — | — |
| num_key_value_heads | 2 (GQA) | — | — |
| max_position_embeddings | 262,144 | 262,144 | 1× |
| mtp_num_hidden_layers | 1 | 1 | 1× |
| layer_types | 3:1 linear:full | (same pattern) | — |

**Critical:** Hybrid attention pattern (3:1 linear:full) matches 122B. `Qwen3_5MoeForConditionalGeneration` architecture, multimodal (vision_config depth 27), `attn_output_gate: true`.

## 4. Portability Assessment

| Optimization | Portable? | Difficulty | Expected Gain |
|---|---|---|---|
| **FlashInfer backend** | YES | Trivial (flag) | +16% |
| **INT8 LM Head v2** | YES (likely drop-in) | Easy | +33% |
| **MTP-2 speculative** | YES | Medium (packaging diff) | +25% |
| **Hybrid INT4+FP8** | YES | Medium-High (both checkpoints exist) | +9% |
| **TurboQuant (optional)** | YES | Easy | — (speed trade for KV capacity) |

## 5. Caveats

1. **MTP packaging**: 122B uses separate `model_extra_tensors.safetensors`, 35B base embeds MTP in shards, Intel INT4 uses separate file again. Loader must handle both conventions.
2. **Kernel tile sizes**: hidden=2048 and 40 layers differ from 122B; kernels compile dynamically but performance verification required.
3. **Hybrid attention**: 3:1 linear:full pattern maintained → DeltaNet prefix caching crash likely recurs on 35B → keep prefix caching OFF.
4. **vLLM version**: Must match albond's pinned v0.19.0 for patch compatibility.
5. **Intel INT4 AutoRound** quantizes MTP to 4-bit (except `mtp.fc` kept fp16). Hybrid checkpoint script must respect this.

## 6. Decision

**PROCEED**: all materials present, port is technically feasible. Move to Phase 1 baseline.

## Sources

- https://huggingface.co/Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- albond repo: https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
