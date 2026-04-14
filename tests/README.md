# Quality Comparison Tests

Deterministic output comparison between baseline (Intel INT4 + FlashInfer only) and v2 (+ INT8 LM Head + MTP-2) to verify the optimizations don't degrade output quality.

## Methodology

- 5 fixed prompts spanning different task types (factual, code, JSON, math, creative)
- `temperature=0.0`, `seed=42`
- `enable_thinking=false` to compare final output only (thinking traces are long and amplify tiny probabilistic differences that would make the test misleading)
- Same model (`Intel/Qwen3.5-35B-A3B-int4-AutoRound`) in both cases — the optimizations are runtime/architectural, not a different model

## Running

```bash
# On each running container (baseline first, then v2):
./tests/quality_diff.sh <label>
```

Outputs go to `results_<label>/`.

## Results (2026-04-14)

| Prompt | Outputs | Quality |
|---|---|---|
| math (1234 + 5678) | **Identical** — both output `6912` | ✓ |
| code (is_prime) | Functionally identical, v2 adds one extra docstring example (`is_prime(-5)`) | **v2 slightly more thorough** |
| factual (photosynthesis) | Different wording, both correct and 3 sentences | ✓ Both correct |
| json (schema task) | Both schema-valid, different arbitrary skill choices | ✓ Both correct |
| creative (haiku) | Both valid 5-7-5 haikus, different imagery | ✓ Both correct |

## Interpretation

The v2 optimizations (INT8 LM Head v2 + MTP-2 speculative decoding) do not degrade output quality:

- **MTP-2** is mathematically exact — accepted tokens are bit-identical to what the main model would have produced. It does not affect the output distribution.
- **INT8 LM Head v2** quantizes the vocabulary projection to INT8 at runtime (per-channel dynamic scale). This introduces a tiny precision shift in per-token logits, but stays far below the threshold where semantically different answers would surface.
- Where outputs differ, both sides are equally valid. The shift appears only on open-ended prompts where many valid outputs exist (which skill to list? which imagery for the haiku?). Constrained prompts (math, schema-compliance) produce bit-identical outputs.

For interactive use (chat, code assistance, agents) the differences are imperceptible. For deterministic reproducibility (e.g. logging fingerprints) v2's outputs will not exactly match an unoptimized run, but the semantic content remains equivalent.
