#!/bin/bash
# Deterministic quality comparison helper.
# Runs 5 fixed prompts at temperature=0.0 and saves outputs to a labelled dir.
# Usage: ./quality_diff.sh <label>
#        e.g.  ./quality_diff.sh v2
#              ./quality_diff.sh baseline

LABEL="${1:?usage: $0 <label>}"
API="http://127.0.0.1:8000/v1/chat/completions"
OUT_DIR="$(dirname "$0")/results_${LABEL}"
mkdir -p "$OUT_DIR"

PROMPTS=(
  "factual:What is photosynthesis? Answer in exactly 3 sentences."
  "code:Write a Python function called is_prime(n) that returns True if n is prime. Include docstring."
  "json:Output a JSON object with exactly these keys: name (string 'Alice'), age (integer 30), skills (array of 3 strings). No explanation, only JSON."
  "math:What is 1234 + 5678? Show only the final answer, nothing else."
  "creative:Write a haiku about a bug in production code. No explanation, just the haiku."
)

for entry in "${PROMPTS[@]}"; do
  name="${entry%%:*}"
  prompt="${entry#*:}"
  echo "  [${name}] ${prompt:0:60}..."

  # enable_thinking=false → compare final output only, not reasoning traces
  # which are long and amplify tiny probabilistic differences
  curl -s "$API" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"qwen\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${prompt//\"/\\\"}\"}],
      \"max_tokens\": 1000,
      \"temperature\": 0.0,
      \"seed\": 42,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['choices'][0]['message'].get('content') or '(empty)')
" > "$OUT_DIR/${name}.txt"
done

echo "Saved to: $OUT_DIR"
