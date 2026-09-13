#!/bin/bash
# Full BFCL v4 grid: 6 variants x 2 modes on the discriminating categories.
#
# Why these categories: they are exactly where our home-grown metric saturated
# (arguments, parallel calls, abstention). simple_python is excluded for now
# (easiest category, 400 entries, low discrimination per GPU-minute).
#
# Why both modes: prompt mode measures "follow THIS benchmark's dialect" and the
# FC mode measures native tool-calling. Our SFT taught a JSON-in-content dialect,
# so the gap between the two is itself the finding (see docs/FAILURES.md #7).
#
#   bash run_bfcl_grid.sh              # everything
#   MODELS="agent-native-qwen38-base" bash run_bfcl_grid.sh
#
# Cost estimate: 840 entries/run at ~1.0-1.2 entries/s -> ~13 min per run,
# 12 runs -> ~2.6-3h of MI300X.

set -u

export BFCL_PROJECT_ROOT=${BFCL_PROJECT_ROOT:-/root/bfcl-project}
export REMOTE_OPENAI_BASE_URL=${REMOTE_OPENAI_BASE_URL:-http://localhost:8000/v1}
export REMOTE_OPENAI_API_KEY=${REMOTE_OPENAI_API_KEY:-dummy}
export REMOTE_OPENAI_TOKENIZER_PATH=${REMOTE_OPENAI_TOKENIZER_PATH:-/root/models/qwen3.8-27b}
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}

BFCL=${BFCL:-/root/bfcl-env/bin/bfcl}
CATS=${CATS:-"multiple parallel parallel_multiple irrelevance"}
THREADS=${THREADS:-16}
MODELS=${MODELS:-"agent-native-qwen38-base agent-native-qwen38-oh agent-native-qwen38-xlam agent-native-qwen38-dbase agent-native-qwen38-doh agent-native-qwen38-dxlam"}

LOG="$BFCL_PROJECT_ROOT/grid.log"
mkdir -p "$BFCL_PROJECT_ROOT"
: > "$LOG"

echo "=== BFCL v4 grid $(date -u) ===" | tee -a "$LOG"
echo "categories: $CATS" | tee -a "$LOG"

# preflight
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$REMOTE_OPENAI_BASE_URL/models" 2>/dev/null)
if [ "$code" != "200" ]; then
  echo "!! endpoint returned $code -- aborting" | tee -a "$LOG"; exit 1
fi

run_mode () {
  local suffix="$1" label="$2"
  for model in $MODELS; do
    for cat in $CATS; do
      name="${model}${suffix}"
      echo "--- [$label] $name / $cat  $(date -u)" | tee -a "$LOG"
      timeout 2400 "$BFCL" generate --model "$name" --test-category "$cat" \
        --skip-server-setup --num-threads "$THREADS" > "/root/logs/bfcl_${name}_${cat}.log" 2>&1
      rc=$?
      if [ "$rc" != "0" ]; then
        echo "    !! generate rc=$rc ($name/$cat)" | tee -a "$LOG"
      fi
      "$BFCL" evaluate --model "$name" --test-category "$cat" --partial-eval \
        >> "/root/logs/bfcl_eval_${name}_${cat}.log" 2>&1
      score=$(grep -aoE '"accuracy": [0-9.]+' "$BFCL_PROJECT_ROOT/score/$name/non_live/BFCL_v4_${cat}_score.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+$')
      echo "    score: ${score:-NA}  ($name/$cat)" | tee -a "$LOG"
    done
  done
}

# FC mode first: native tool-calling is the right reading for our SFT dialect
run_mode "-FC" "FC/native"
# then prompt mode: measures the dialect degradation the SFT caused
run_mode "" "prompt"

echo "=== BFCL_GRID_DONE $(date -u) ===" | tee -a "$LOG"
