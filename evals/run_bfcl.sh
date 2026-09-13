#!/bin/bash
# Run BFCL v4 against the vLLM server started by serve_lora.sh.
#
#   MODELS="agent-native-qwen38-base agent-native-qwen38-oh" CATS="non_live" bash run_bfcl.sh
#   SUFFIX=-FC ...   # native tool-call template instead of prompt mode
#
# Skips server setup: we own the endpoint, so the LoRA modules stay addressable
# by served name (the registry entries map registry name -> served name).

set -u

export BFCL_PROJECT_ROOT=${BFCL_PROJECT_ROOT:-/root/bfcl-project}
export LOCAL_SERVER_ENDPOINT=${LOCAL_SERVER_ENDPOINT:-localhost}
export LOCAL_SERVER_PORT=${LOCAL_SERVER_PORT:-8000}
export REMOTE_OPENAI_BASE_URL=${REMOTE_OPENAI_BASE_URL:-http://localhost:8000/v1}
export REMOTE_OPENAI_API_KEY=${REMOTE_OPENAI_API_KEY:-dummy}
# a LoRA module name is not a resolvable HF repo id, so the tokenizer is loaded locally
export REMOTE_OPENAI_TOKENIZER_PATH=${REMOTE_OPENAI_TOKENIZER_PATH:-/root/models/qwen3.8-27b}

export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
PY=${PY:-/root/.unsloth/studio/unsloth_studio/bin/python}
BFCL=${BFCL:-bfcl}
THREADS=${THREADS:-16}
SUFFIX=${SUFFIX:-}

MODELS=${MODELS:-"agent-native-qwen38-base agent-native-qwen38-oh agent-native-qwen38-xlam agent-native-qwen38-dbase agent-native-qwen38-doh agent-native-qwen38-dxlam"}
CATS=${CATS:-"non_live live multi_turn"}

mkdir -p "$BFCL_PROJECT_ROOT"
LOG="$BFCL_PROJECT_ROOT/run_bfcl.log"
: > "$LOG"

echo "=== BFCL v4 run $(date -u) ===" | tee -a "$LOG"
echo "models: $MODELS" | tee -a "$LOG"
echo "categories: $CATS  suffix='${SUFFIX}'" | tee -a "$LOG"

# preflight: the API must actually answer, not just have started
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$REMOTE_OPENAI_BASE_URL/models" 2>/dev/null)
if [ "$code" != "200" ]; then
  echo "!! endpoint $REMOTE_OPENAI_BASE_URL/models returned $code -- aborting" | tee -a "$LOG"
  exit 1
fi
curl -s "$REMOTE_OPENAI_BASE_URL/models" | tr ',' '\n' | grep -o '"id": *"[^"]*"' | sed 's/.*: *"/served: /;s/"$//' | tee -a "$LOG"

for model in $MODELS; do
  reg="${model}${SUFFIX}"
  for cat in $CATS; do
    echo "--- $reg / $cat  $(date -u)" | tee -a "$LOG"
    $BFCL generate --model "$reg" --test-category "$cat" \
      --skip-server-setup --num-threads "$THREADS" >> "$LOG" 2>&1
    gen_rc=$?
    $BFCL evaluate --model "$reg" --test-category "$cat" >> "$LOG" 2>&1
    eval_rc=$?
    acc=$(grep -ao 'Accuracy: [0-9.]*%' "$LOG" | tail -1)
    echo "    generate_rc=$gen_rc evaluate_rc=$eval_rc  $acc" | tee -a "$LOG"
    if [ "$gen_rc" != "0" ]; then
      echo "    !! generation failed for $reg/$cat -- check the log tail above" | tee -a "$LOG"
    fi
  done
done

echo "=== summary ===" | tee -a "$LOG"
for f in "$BFCL_PROJECT_ROOT"/score/*/*/*_score.json; do
  [ -f "$f" ] || continue
  printf '%s  %s\n' "$(head -1 "$f")" "${f#"$BFCL_PROJECT_ROOT"/score/}" | tee -a "$LOG"
done
echo "=== BFCL_DONE $(date -u) ===" | tee -a "$LOG"
