#!/bin/bash
# BFCL v4 matrix driver: base + 5 LoRA adapters, FC handler (native tool calls).
#
# Why FC: our models were fine-tuned on the Hermes <tool_call> dialect and vLLM runs with
# --tool-call-parser hermes. The prompt-mode registry entries score ~1.5% because BFCL's
# text template asks for [func_name1(...)] placeholders and the model copies them literally.
# The FC entries send tool schemas and parse <tool_call> blocks, which is what the models
# were trained to emit.
#
# One adapter at a time (load -> run -> unload) so the engine keeps a single LoRA resident.
#
# Usage: run_matrix.sh [categories]      (default: non_live,multi_turn)
set -u

CATS=${1:-non_live,multi_turn}
export BFCL_PROJECT_ROOT=${BFCL_PROJECT_ROOT:-/root/bfcl-project}
export REMOTE_OPENAI_BASE_URL=${REMOTE_OPENAI_BASE_URL:-http://localhost:8000/v1}
export REMOTE_OPENAI_API_KEY=${REMOTE_OPENAI_API_KEY:-$(cat /root/vllm_key)}
export REMOTE_OPENAI_TOKENIZER_PATH=${REMOTE_OPENAI_TOKENIZER_PATH:-/root/models/qwen3.8-27b}
export PATH=/root/bfcl-env/bin:$PATH
# Thinking MUST stay on for BFCL: measured on `irrelevance` (240 items), the same base model
# scores 69.17% with reasoning vs 17.08% without, because with reasoning suppressed it answers
# the question instead of declining to call a tool. Costs 3.4x wall time (2m15s vs 44s).
export BFCL_ENABLE_THINKING=${BFCL_ENABLE_THINKING:-1}
THREADS=${THREADS:-16}
SUMMARY=$BFCL_PROJECT_ROOT/matrix_summary.txt

# variant:served-name:adapter-dir   (base has no adapter)
VARIANTS="
base::-
oh:qwen38-oh:/root/adapters/oh-adapter
xlam:qwen38-xlam:/root/adapters/xlam-adapter
dbase:qwen38-dbase:/root/adapters/distill-base-adapter
doh:qwen38-doh:/root/adapters/doh-adapter
dxlam:qwen38-dxlam:/root/adapters/dxlam-adapter
"

echo "=== BFCL matrix $(date -u) cats=$CATS ===" | tee -a "$SUMMARY"

api() {  # api <method> <path> [body]
  curl -s -m 120 -X "$1" "$REMOTE_OPENAI_BASE_URL$2" \
    -H "Authorization: Bearer $REMOTE_OPENAI_API_KEY" -H "Content-Type: application/json" \
    ${3:+-d "$3"}
}

for entry in $VARIANTS; do
  slug=${entry%%:*}; rest=${entry#*:}
  served=${rest%%:*}; adapter=${rest#*:}
  reg="agent-native-qwen38-${slug}-FC"

  echo "" | tee -a "$SUMMARY"
  echo "########## $slug  ($(date -u)) ##########" | tee -a "$SUMMARY"

  if [ "$adapter" != "-" ]; then
    out=$(api POST /load_lora_adapter "{\"lora_name\":\"$served\",\"lora_path\":\"$adapter\"}")
    echo "  load: $out" | tee -a "$SUMMARY"
  fi

  for cat in ${CATS//,/ }; do
    echo "--- $reg / $cat $(date -u)" | tee -a "$SUMMARY"
    bfcl generate --model "$reg" --test-category "$cat" --skip-server-setup \
      --num-threads "$THREADS" >> "$BFCL_PROJECT_ROOT/matrix.log" 2>&1
    gen_rc=$?
    bfcl evaluate --model "$reg" --test-category "$cat" >> "$BFCL_PROJECT_ROOT/matrix.log" 2>&1
    eval_rc=$?
    acc=$(grep -ao 'Accuracy: [0-9.]*%' "$BFCL_PROJECT_ROOT/matrix.log" | tail -1)
    echo "    gen_rc=$gen_rc eval_rc=$eval_rc  $acc" | tee -a "$SUMMARY"
  done

  if [ "$adapter" != "-" ]; then
    out=$(api POST /unload_lora_adapter "{\"lora_name\":\"$served\"}")
    echo "  unload: $out" | tee -a "$SUMMARY"
  fi
done

echo "" | tee -a "$SUMMARY"
echo "=== MATRIX_DONE $(date -u) ===" | tee -a "$SUMMARY"
