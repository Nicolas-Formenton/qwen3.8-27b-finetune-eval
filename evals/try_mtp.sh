#!/bin/bash
# Trial: MTP (multi-token prediction) speculative decoding for Qwen3.8-27B on ROCm.
#
# Why: the checkpoint ships a built-in MTP head (15 `mtp.*` tensors, mtp_num_hidden_layers=1).
# vLLM auto-resolves `model_type: qwen3_5` -> `qwen3_5_mtp` with architectures Qwen3_5MTP
# (see vllm/config/speculative.py), so the head can be used as a speculative drafter
# without a separate draft model. Published figures for this model with MTP: 140+ tok/s
# single-stream. Ours measured 52.6 tok/s (FULL_DECODE_ONLY, no MTP).
#
# This script ALSO finally answers the context question, because the previous attempt was
# contaminated by the user's own session running concurrently (the engine showed
# "Running: 2 reqs"). It waits for a genuinely idle GPU, then measures decode at small
# vs large (90k) context for both MTP on and off.
#
# Usage: bash try_mtp.sh

set -u
K=$(cat /root/vllm_key)
LOG=/root/logs/mtp_trial.log
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

wait_idle() {
  # Wait for the engine to report 0 running requests, so measurements are clean.
  local tries=${1:-40}
  for i in $(seq 1 "$tries"); do
    local running
    running=$(docker logs vllm-lora 2>&1 | grep -a "Engine 000: Avg" | tail -1 | grep -oE "Running: [0-9]+" | grep -oE "[0-9]+")
    if [ "${running:-1}" = "0" ]; then echo "  GPU idle after $((i*15))s"; return 0; fi
    sleep 15
  done
  echo "  !! GPU never went idle (${tries} tries) -- measurements will be contaminated"
  return 1
}

measure() {
  # $1 label  $2 model  $3 target prompt tokens (0 = short)  $4 max_tokens
  local label="$1" model="$2" want_tok="$3" mt="$4"
  local prompt='Count from 1 to 1000, one per line.'
  if [ "$want_tok" -gt 1000 ]; then
    # repeat filler until we reach roughly the requested token count (~1.3 tok/word)
    prompt=$(python3 -c "
w = 'the quick brown fox jumps over the lazy dog and then continues walking through the forest'
n = int($want_tok / 1.3)
print((w + ' ') * n + '\n\nNow count from 1 to 1000, one per line.')")
  fi
  local t0 t1 out
  t0=$(date +%s.%N)
  out=$(curl -s -m 900 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
    -d "$(python3 - "$model" "$prompt" "$mt" <<'PY'
import json, sys
model, prompt, mt = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": mt, "temperature": 0, "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
)")
  t1=$(date +%s.%N)
  python3 - "$label" "$out" "$t0" "$t1" <<'PY' | tee -a "$LOG"
import json, sys
label, raw, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
try:
    d = json.loads(raw)
    u = d["usage"]
    pt, ct = u["prompt_tokens"], u["completion_tokens"]
    dt = t1 - t0
    print(f"  {label:22s} prompt={pt:6d} out={ct:4d} {dt:6.2f}s -> {ct/dt:5.1f} tok/s")
except Exception as e:
    print(f"  {label:22s} ERROR {e} :: {raw[:160]}")
PY
}

run_trial() {
  local tag="$1" spec="$2"
  log "=== attempt $tag (spec=${spec:-none}) ==="
  docker rm -f vllm-lora >/dev/null 2>&1
  MAXLEN=262144 CUDAGRAPH=FULL_DECODE_ONLY EXTRA_ARGS="$spec" nohup bash /root/scripts2/serve_lora.sh \
    > "/root/logs/serve_${tag}.log" 2>&1 &
  for i in $(seq 1 40); do
    sleep 15
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then
      log "  READY after $((i*15))s"
      break
    fi
  done
  if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://localhost:8000/v1/models 2>/dev/null)" != "200" ]; then
    log "  !! NOT READY -- last log lines:"; docker logs vllm-lora 2>&1 | tail -12 | cut -c1-180 | tee -a "$LOG"
    return 1
  fi
  wait_idle 20 || true
  measure "${tag} short" qwen38-dxlam 0 200
  measure "${tag} 90k-ctx" qwen38-dxlam 90000 200
  return 0
}

log "=== MTP trial $(date -u) ==="
run_trial "nomtp" "" || log "!! attempt without MTP failed"
run_trial "mtp1"  '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}' || log "!! MTP attempt failed"
log "=== done $(date -u) ==="
