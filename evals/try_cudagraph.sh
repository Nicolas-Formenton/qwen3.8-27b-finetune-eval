#!/bin/bash
# Trial: can we serve with CUDA graphs (FULL_DECODE_ONLY) instead of --enforce-eager?
#
# PIECEWISE capture hangs this stack (docs/FAILURES.md #3), but FULL_DECODE_ONLY is a
# different code path: full graphs for DECODE only (what tok/s measures), NONE for prefill.
# If the engine does not answer within the window, fall back to --enforce-eager so the
# endpoint is never left down.
#
# Runs ON the droplet:  bash /root/scripts2/try_cudagraph.sh

set -u
LOG=/root/logs/cudagraph_trial.log
: > "$LOG"

bench () {
  local K="$1"
  for M in qwen38-base qwen38-dxlam; do
    for R in 1 2 3; do
      S=$(date +%s%N)
      curl -s -m 200 http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
        -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 1000, one per line.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":200,\"temperature\":0,\"ignore_eos\":true}" \
        > /tmp/cg.json
      E=$(date +%s%N)
      python3 -c "
import json
try:
    d=json.load(open('/tmp/cg.json'))
    n=d['usage']['completion_tokens']
    ms=(${E}-${S})/1e6
    print(f'  $M run$R: {n} tok in {ms/1000:6.2f}s -> {n/(ms/1000):5.1f} tok/s')
except Exception as e:
    print(f'  $M run$R: FAILED ({e})')
" | tee -a "$LOG"
    done
  done
}

wait_ready () {
  for i in $(seq 1 24); do          # 24 x 15s = 6 min
    sleep 15
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat /root/vllm_key)" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then
      echo "READY after $((i*15))s" | tee -a "$LOG"
      return 0
    fi
  done
  return 1
}

K_READY=0

echo "=== attempt 1: CUDAGRAPH=FULL_DECODE_ONLY ===" | tee -a "$LOG"
docker rm -f vllm-lora >/dev/null 2>&1
CUDAGRAPH=FULL_DECODE_ONLY nohup bash /root/scripts2/serve_lora.sh > /root/logs/serve_cg.log 2>&1 &
sleep 5

if wait_ready; then
  echo "--- graph capture lines ---" | tee -a "$LOG"
  docker logs vllm-lora 2>&1 | grep -aiE "captur|cudagraph" | tail -4 | cut -c1-160 | tee -a "$LOG"
  echo "--- bench (FULL_DECODE_ONLY) ---" | tee -a "$LOG"
  bench "$(cat /root/vllm_key)"
  K_READY=1
else
  echo "!! FULL_DECODE_ONLY did not come up in 6 min -- falling back" | tee -a "$LOG"
  docker logs vllm-lora 2>&1 | tail -6 | cut -c1-180 | tee -a "$LOG"
fi

if [ "$K_READY" = "1" ]; then
  echo "TRIAL_RESULT=GRAPHS_OK" | tee -a "$LOG"
else
  echo "=== attempt 2: fallback --enforce-eager ===" | tee -a "$LOG"
  docker rm -f vllm-lora >/dev/null 2>&1
  nohup bash /root/scripts2/serve_lora.sh > /root/logs/serve_fallback.log 2>&1 &
  if wait_ready; then
    echo "--- bench (enforce-eager) ---" | tee -a "$LOG"
    bench "$(cat /root/vllm_key)"
    echo "TRIAL_RESULT=FALLBACK_OK" | tee -a "$LOG"
  else
    echo "TRIAL_RESULT=BOTH_FAILED" | tee -a "$LOG"
  fi
fi

echo "=== done $(date -u) ===" | tee -a "$LOG"
