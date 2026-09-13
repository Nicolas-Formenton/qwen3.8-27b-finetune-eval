#!/bin/bash
# MTP trial v2 — fixes the two bugs that broke v1:
#   1. the --speculative-config flag was dropped by shell word-splitting (now an array)
#   2. the 90k-context measurement passed a huge prompt through argv ("Argument list too
#      long"); context curves are now measured inside Python (measure_ctx.py)
# Also stops wasting 5 min: wait_idle treats a fresh server (no throughput lines yet) as idle.
#
# Usage: bash try_mtp2.sh
set -u
K=$(cat /root/vllm_key)
LOG=/root/logs/mtp2.log
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

wait_instance() {
  for i in $(seq 1 45); do
    sleep 15
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then
      log "  READY after $((i*15))s"; return 0
    fi
  done
  log "  !! NOT READY after 675s"; docker logs vllm-lora 2>&1 | tail -10 | cut -c1-170 | tee -a "$LOG"
  return 1
}

start_server() {
  local tag="$1" spec="$2"
  docker rm -f vllm-lora >/dev/null 2>&1
  MAXLEN=262144 CUDAGRAPH=FULL_DECODE_ONLY SPEC_JSON="$spec" \
    nohup bash /root/scripts2/serve_lora.sh > "/root/logs/serve_${tag}.log" 2>&1 &
  wait_instance
}

measure_ctx() {
  local label="$1"
  LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
    /root/.unsloth/studio/unsloth_studio/bin/python /root/scripts2/measure_ctx.py qwen38-dxlam \
    "0,40000,90000,130000" 2>&1 | tee -a "$LOG"
}

log "=== MTP trial v2 $(date -u) ==="

log "=== attempt WITH MTP (qwen3_5_mtp, 1 spec token) ==="
if start_server "mtp" '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'; then
  docker logs vllm-lora 2>&1 | grep -aiE "speculat|mtp|draft" | tail -6 | cut -c1-170 | tee -a "$LOG"
  measure_ctx "mtp"
else
  log "!! MTP failed to serve (see serve_mtp.log)"
fi

log "=== done $(date -u) ==="
