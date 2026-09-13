#!/bin/bash
# AITER backend matrix on the BASE model (no LoRA needed).
#
# Why base-only: the fine-tuned adapter uploads at ~90 KB/s (465 MB -> ~85 min) and the
# AITER question does not depend on it. LoRA costs a known ~25% (66.4 -> 49.5 tok/s), so
# base numbers are directly comparable with yesterday's base numbers (14.8 eager / 66-67
# graphs / 103-118 with MTP).
#
# Question: our server runs VLLM_ROCM_USE_AITER=0 (a workaround from the OLD image). With
# AITER off vLLM picks [ROCM_ATTN, TRITON_ATTN, ...] = the "legacy" backend AMD measures at
# 2.7-4.4x below ROCM_AITER_FA. Does enabling AITER recover that, and does it help at 90k
# context (where we measured 46 -> 4.8 tok/s)?
#
# Usage: bash try_aiter.sh
set -u
K=$(cat /root/vllm_key)
LOG=/root/logs/aiter_matrix.log
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

start_and_measure() {
  local tag="$1" aiter="$2" attn="$3" desc="$4"
  log ""
  log "===== $tag: $desc ====="
  log "      AITER=$aiter  attn=${attn:-auto}  LORAS=none"
  docker rm -f vllm-lora >/dev/null 2>&1
  LORAS="" AITER="$aiter" ATTN_BACKEND="$attn" MAXLEN=262144 CUDAGRAPH=FULL_DECODE_ONLY \
    nohup bash /root/scripts2/serve_lora.sh > "/root/logs/serve_${tag}.log" 2>&1 &
  local ready=0
  for i in $(seq 1 45); do
    sleep 15
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then
      log "  READY after $((i*15))s"; ready=1; break
    fi
  done
  if [ "$ready" != "1" ]; then
    log "  !! NOT READY after 675s -- tail:"
    docker logs vllm-lora 2>&1 | tail -8 | cut -c1-170 | sed 's/^/    /' | tee -a "$LOG"
    return 1
  fi
  # Which backend did it actually choose? This is the whole point of the experiment.
  docker logs vllm-lora 2>&1 \
    | grep -aiE "Using .*backend|attention backend|ROCM_AITER|ROCM_ATTN|TRITON_ATTN|GDN decode kernel" \
    | tail -4 | cut -c1-165 | sed 's/^/    /' | tee -a "$LOG"
  LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
    stdbuf -oL -eL /root/.unsloth/studio/unsloth_studio/bin/python /root/scripts2/measure_stream.py \
    qwen38-base "0,90000" 2>&1 \
    | awk '{print "  " $0; fflush()}' | tee -a "$LOG"
}

log "=== AITER matrix (base model) $(date -u) ==="
start_and_measure A 0 ""              "controle: AITER off (nosso estado atual)"
start_and_measure B 1 "ROCM_AITER_FA" "AITER on + ROCM_AITER_FA (recomendado pela AMD)"
log ""
log "=== done $(date -u) ==="
