#!/bin/bash
# Kernel/backend matrix — the GPU session's experiment (see docs/optimization-plan-2026-09-11.md).
#
# WHY: our server runs with VLLM_ROCM_USE_AITER=0, a workaround carried over from the OLD
# image (load-time segfault) that was never re-tested on the current one. With AITER off
# vLLM's backend priority is [ROCM_ATTN, TRITON_ATTN, ...] — the "legacy" backend AMD
# measures at 2.7-4.4x below ROCM_AITER_FA. Our measured 46 tok/s vs the published 80-120+
# for this model in BF16 is almost exactly that gap.
#
# WHAT IT DOES NOT DO: the AITER GDN decode fast path is gated on `gqa_interleaved_layout`
# (qwen_gdn_linear_attn.py:1229) which is False for Qwen3.8, and the fix (vLLM PR #53623) is
# not in 0.29.0 nor merged. So do not expect AITER to speed up the 48 GDN layers.
#
# Matrix:
#   1  AITER=0, no MTP              control (reproduces yesterday's 46 / 4.8 tok/s)
#   2  AITER=1 + ROCM_AITER_FA, no MTP   pure AITER effect (the 2.7-4.4x claim)
#   3  AITER=1 + ROCM_AITER_FA + MTP     candidate final config
#   4  AITER=1 + TRITON_ATTN + MTP       plan B if ROCM_AITER_FA crashes at load
#
# Each row measures short + ~90k context, thinking OFF and ON, 2 reps, waiting for real idle.
# Usage: bash try_kernels.sh
set -u
K=$(cat /root/vllm_key)
LOG=/root/logs/kernels.log
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

MTP1='{"method":"qwen3_5_mtp","num_speculative_tokens":1}'

start_server() {
  local aiter="$1" attn="$2" spec="$3" tag="$4"
  docker rm -f vllm-lora >/dev/null 2>&1
  AITER="$aiter" ATTN_BACKEND="$attn" MAXLEN=262144 CUDAGRAPH=FULL_DECODE_ONLY SPEC_JSON="$spec" \
    nohup bash /root/scripts2/serve_lora.sh > "/root/logs/serve_${tag}.log" 2>&1 &
  for i in $(seq 1 45); do
    sleep 15
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then
      log "  READY after $((i*15))s"; return 0
    fi
  done
  log "  !! NOT READY after 675s -- last log lines:"
  docker logs vllm-lora 2>&1 | tail -10 | cut -c1-170 | tee -a "$LOG"
  return 1
}

backend_lines() {
  # which attention backend and GDN path did this config actually take?
  docker logs vllm-lora 2>&1 \
    | grep -aiE "attention backend|Using .* backend|GDN decode kernel|Falling back to the Triton GDN|AITER" \
    | tail -5 | cut -c1-175 | sed 's/^/    /' | tee -a "$LOG"
}

run_config() {
  local tag="$1" aiter="$2" attn="$3" spec="$4" desc="$5"
  log ""
  log "===== $tag: $desc ====="
  log "      AITER=$aiter  attn=${attn:-auto}  MTP=$([ -n "$spec" ] && echo on || echo off)"
  if start_server "$aiter" "$attn" "$spec" "$tag"; then
    backend_lines
    LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
      /root/.unsloth/studio/unsloth_studio/bin/python /root/scripts2/measure_stream.py \
      qwen38-dxlam "0,90000" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  fi
}

log "=== kernel/backend matrix $(date -u) ==="
run_config A 0 ""      ""     "controle: sem AITER, sem MTP (reproduz ontem)"
run_config B 1 "ROCM_AITER_FA" ""     "AITER + ROCM_AITER_FA, sem MTP (efeito puro)"
run_config C 1 "ROCM_AITER_FA" "$MTP1" "AITER + ROCM_AITER_FA + MTP (candidata final)"
run_config D 1 "TRITON_ATTN"   "$MTP1" "plano B: TRITON_ATTN + MTP"
log ""
log "=== done $(date -u) ==="
