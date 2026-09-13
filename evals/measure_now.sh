#!/bin/bash
# Measure the CURRENTLY RUNNING server (no restart). Records which config it is.
#   bash measure_now.sh <label> [model] [ctxs]
set -u
LABEL=${1:-unknown}
MODEL=${2:-qwen38-base}
CTXS=${3:-"0,90000"}
K=$(cat /root/vllm_key)
LOG=/root/logs/measure_${LABEL}.log

echo "=== measuring $LABEL (model=$MODEL, ctx=$CTXS) $(date -u) ===" | tee "$LOG"
# what is actually running: aiter env + attention backend + spec config
docker inspect vllm-lora --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
  | grep -E "VLLM_ROCM_USE_AITER" | sed 's/^/  env: /' | tee -a "$LOG"
docker inspect vllm-lora --format '{{range .Args}}{{println .}}{{end}}' 2>/dev/null \
  | grep -A1 -E "attention-backend|speculative-config" | paste - - | sed 's/^/  arg: /' | tee -a "$LOG"
docker logs vllm-lora 2>&1 | grep -aiE "GDN decode kernel|Using .*backend for ViT|Overriding with" \
  | tail -3 | cut -c1-150 | sed 's/^/  log: /' | tee -a "$LOG"

LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
  stdbuf -oL -eL /root/.unsloth/studio/unsloth_studio/bin/python /root/scripts2/measure_stream.py \
  "$MODEL" "$CTXS" 2>&1 | awk '{print "  " $0; fflush()}' | tee -a "$LOG"
echo "=== done $(date -u) ===" | tee -a "$LOG"
