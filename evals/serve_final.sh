#!/bin/bash
# Final validation: the winning config WITH the fine-tuned LoRA, then measure.
#
# Winning config (measured 11/09, base model, same droplet, one variable at a time):
#   AITER=0                    46.5 short /  4.8 @90k    <- what we shipped until now
#   AITER=1 + ROCM_AITER_FA    67.3 short / 60.3 @90k    <- WINNER (12.5x at long ctx)
#   AITER=1 + MTP              58.1 short / 52.1 @90k    <- MTP hurts once AITER is on
# so: AITER on, MTP OFF, 256k, FULL_DECODE_ONLY graphs.
#
# LoRA costs ~25% (measured earlier: 66.4 -> 49.5), so expect ~50 tok/s here.
# Usage: bash serve_final.sh
set -u
K=$(cat /root/vllm_key)
LOG=/root/logs/final.log
: > "$LOG"

echo "=== final config + LoRA $(date -u) ===" | tee -a "$LOG"
docker rm -f vllm-lora >/dev/null 2>&1
AITER=1 ATTN_BACKEND=ROCM_AITER_FA MAXLEN=262144 CUDAGRAPH=FULL_DECODE_ONLY \
  nohup bash /root/scripts2/serve_lora.sh > /root/logs/serve_final2.log 2>&1 &

for i in $(seq 1 45); do
  sleep 15
  if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then
    echo "  READY after $((i*15))s" | tee -a "$LOG"; break
  fi
done
curl -s -H "Authorization: Bearer $K" http://localhost:8000/v1/models \
  | tr ',' '\n' | grep -o '"id": *"[^"]*"' | grep -v modelperm | sed 's/.*: *"/  served: /;s/"$//' | tee -a "$LOG"

echo "--- LoRA realmente aplicada? (probe) ---" | tee -a "$LOG"
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
  /root/.unsloth/studio/unsloth_studio/bin/python /root/scripts2/probe_served_loras.py 2>&1 \
  | tail -10 | sed 's/^/  /' | tee -a "$LOG"

echo "--- decode rate com LoRA ---" | tee -a "$LOG"
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
  stdbuf -oL -eL /root/.unsloth/studio/unsloth_studio/bin/python /root/scripts2/measure_stream.py \
  qwen38-dxlam "0,90000" 2>&1 | awk '{print "  " $0; fflush()}' | tee -a "$LOG"
echo "=== done $(date -u) ===" | tee -a "$LOG"
