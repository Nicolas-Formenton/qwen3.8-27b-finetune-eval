#!/bin/bash
# Fix chain: runs AFTER the main n=200 chain finishes.
#  1) free the two invalid duplicate checkpoints (merged-xlam was bit-identical to base;
#     final-dxlam was trained on that broken base -> duplicate of final-dbase)
#  2) eval the corrected xLAM merge
#  3) retrain the stacked distill on the corrected xLAM base, merge it, eval it
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib
PY=/root/.unsloth/studio/unsloth_studio/bin/python
say(){ echo "=== $* $(date -u)"; }

for i in $(seq 1 200); do grep -q EVAL_ALL_DONE /root/eval_all.log 2>/dev/null && break; sleep 30; done
say MAIN_CHAIN_DONE

rm -rf /root/models/merged-xlam /root/models/final-dxlam
say FREED_DUPLICATES $(df -h / | awk 'NR==2{print $4}')

run_one () {
  local label="$1" path="$2"
  say "$label START"
  docker rm -f vllm-eval >/dev/null 2>&1
  docker run -d --name vllm-eval \
    --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --shm-size 16G \
    -p 8000:8000 -e VLLM_ROCM_USE_AITER=0 -v /root/models:/models \
    vllm/vllm-openai-rocm:latest \
    --model "$path" --served-model-name m qwen3.8-27b \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager > /dev/null 2>&1
  local ok=0
  for i in $(seq 1 30); do
    sleep 15
    [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models 2>/dev/null)" = "200" ] && { ok=1; break; }
  done
  if [ "$ok" = "1" ]; then
    $PY /root/scripts/eval_suite.py "$label" > /root/eval_${label}_stdout.log 2>&1
    echo "--- suite($label): $(grep -aoE '\"passed\": [0-9]+' /root/eval_${label}_stdout.log | head -1)"
    $PY /root/scripts/xlam_heldout_200.py "$label" > /root/xlam200_${label}_stdout.log 2>&1
    echo "--- xlam200($label): $(grep -aoE '\"accuracy\": [0-9.]+' /root/xlam200_${label}_stdout.log | tail -1)"
    $PY /root/scripts/bench_vllm2.py > /root/bench_${label}.json 2>&1
    echo "--- bench($label): $(tail -1 /root/bench_${label}.json)"
  else
    echo "!!! $label NOT READY"
  fi
  docker rm -f vllm-eval >/dev/null 2>&1
  say "$label DONE"
}

run_one xlamfix /models/merged-xlam-fixed

say RETRAIN_DXLAMFIX
$PY /root/scripts/train_distill_unsloth.py /root/models/merged-xlam-fixed dxlamfix 240 1500 > /root/retrain_dxlamfix.log 2>&1
AD=$(ls -d /root/agent-native-distill-dxlamfix-*-adapter 2>/dev/null | tail -1)
say MERGING_DXLAMFIX "$AD"
$PY /root/scripts/manual_merge.py /root/models/merged-xlam-fixed "$AD" /root/models/final-dxlam-fixed > /root/merge_dxlamfix.log 2>&1
grep -aE "touched|MERGE_DONE" /root/merge_dxlamfix.log | tail -2

run_one dxlamfix /models/final-dxlam-fixed
say FIX_CHAIN_DONE
