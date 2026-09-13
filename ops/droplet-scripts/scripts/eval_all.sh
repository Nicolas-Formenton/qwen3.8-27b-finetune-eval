#!/bin/bash
# Sequential eval: serve each model with vLLM (docker, ROCm) -> suite20 -> xlam n=200 -> bench
# --enforce-eager is REQUIRED on this stack: PIECEWISE cudagraph capture hangs the engine.
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib
PY=/root/.unsloth/studio/unsloth_studio/bin/python
N=200

run_one () {
  local label="$1"; local path="$2"
  echo "=== $label START $(date -u) ==="
  docker rm -f vllm-eval >/dev/null 2>&1
  docker run -d --name vllm-eval \
    --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --shm-size 16G \
    -p 8000:8000 -e VLLM_ROCM_USE_AITER=0 \
    -v /root/models:/models \
    vllm/vllm-openai-rocm:latest \
    --model "$path" --served-model-name m qwen3.8-27b \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager > /dev/null 2>&1
  local ready=0
  for i in $(seq 1 30); do
    sleep 15
    if [ "$(curl -s -m 4 -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; then ready=1; break; fi
  done
  if [ "$ready" != "1" ]; then
    echo "!!! $label NOT READY after 450s -- skipping evals"
    docker logs vllm-eval 2>&1 | tail -6 | cut -c1-200 > /root/vllm_${label}_FAIL.log
  else
    $PY /root/scripts/eval_suite.py "$label" > /root/eval_${label}_stdout.log 2>&1
    echo "--- suite($label): $(grep -aoE '\"passed\": [0-9]+' /root/eval_${label}_stdout.log | head -1) /20"
    $PY /root/scripts/xlam_heldout_200.py "$label" $N 12 > /root/xlam200_${label}_stdout.log 2>&1
    echo "--- xlam200($label): $(tail -1 /root/xlam200_${label}_stdout.log | cut -c1-220)"
    $PY /root/scripts/bench_vllm2.py > /root/bench_${label}.json 2>&1
    echo "--- bench($label): $(tail -1 /root/bench_${label}.json)"
  fi
  docker logs vllm-eval 2>&1 | tail -3 | cut -c1-200 > /root/vllm_${label}_tail.log
  docker rm -f vllm-eval >/dev/null 2>&1
  echo "=== $label DONE $(date -u) ==="
}

run_one base  /models/qwen3.8-27b
run_one oh    /models/merged-oh
run_one xlam  /models/merged-xlam
run_one dbase /models/final-dbase
run_one doh   /models/final-doh
run_one dxlam /models/final-dxlam
echo "EVAL_ALL_DONE"
