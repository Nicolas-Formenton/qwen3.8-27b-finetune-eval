#!/bin/bash
# Clean decode measurement: wait until the GPU has NO running requests, then measure.
#
# Why: the engine's "Avg generation throughput" divides by the logging window, so it
# under-reports (idle time included); and any concurrent request (another chat, a benchmark)
# contends for the batch. Earlier numbers were taken with 2 reqs running, so they cannot
# separate "long context is slow" from "another request is sharing the GPU".
#
# Prints decode tok/s for a SHORT and a LONG prompt, each measured while idle.

set -u
K=$(cat /root/vllm_key)
PY=/root/.unsloth/studio/unsloth_studio/bin/python
RUNLOG=/root/logs/measure_clean.log
: > "$RUNLOG"

idle_wait () {
  for i in $(seq 1 60); do
    R=$(docker logs vllm-lora 2>&1 | grep -a "Engine 000: Avg" | tail -1 | grep -oE "Running: [0-9]+" | grep -oE "[0-9]+")
    if [ "${R:-0}" = "0" ]; then
      # confirm with a second consecutive idle sample (15s apart)
      sleep 15
      R2=$(docker logs vllm-lora 2>&1 | grep -a "Engine 000: Avg" | tail -1 | grep -oE "Running: [0-9]+" | grep -oE "[0-9]+")
      [ "${R2:-0}" = "0" ] && return 0
    fi
    sleep 10
  done
  return 1
}

measure () {
  local label="$1" units="$2"
  $PY - "$label" "$units" <<'EOF' 2>&1 | tee -a "$RUNLOG"
import sys, time, json
from openai import OpenAI
label, units = sys.argv[1], int(sys.argv[2])
KEY = open("/root/vllm_key").read().strip()
c = OpenAI(base_url="http://localhost:8000/v1", api_key=KEY)
UNIT = ("Distributed systems background notes: consistency, quorums, leases, and failure "
        "detectors are described here purely to pad the prompt to a target length. ")
prompt = UNIT * units + "\n\nCount from 1 upward, one number per line, no commentary."
def ask(mt):
    t0 = time.time()
    r = c.chat.completions.create(model="qwen38-dxlam",
        messages=[{"role": "user", "content": prompt}], max_tokens=mt, temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True})
    return time.time() - t0, r.usage
ask(1); ask(1)                       # warm prefix cache
ta, ua = ask(1)
tb, ub = ask(200)
dec = max(tb - ta, 1e-6)
n = ub.completion_tokens or 200
print(json.dumps({"label": label, "prompt_tokens": ua.prompt_tokens,
                  "decode_s": round(dec,2), "decode_tok_s": round((n-1)/dec,1),
                  "prefill_tok_s": round(ua.prompt_tokens/max(ta,1e-6))}))
EOF
}

echo "=== waiting for idle GPU $(date -u) ===" | tee -a "$RUNLOG"
if idle_wait; then
  echo "idle confirmed" | tee -a "$RUNLOG"
  measure short 5
  idle_wait && measure long 2400
else
  echo "!! GPU never went idle in 10 min -- measurements would be contended" | tee -a "$RUNLOG"
fi
echo "=== done $(date -u) ===" | tee -a "$RUNLOG"
