#!/bin/bash
# τ³-bench (tau2-bench) against the vLLM endpoint started by serve_lora.sh.
#
#   MODELS="qwen38-base qwen38-xlam" DOMAINS="airline retail" bash run_tau3.sh
#
# Cost design: the user simulator is another LLM, and τ³ defaults to an OpenAI model.
# We instead point LiteLLM at OUR OWN vLLM and use the frozen base checkpoint as the
# simulator (--user-llm), so:
#   * external API spend is exactly zero
#   * the simulator is identical for every variant, which is required for a fair
#     comparison (a simulator that changes with the agent would confound the result)
#
# LiteLLM speaks the OpenAI protocol, so any model name our vLLM serves works as
# "openai/<served-name>" with OPENAI_API_BASE pointing at the endpoint.
#
# Setup on a fresh droplet (Python 3.12-3.13 required; the Studio venv is 3.13):
#   uv venv /root/tau-env --python 3.13
#   git clone --depth 1 https://github.com/sierra-research/tau2-bench /root/tools-tau3-bench
#   uv pip install --python /root/tau-env/bin/python -e /root/tools-tau3-bench
#   uv pip install --python /root/tau-env/bin/python websockets   # NOT optional:
#       `import tau2` fails without it (voice provider dep, undeclared in core)
#   cd /root/tools-tau3-bench && /root/tau-env/bin/python -m tau2.cli check-data

set -u

export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8000/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
# LiteLLM also reads OPENAI_BASE_URL in some versions; set both to be safe
export OPENAI_BASE_URL="$OPENAI_API_BASE"

TAU_DIR=${TAU_DIR:-/root/tools-tau3-bench}
PY=${PY:-/root/tau-env/bin/python}
SIM=${SIM:-qwen38-base}          # frozen user simulator
MODELS=${MODELS:-"qwen38-base qwen38-oh qwen38-xlam qwen38-dbase qwen38-doh qwen38-dxlam"}
DOMAINS=${DOMAINS:-"airline retail"}
NUM_TASKS=${NUM_TASKS:-20}
NUM_TRIALS=${NUM_TRIALS:-1}
MAX_STEPS=${MAX_STEPS:-60}
SEED=${SEED:-300}

LOG=${TAU_DIR}/run_tau3.log
: > "$LOG"
cd "$TAU_DIR" || { echo "!! tau dir missing: $TAU_DIR"; exit 1; }

echo "=== τ³ run $(date -u) ===" | tee -a "$LOG"
echo "endpoint=$OPENAI_API_BASE  simulator=$SIM (frozen)" | tee -a "$LOG"

# preflight: the endpoint must actually answer before we burn time
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${OPENAI_API_BASE}/models" 2>/dev/null)
if [ "$code" != "200" ]; then
  echo "!! endpoint returned $code -- aborting (start serve_lora.sh first)" | tee -a "$LOG"
  exit 1
fi

for model in $MODELS; do
  for domain in $DOMAINS; do
    save_to="${domain}-${model}"
    echo "--- $model / $domain  $(date -u)" | tee -a "$LOG"
    timeout 3600 "$PY" -m tau2.cli run \
      --domain "$domain" \
      --agent-llm "openai/${model}" \
      --user-llm "openai/${SIM}" \
      --num-trials "$NUM_TRIALS" \
      --num-tasks "$NUM_TASKS" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --save-to "$save_to" >> "$LOG" 2>&1
    rc=$?
    echo "    rc=$rc" | tee -a "$LOG"
    if [ "$rc" != "0" ]; then
      echo "    !! run failed for $model/$domain -- see log tail" | tee -a "$LOG"
      tail -5 "$LOG" | cut -c1-200 | tee -a "$LOG"
    fi
  done
done

echo "=== TAU3_DONE $(date -u) ===" | tee -a "$LOG"
echo "results under: $TAU_DIR/data/simulations/" | tee -a "$LOG"
ls -1 "$TAU_DIR/data/simulations/" 2>/dev/null | tee -a "$LOG"
