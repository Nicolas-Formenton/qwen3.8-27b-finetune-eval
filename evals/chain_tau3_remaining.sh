#!/bin/bash
# tau3 (tau2-bench) — SO o que ficou faltando em 2026-09-12:
#   * xlam  / retail      (o airline ja rodou: 0,79 com 19/20)
#   * dxlam / airline + retail  (adiado por decisao do dono)
#
# Um adapter residente por vez (load -> run -> unload): o engine roda com --max-loras 1, e
# manter mais de um carregado custou ~36% de throughput na medicao anterior.
#
# Duas correcoes em relacao ao run_tau3.sh original, ambas por bugs ja identificados:
#   1. o preflight do original faz curl em /v1/models SEM header de auth -> 401 -> aborta.
#      Aqui a chave vai no header.
#   2. o original deixa OPENAI_API_KEY="dummy"; o LiteLLM le essa variavel para falar com o
#      endpoint, entao ficava em 401. Aqui e a chave real do vLLM.
#
# Simulador de usuario = qwen38-base CONGELADO, igual em todos os variantes. Um simulador que
# muda junto com o agente confundiria o resultado.
#
# Uso:  nohup bash chain_tau3_remaining.sh > /dev/null 2>&1 &

set -u
export PATH=$HOME/.local/bin:$PATH
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8000/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-$(cat /root/vllm_key)}
export OPENAI_BASE_URL=$OPENAI_API_BASE   # LiteLLM le as duas formas

TAU_DIR=${TAU_DIR:-/root/tools-tau3-bench}
PY=${PY:-/root/tau-env/bin/python}
SIM=${SIM:-qwen38-base}
NUM_TASKS=${NUM_TASKS:-20}
MAX_STEPS=${MAX_STEPS:-60}
SEED=${SEED:-300}
LOG=$TAU_DIR/tau3_remaining.log
: > "$LOG"

echo "=== tau3 restantes $(date -u) ===" | tee -a "$LOG"
echo "endpoint=$OPENAI_API_BASE  simulador=$SIM (congelado)" | tee -a "$LOG"

code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_API_BASE/models")
if [ "$code" != "200" ]; then
  echo "!! preflight=$code -- abortando" | tee -a "$LOG"
  exit 1
fi
echo "preflight=200 ok" | tee -a "$LOG"

cd "$TAU_DIR" || exit 1

run_one() {
  model=$1; adapter=$2; domains=$3
  if [ "$adapter" != "-" ]; then
    echo "--- load $model" | tee -a "$LOG"
    curl -s -m 120 -X POST "$OPENAI_API_BASE/load_lora_adapter" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$model\",\"lora_path\":\"$adapter\"}" | tee -a "$LOG"
    echo "" >> "$LOG"
    sleep 15   # deixa o adapter aquecer antes do primeiro request
  fi

  for domain in $domains; do
    echo "--- $model / $domain  $(date -u)" | tee -a "$LOG"
    timeout 5400 "$PY" -m tau2.cli run \
      --domain "$domain" \
      --agent-llm "openai/$model" \
      --user-llm "openai/$SIM" \
      --num-trials 1 \
      --num-tasks "$NUM_TASKS" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --save-to "${domain}-${model}" >> "$LOG" 2>&1
    rc=$?
    echo "    rc=$rc  $(date -u)" | tee -a "$LOG"
    if [ "$rc" != "0" ]; then
      echo "    !! falhou $model/$domain -- ultimas linhas:" | tee -a "$LOG"
      tail -5 "$LOG" | cut -c1-200 | tee -a "$LOG"
    fi
  done

  if [ "$adapter" != "-" ]; then
    curl -s -m 120 -X POST "$OPENAI_API_BASE/unload_lora_adapter" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$model\"}" | tee -a "$LOG"
    echo "" >> "$LOG"
  fi
}

# Só o que falta. O airline do xlam já está medido (0,79, 19/20).
run_one qwen38-xlam  /root/adapters/xlam-adapter  retail
run_one qwen38-dxlam /root/adapters/dxlam-adapter "airline retail"

echo "=== TAU3_REMAINING_DONE $(date -u) ===" | tee -a "$LOG"
echo "resultados em $TAU_DIR/data/simulations/" | tee -a "$LOG"
ls -1 "$TAU_DIR/data/simulations/" 2>/dev/null | tee -a "$LOG"
