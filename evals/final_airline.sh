#!/bin/bash
# Rodada final: base vs xLAM no airline, 4 trials (80 amostras cada), config corrigida.
#
# Objetivo: cruzar o 95% de confianca. Com os dados truncados de hoje (n=51-54) o teste deu
# z=1,81 / p=0,071 — 93%, nao 95%. Mantendo as taxas observadas (base 85,2%, xLAM 70,6%),
# n=80 por modelo da z=2,27 / p=0,023.
#
# TRES correcoes em relacao a rodada anterior, todas por defeitos ja identificados:
#   1. PREFIXO DO PROVEDOR. O juiz ia como `qwen38-doh`, e o litellm exige `openai/qwen38-doh`.
#      Sem o prefixo: `BadRequestError: LLM Provider NOT provided` -> 67 retries na madrugada.
#      Verificado: com prefixo responde, sem prefixo falha.
#   2. SEM TIMEOUT CURTO. Os `timeout 7200` truncaram TODOS os blocos (rc=124) e por isso o
#      dxlam ficou com 16 de 60. Aqui o teto e 4h, so como rede de seguranca.
#   3. JUIZ INDEPENDENTE mantido: qwen38-doh esta fora do conjunto {base, xlam}.

set -u
export PATH=$HOME/.local/bin:$PATH
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8000/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-$(cat /root/vllm_key)}
export OPENAI_BASE_URL=$OPENAI_API_BASE

# PREFIXO openai/ e obrigatorio — sem ele o litellm nao resolve o provedor.
export TAU3_JUDGE_LLM=openai/qwen38-doh
export TAU3_INTERFACE_LLM=openai/qwen38-doh
export TAU3_USER_LLM=openai/qwen38-doh
export TAU3_EVAL_SIM_LLM=openai/qwen38-doh

TAU_DIR=${TAU_DIR:-/root/tools-tau3-bench}
PY=${PY:-/root/tau-env/bin/python}
SIM=${SIM:-openai/qwen38-base}
TRIALS=${TRIALS:-4}
NUM_TASKS=${NUM_TASKS:-20}
MAX_STEPS=${MAX_STEPS:-60}
SEED=${SEED:-300}
LOG=$TAU_DIR/final_airline.log
: > "$LOG"

echo "=== FINAL airline base vs xlam, ${TRIALS} trials ===" | tee -a "$LOG"
echo "juiz=$TAU3_JUDGE_LLM  simulador=$SIM  $(date -u)" | tee -a "$LOG"

# preflight: endpoint E juiz tem que responder, senao repetimos o bug
code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_API_BASE/models")
[ "$code" = "200" ] || { echo "!! preflight=$code" | tee -a "$LOG"; exit 1; }

cd "$TAU_DIR" || exit 1
"$PY" -c "
from tau2.utils.llm_utils import generate
from tau2.data_model.message import SystemMessage, UserMessage
r = generate(model='openai/qwen38-doh', messages=[SystemMessage(role='system',content='Reply OK.'),UserMessage(role='user',content='go')], temperature=0.0)
print('juiz responde:', repr(str(r.content)[:30]))
" 2>&1 | tail -2 | tee -a "$LOG"

run_one() {
  model=$1; adapter=$2
  if [ "$adapter" != "-" ]; then
    echo "--- load $model $(date -u)" | tee -a "$LOG"
    curl -s -m 120 -X POST "$OPENAI_API_BASE/load_lora_adapter" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$model\",\"lora_path\":\"$adapter\"}" | tee -a "$LOG"
    echo "" >> "$LOG"; sleep 15
  fi
  echo "--- $model / airline / $TRIALS trials  $(date -u)" | tee -a "$LOG"
  timeout 14400 "$PY" -m tau2.cli run \
    --domain airline \
    --agent-llm "openai/$model" \
    --user-llm "$SIM" \
    --num-trials "$TRIALS" \
    --num-tasks "$NUM_TASKS" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --save-to "FINAL-airline-${model}-t${TRIALS}" >> "$LOG" 2>&1
  echo "    rc=$?  $(date -u)" | tee -a "$LOG"
  nr=$(grep -ac "LLM Provider NOT provided" "$LOG" 2>/dev/null || echo 0)
  echo "    BadRequestError de provedor (deve ser 0): $nr" | tee -a "$LOG"
  if [ "$adapter" != "-" ]; then
    curl -s -m 120 -X POST "$OPENAI_API_BASE/unload_lora_adapter" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$model\"}" | tee -a "$LOG"; echo "" >> "$LOG"
  fi
}

run_one qwen38-base -
run_one qwen38-xlam /root/adapters/xlam-adapter

echo "=== FINAL_DONE $(date -u) ===" | tee -a "$LOG"
