#!/bin/bash
# Matriz tau3 completa, com o juiz CORRIGIDO e um juiz independente.
#
# Diferencas em relacao ao chain anterior (todas por defeitos encontrados):
#   1. O juiz das NL assertions / env interface esta hardcoded num modelo externo em
#      tau2/config.py. `fix_tau3_judge.py` trocou as 4 constantes por leitura de ambiente;
#      aqui exportamos TAU3_JUDGE_LLM. Sem isso, tasks de airline E retail falham em retry
#      (litellm.NotFoundError gpt-4.1-2025-04-14) e o score vira lixo.
#   2. JUIZ INDEPENDENTE: o juiz NAO pode ser o modelo sob teste. Usamos qwen38-doh, que esta
#      fora do conjunto {base, xlam, dxlam}. O vLLM roda com --max-loras 2 para manter doh
#      (juiz, fixo) + o agente da vez.
#   3. 3 TRIALS em vez de 1: com 20 tarefas, 1 trial da ~+-20 pontos de IC e os modelos ficam
#      indistinguiveis. 3 trials = 60 amostras -> IC ~+-8 pontos, que ja separa 90% de 79%.
#   4. timeout por dominio 7200s (2h) em vez de 5400s: o run anterior era truncado no meio.
#
# ORDEM: airline primeiro. E o dominio com menos dependencia de juiz e ja mostrou ser mais
# rapido, entao garante resultado utilizavel mesmo se o credito acabar no meio.

set -u
export PATH=$HOME/.local/bin:$PATH
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:8000/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-$(cat /root/vllm_key)}
export OPENAI_BASE_URL=$OPENAI_API_BASE
# Nomes de env que o fix_tau3_judge.py le. Todos apontam para o MESMO juiz independente:
# deixar algum no default (qwen38-base) reintroduziria auto-avaliacao, porque o base esta
# no conjunto testado.
export TAU3_JUDGE_LLM=${TAU3_JUDGE_LLM:-qwen38-doh}          # DEFAULT_LLM_NL_ASSERTIONS
export TAU3_INTERFACE_LLM=${TAU3_INTERFACE_LLM:-qwen38-doh}  # DEFAULT_LLM_ENV_INTERFACE
export TAU3_USER_LLM=${TAU3_USER_LLM:-qwen38-doh}            # DEFAULT_LLM_USER
export TAU3_EVAL_SIM_LLM=${TAU3_EVAL_SIM_LLM:-qwen38-doh}    # DEFAULT_LLM_EVAL_USER_SIMULATOR

TAU_DIR=${TAU_DIR:-/root/tools-tau3-bench}
PY=${PY:-/root/tau-env/bin/python}
SIM=${SIM:-qwen38-base}
TRIALS=${TRIALS:-3}
NUM_TASKS=${NUM_TASKS:-20}
MAX_STEPS=${MAX_STEPS:-60}
SEED=${SEED:-300}
LOG=$TAU_DIR/matrix_tau3.log
: > "$LOG"

echo "=== MATRIZ tau3 $(date -u) ===" | tee -a "$LOG"
echo "juiz=$TAU3_JUDGE_LLM (independente do conjunto testado)" | tee -a "$LOG"
echo "simulador=$SIM  trials=$TRIALS  tarefas=$NUM_TASKS  seed=$SEED" | tee -a "$LOG"

code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_API_BASE/models")
if [ "$code" != "200" ]; then
  echo "!! preflight=$code -- abortando" | tee -a "$LOG"; exit 1
fi

# confirma que o juiz esta de fato servido
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_API_BASE/models" \
  | tr ',' '\n' | grep -o '"id":"[^"]*"' | sed 's/.*"id":"//;s/"//' | grep -v modelperm \
  | tee -a "$LOG"
if ! curl -s -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_API_BASE/models" | grep -q "$TAU3_JUDGE_LLM"; then
  echo "!! juiz $TAU3_JUDGE_LLM NAO esta servido -- abortando (seria o bug de novo)" | tee -a "$LOG"
  exit 1
fi

cd "$TAU_DIR" || exit 1

run_one() {
  model=$1; adapter=$2; domains=$3
  if [ "$adapter" != "-" ]; then
    echo "--- load $model  $(date -u)" | tee -a "$LOG"
    curl -s -m 120 -X POST "$OPENAI_API_BASE/load_lora_adapter" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$model\",\"lora_path\":\"$adapter\"}" | tee -a "$LOG"
    echo "" >> "$LOG"; sleep 15
  fi

  for domain in $domains; do
    echo "--- $model / $domain / ${TRIALS} trials  $(date -u)" | tee -a "$LOG"
    timeout 7200 "$PY" -m tau2.cli run \
      --domain "$domain" \
      --agent-llm "openai/$model" \
      --user-llm "openai/$SIM" \
      --num-trials "$TRIALS" \
      --num-tasks "$NUM_TASKS" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --save-to "${domain}-${model}-t${TRIALS}" >> "$LOG" 2>&1
    rc=$?
    echo "    rc=$rc  $(date -u)" | tee -a "$LOG"
    # conta os retries de juiz: se houver, o fix nao pegou
    nr=$(grep -ac "litellm.NotFoundError" "$LOG" 2>/dev/null || echo 0)
    echo "    NotFoundError acumulados no log: $nr" | tee -a "$LOG"
  done

  if [ "$adapter" != "-" ]; then
    curl -s -m 120 -X POST "$OPENAI_API_BASE/unload_lora_adapter" \
      -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
      -d "{\"lora_name\":\"$model\"}" | tee -a "$LOG"
    echo "" >> "$LOG"
  fi
}

# ORDEM: todos os AIRLINE primeiro, depois todos os RETAIL.
# Motivo: airline e mais rapido e menos dependente do juiz. Rodando os 3 modelos no airline
# antes de tocar em retail, a comparacao principal (base vs xLAM vs D-xLAM, 60 amostras cada)
# fecha em ~6h. Se o credito acabar no meio, o que sobra e a comparacao completa, nao um
# modelo pela metade.
run_one qwen38-base  -                              "airline"
run_one qwen38-xlam  /root/adapters/xlam-adapter    "airline"
run_one qwen38-dxlam /root/adapters/dxlam-adapter   "airline"
run_one qwen38-base  -                              "retail"
run_one qwen38-xlam  /root/adapters/xlam-adapter    "retail"
run_one qwen38-dxlam /root/adapters/dxlam-adapter   "retail"

echo "=== MATRIZ_TAU3_DONE $(date -u) ===" | tee -a "$LOG"
ls -1 "$TAU_DIR/data/simulations/" 2>/dev/null | tee -a "$LOG"
