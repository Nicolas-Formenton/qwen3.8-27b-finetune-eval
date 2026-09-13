#!/bin/bash
# Live status of the agent-native BFCL grid (runs ON the droplet).
#   ssh amd 'bash /root/status.sh'
#
# The run total is DERIVED from the grid script (models x categories x 2 modes)
# instead of hard-coded — a hard-coded 24 read "26/24" once the prompt-mode pass
# started, because 6 models x 4 cats x 2 modes = 48.

LOGD=/root/bfcl-project/grid.log
GRID=/root/scripts2/run_bfcl_grid.sh

now=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
done_runs=$(grep -ac '^    score:' "$LOGD" 2>/dev/null | head -1); done_runs=${done_runs:-0}
errs=$(grep -ac '!!' "$LOGD" 2>/dev/null | head -1); errs=${errs:-0}
api=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models 2>/dev/null)
cur=$(grep -a '^--- \[' "$LOGD" 2>/dev/null | tail -1)

# ---- expected total, derived from the running script ----
MODELS_DEF=$(grep -oE '^MODELS=\$\{MODELS:-"[^"]*"' "$GRID" 2>/dev/null | sed 's/.*:-"//; s/"$//')
CATS_DEF=$(grep -oE '^CATS=\$\{CATS:-"[^"]*"' "$GRID" 2>/dev/null | sed 's/.*:-"//; s/"$//')
n_models=$(echo "$MODELS_DEF" | wc -w | tr -d ' ')
n_cats=$(echo "$CATS_DEF" | wc -w | tr -d ' ')
TOTAL=$(( n_models * n_cats * 2 ))
[ "$TOTAL" -gt 0 ] 2>/dev/null || TOTAL=48

# ---- elapsed + rate + ETA from the log's own timestamps ----
first=$(grep -am1 'BFCL v4 grid' "$LOGD" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8}')
eta="--"
if [ -n "$first" ] && [ "$done_runs" -gt 0 ]; then
  t0=$(date -u -d "$first" +%s 2>/dev/null)
  t1=$(date -u +%s)
  if [ -n "$t0" ]; then
    el=$((t1 - t0))
    per=$((el / done_runs))
    rem=$(( (TOTAL - done_runs) * per ))
    eta="~( $((rem/60))min restantes ) fim previsto $(date -u -d "@$((t1+rem))" '+%H:%M UTC' 2>/dev/null)"
    elmin=$((el/60))
  fi
fi

echo "=========================================================="
echo " agent-native | BFCL v4 grid"
echo " $now"
echo "=========================================================="
echo " progresso : $done_runs / $TOTAL runs  (${n_models} modelos x ${n_cats} categorias x 2 modos)"
[ -n "${elmin:-}" ] && echo " decorrido : ${elmin} min   |   ritmo: ~$((per/60))min $((per%60))s por run"
[ "$eta" != "--" ] && echo " ETA       : $eta"
echo " atual     : ${cur:-aguardando}"
echo " endpoint  : http://localhost:8000/v1/models -> $api"
if [ "$api" = "200" ]; then echo " servidor  : ok"; else echo " servidor  : !! API NAO RESPONDE"; fi
[ "$errs" -gt 0 ] && echo " ERROS     : $errs marcados com '!!' no grid.log"
echo
echo "--- placar (FC = dialeto nativo | prompt = dialeto do BFCL) ---"
grep -a '^    score:' "$LOGD" 2>/dev/null | awk '{printf "  %6.3f  %s\n", $2, $3}' | sed 's|(agent-native-qwen38-|(|'
echo
echo "--- run em andamento ---"
last=$(ls -t /root/logs/bfcl_*.log 2>/dev/null | head -1)
if [ -n "$last" ]; then
  echo "  $(basename "$last")"
  tail -c 400 "$last" | tr '\r' '\n' | grep -aE 'Generating results' | tail -1 | cut -c1-110
fi
echo
echo "--- maquina ---"
rocm-smi --showuse --showmemuse 2>/dev/null | grep -E 'GPU use|VRAM' | head -2 | sed 's/^/  /'
echo "  disco livre: $(df -h / | awk 'NR==2{print $4}')"
echo "  vllm: $(docker ps --filter name=vllm-lora --format '{{.Status}}' 2>/dev/null)"
echo "=========================================================="