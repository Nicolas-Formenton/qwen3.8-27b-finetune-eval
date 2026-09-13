#!/bin/bash
# Watcher with early failure detection (not just end-of-run notification).
#
# Exits (and therefore notifies) as soon as ANY of these is true:
#   1. the grid log gains a "!!" line (our own error marker: generate rc != 0)
#   2. the vLLM endpoint stops answering 200
#   3. the score count has not advanced for STALL_POLLS consecutive polls
#   4. the grid finished (BFCL_GRID_DONE)
#
# Exit codes: 0 = done, 1 = endpoint/error, 2 = stalled, 3 = outer timeout

set -u
POLL=90          # seconds between checks
STALL_POLLS=14   # ~21 min without a new score => stalled
MAX_POLLS=260    # ~6.5h outer bound

prev_scores=-1
prev_errs=-1
stall=0

for i in $(seq 1 $MAX_POLLS); do
  SNAP=$(ssh -o ConnectTimeout=20 amd '
    scores=$(grep -ac "^    score:" /root/bfcl-project/grid.log 2>/dev/null | head -1); scores=${scores:-0}
    errs=$(grep -ac "!!" /root/bfcl-project/grid.log 2>/dev/null | head -1); errs=${errs:-0}
    api=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null)
    done=$(grep -aq BFCL_GRID_DONE /root/bfcl-project/grid.log 2>/dev/null && echo 1 || echo 0)
    cur=$(grep -a "^--- \[" /root/bfcl-project/grid.log 2>/dev/null | tail -1)
    echo "$scores|$errs|$api|$done|$cur"
  ' 2>/dev/null)

  if [ -z "$SNAP" ]; then
    echo "[$(date -u '+%H:%M:%S')] ssh falhou; retentando"
    stall=$((stall+1))
  else
    scores=$(echo "$SNAP" | cut -d'|' -f1)
    errs=$(echo "$SNAP" | cut -d'|' -f2)
    api=$(echo "$SNAP" | cut -d'|' -f3)
    done=$(echo "$SNAP" | cut -d'|' -f4)
    cur=$(echo "$SNAP" | cut -d'|' -f5)

    # failure 1: a new "!!" error marker
    if [ "$errs" -gt "${prev_errs:-0}" ] 2>/dev/null && [ "${prev_errs:-0}" -ge 0 ] 2>/dev/null; then
      echo "ERRO NO GRID (!! marcado). Ultimo run: $cur"
      ssh -o ConnectTimeout=20 amd 'grep -a "!!" /root/bfcl-project/grid.log | tail -5'
      exit 1
    fi

    # failure 2: endpoint down
    if [ "$api" != "200" ]; then
      echo "ENDPOINT PARADO (http=$api). Ultimo run: $cur"
      exit 1
    fi

    # completion
    if [ "$done" = "1" ]; then
      echo "GRID COMPLETO ($scores runs)"
      ssh -o ConnectTimeout=20 amd 'grep -a "^    score:" /root/bfcl-project/grid.log'
      exit 0
    fi

    # failure 3: stall detection
    if [ "$scores" = "$prev_scores" ]; then
      stall=$((stall+1))
    else
      stall=0
      echo "[$(date -u '+%H:%M:%S')] progresso: $scores/24 | $cur"
    fi
    if [ "$stall" -ge "$STALL_POLLS" ]; then
      echo "TRAVADO: sem novo score em ~$((STALL_POLLS*POLL/60)) min. Ultimo run: $cur"
      ssh -o ConnectTimeout=20 amd 'tail -c 400 $(ls -t /root/logs/bfcl_*.log | head -1) | tr "\r" "\n" | tail -3'
      exit 2
    fi

    prev_scores=$scores
    prev_errs=$errs
  fi
  sleep $POLL
done

echo "WATCHER STALE: sem conclusao apos $MAX_POLLS polls"
exit 3