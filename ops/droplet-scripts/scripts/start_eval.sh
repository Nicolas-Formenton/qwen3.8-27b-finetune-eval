#!/bin/bash
# Launcher: kill any previous chain (by pid, never by self-matching pattern), then start the n=200 chain.
PAT="eval_all"
for x in $(ps -eo pid=,args= | grep -v grep | awk -v p="$PAT" 'index($0,p){print $1}'); do
  if [ "$x" != "$$" ] && [ "$x" != "$PPID" ]; then kill "$x" 2>/dev/null; fi
done
sleep 2
pkill -f "scripts/xlam_heldout" 2>/dev/null
docker rm -f vllm-eval >/dev/null 2>&1
sleep 3
setsid bash /root/scripts/eval_all.sh > /root/eval_all.log 2>&1 &
echo "STARTED $(date -u)"
