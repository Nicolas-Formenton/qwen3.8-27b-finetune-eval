#!/bin/bash
# Talk to the droplet from ANY shell (git-bash, WSL, plain bash) without depending
# on an ~/.ssh/config alias — that alias only exists on the Windows side, so
# `bash ops/watch.sh` run from PowerShell (which resolves `bash` to WSL) used to fail
# with "Could not resolve hostname amd".
#
# Usage:
#   bash ops/watch.sh            # one snapshot
#   bash ops/watch.sh live       # refresh every 30s (Ctrl+C to exit)
#   bash ops/watch.sh raw CMD    # run an arbitrary command on the droplet

cd "$(dirname "$0")/.." || exit 1
ENV_FILE="ops/droplet.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE"; exit 1; }
# shellcheck disable=SC1090
. "$ENV_FILE"

# pick the key that actually exists in this environment (Windows path vs /mnt/c)
KEY=""
for cand in "$AMD_KEY_DEFAULT" "$AMD_KEY_WSL" "$AMD_KEY_WIN"; do
  if [ -f "$cand" ]; then KEY="$cand"; break; fi
done
if [ -z "$KEY" ]; then
  echo "!! private key not found. Tried:"
  printf '   %s\n' "$AMD_KEY_DEFAULT" "$AMD_KEY_WSL" "$AMD_KEY_WIN"
  echo "   Fix: set AMD_KEY_* in $ENV_FILE"
  exit 1
fi

SSH_OPTS=(-i "$KEY" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
TARGET="${AMD_USER}@${AMD_HOST}"

if [ "${1:-once}" = "raw" ]; then
  shift
  exec ssh "${SSH_OPTS[@]}" "$TARGET" "$@"
fi

if [ "${1:-once}" = "live" ]; then
  echo "Ao vivo (Ctrl+C para sair). Atualiza a cada 30s."
  while true; do
    clear 2>/dev/null || true
    ssh "${SSH_OPTS[@]}" "$TARGET" 'bash /root/status.sh'
    sleep 30
  done
fi

ssh "${SSH_OPTS[@]}" "$TARGET" 'bash /root/status.sh'