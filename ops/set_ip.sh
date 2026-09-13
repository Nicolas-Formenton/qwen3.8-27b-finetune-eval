#!/bin/bash
# Point everything at a (new) droplet IP in one shot.
#   bash ops/set_ip.sh 129.212.188.58
#
# Updates: ~/.ssh/config alias `amd`, ops/droplet.env (used by ops/watch.sh), and the two
# Hermes providers + model.base_url in the qwen-dxlam profile. Run it after every droplet
# recreation, otherwise `ssh amd` and the profile keep pointing at a dead IP.
set -euo pipefail

IP="${1:-}"
if [ -z "$IP" ]; then
  echo "usage: bash ops/set_ip.sh <droplet-ip>" >&2
  exit 2
fi
case "$IP" in
  *[!0-9.]*) echo "!! '$IP' does not look like an IPv4 address" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
KEY_PATH="\$HOME/.ssh/amd_devcloud"

# 1) ssh alias (git-bash / Linux side)
SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
touch "$SSH_CONFIG"
if grep -q "^Host amd$" "$SSH_CONFIG" 2>/dev/null; then
  # rewrite the HostName line inside the `Host amd` block.
  # NOTE: this runs under git-bash, whose Python is a NATIVE Windows binary and does not
  # understand MSYS paths like /c/Users/... -> convert to C:/Users/... first (MSYS path
  # translation is disabled here, so `git -C /c/...`-style calls fail the same way).
  SSH_CONFIG_NATIVE="$SSH_CONFIG"
  case "$SSH_CONFIG_NATIVE" in
    /[A-Za-z]/*) SSH_CONFIG_NATIVE="$(printf '%s' "$SSH_CONFIG_NATIVE" | sed -E 's|^/([A-Za-z])/|\1:/|')" ;;
  esac
  python - "$SSH_CONFIG_NATIVE" "$IP" <<'PY'
import re, sys, pathlib
p, ip = pathlib.Path(sys.argv[1]), sys.argv[2]
text = p.read_text()
block = re.search(r"(?ms)^Host amd$.*?(?=^Host |\Z)", text)
if not block:
    sys.exit("no 'Host amd' block found")
new = re.sub(r"(?m)^(\s*HostName\s+).*$", lambda m: m.group(1) + ip, block.group(0))
p.write_text(text[:block.start()] + new + text[block.end():])
print("  ssh config: HostName ->", ip)
PY
else
  cat >> "$SSH_CONFIG" <<EOF

Host amd
  HostName $IP
  User root
  IdentityFile $KEY_PATH
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
EOF
  echo "  ssh config: bloco 'Host amd' criado ($IP)"
fi

# 2) ops/droplet.env (consumed by ops/watch.sh so it works from any shell, incl. WSL)
cat > "$HERE/droplet.env" <<EOF
# Droplet connection settings, sourced by ops/*.sh. Updated by ops/set_ip.sh.
AMD_HOST=$IP
AMD_USER=root
AMD_KEY_WIN="${AMD_KEY_WIN:-$(cygpath -m "$HOME/.ssh/amd_devcloud" 2>/dev/null || echo "$HOME/.ssh/amd_devcloud")}"
AMD_KEY_POSIX=\$HOME/.ssh/amd_devcloud
EOF
echo "  ops/droplet.env atualizado"

# 3) Hermes profile providers (both thinking-on and thinking-off entries + base_url)
PROFILE=qwen-dxlam
if command -v hermes >/dev/null 2>&1; then
  for prov in qwen-amd qwen-amd-fast; do
    hermes --profile "$PROFILE" config set "providers.$prov.api" "http://$IP:8000/v1" >/dev/null
  done
  hermes --profile "$PROFILE" config set model.base_url "http://$IP:8000/v1" >/dev/null
  echo "  perfil $PROFILE: providers e base_url -> $IP"
else
  echo "  !! 'hermes' nao esta no PATH: atualize o perfil manualmente"
fi

# 4) Sessions store their OWN copy of base_url (state.db -> sessions.model_config JSON and
#    billing_base_url). A stale copy makes the context-length probe fail silently, so the
#    client falls back to a hardcoded catalog window (observed: 131,072 instead of 262,144).
#    With the prompt then exceeding the BELIEVED limit the output allowance collapses and a
#    turn is cut mid-sentence. Recreating a droplet without this step reintroduces the bug.
STATE_DB="$HOME/AppData/Local/hermes/profiles/$PROFILE/state.db"
[ -f "$STATE_DB" ] || STATE_DB="$HOME/.hermes/profiles/$PROFILE/state.db"
if [ -f "$STATE_DB" ]; then
  # native Python on Windows cannot read /c/... paths (MSYS translation is off)
  STATE_DB_NATIVE="$STATE_DB"
  case "$STATE_DB_NATIVE" in
    /[A-Za-z]/*) STATE_DB_NATIVE="$(printf '%s' "$STATE_DB_NATIVE" | sed -E 's|^/([A-Za-z])/|\1:/|')" ;;
  esac
  python - "$STATE_DB_NATIVE" "http://$IP:8000/v1" <<'PY'
import json, re, sqlite3, sys
db, new_url = sys.argv[1], sys.argv[2]
pat = re.compile(r"http://\d{1,3}(?:\.\d{1,3}){3}:8000/v1")
con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id, model_config, billing_base_url FROM sessions")
rows = cur.fetchall()
changed = 0
for sid, mc, bbu in rows:
    new_mc, new_bbu = mc, bbu
    if isinstance(mc, str) and pat.search(mc):
        try:
            obj = json.loads(mc)
            if isinstance(obj, dict) and isinstance(obj.get("base_url"), str) and pat.search(obj["base_url"]):
                obj["base_url"] = new_url
                new_mc = json.dumps(obj)
        except Exception:
            new_mc = pat.sub(new_url, mc)
    if isinstance(bbu, str) and pat.search(bbu):
        new_bbu = pat.sub(new_url, bbu)
    if (new_mc, new_bbu) != (mc, bbu):
        cur.execute("UPDATE sessions SET model_config=?, billing_base_url=? WHERE id=?", (new_mc, new_bbu, sid))
        changed += 1
con.commit(); con.close()
print(f"  state.db: {changed} sessao(oes) reapontada(s) para {new_url.split('/v1')[0]}")
PY
else
  echo "  !! state.db nao encontrado -- sessoes seguiriam apontando para o IP antigo"
fi

echo
echo "verificando conexao..."
if ssh -o ConnectTimeout=20 -o BatchMode=yes amd 'echo CONEXAO_OK' 2>/dev/null; then
  echo "tudo apontando para $IP"
else
  echo "!! ssh falhou -- o droplet pode estar iniciando (aguarde ~1 min) ou a chave nao confere"
  exit 1
fi
