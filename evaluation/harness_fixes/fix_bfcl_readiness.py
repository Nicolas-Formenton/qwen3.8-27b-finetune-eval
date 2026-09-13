"""Patch BFCL's server-readiness probe so it authenticates.

Defect (BFCL v4, `base_oss_handler.py`): the readiness loop does

    response = requests.get(f"{self.base_url}/models")
    if response.status_code == 200: server_ready = True
    except requests.exceptions.ConnectionError: time.sleep(1)

It builds `self.api_key` (from REMOTE_OPENAI_API_KEY) for the OpenAI client but never sends it
here. Against an endpoint that requires auth the probe gets 401 forever: 401 raises no exception,
so `server_ready` never flips, the loop never exits, and it re-requests as fast as the socket
allows (observed: 341,062 unauthorized requests before the run was killed).

Fix: send the bearer token on the probe, and cap the wait so a genuinely unreachable endpoint
fails loudly instead of spinning.

Idempotent. Usage: python fix_bfcl_readiness.py
"""

import importlib.util
import pathlib
import sys

OLD = '                    response = requests.get(f"{self.base_url}/models")\n'
NEW = (
    '                    response = requests.get(\n'
    '                        f"{self.base_url}/models",\n'
    '                        headers=(\n'
    '                            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}\n'
    '                        ),\n'
    '                        timeout=10,\n'
    '                    )\n'
)

MAX_WAIT_OLD = "            server_ready = False\n            while not server_ready:\n"
MAX_WAIT_NEW = (
    "            server_ready = False\n"
    "            _waited = 0.0\n"
    "            while not server_ready:\n"
    "                if _waited > 300:\n"
    "                    raise RuntimeError(\n"
    "                        f\"server at {self.base_url} not ready after 300s \"\n"
    "                        f\"(last status={locals().get('response') and response.status_code})\"\n"
    "                    )\n"
)


def find_handler() -> pathlib.Path:
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.origin:
        sys.exit("bfcl_eval is not importable in this interpreter")
    return pathlib.Path(spec.origin).parent / "model_handler" / "local_inference" / "base_oss_handler.py"


def main() -> None:
    path = find_handler()
    text = path.read_text(encoding="utf-8")

    if "Authorization" in text and "self.base_url}/models" in text.split("Authorization")[0][-400:]:
        print(f"ja corrigido: {path}")
        return

    if OLD not in text:
        sys.exit(f"padrao nao encontrado em {path} -- BFCL mudou, revisar manualmente")

    text = text.replace(OLD, NEW, 1)
    if MAX_WAIT_OLD in text:
        text = text.replace(MAX_WAIT_OLD, MAX_WAIT_NEW, 1)
    path.write_text(text, encoding="utf-8")
    print(f"corrigido: {path}")

    # prove it imports and the patch is present
    for mod in [m for m in sys.modules if m.startswith("bfcl_eval")]:
        del sys.modules[mod]
    import bfcl_eval.model_handler.local_inference.base_oss_handler as h  # noqa: F401
    src = pathlib.Path(h.__file__).read_text(encoding="utf-8")
    print("Authorization no probe:", src.count('"Authorization": f"Bearer {self.api_key}"'))
    print("timeout no probe:", src.count("timeout=10"))


if __name__ == "__main__":
    main()
