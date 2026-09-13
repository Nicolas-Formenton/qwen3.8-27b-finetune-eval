"""Aponta os LLMs internos do tau2/tau3 para o NOSSO endpoint.

Defeito (tau2-bench, `src/tau2/config.py`): quatro modelos internos estao HARDCODED em
modelos externos que nao existem do nosso lado:

    DEFAULT_LLM_USER              = "gpt-4.1-2025-04-14"   (simulador de usuario padrao)
    DEFAULT_LLM_NL_ASSERTIONS     = "gpt-4.1-2025-04-14"   (JUIZ das assertions em linguagem natural)
    DEFAULT_LLM_ENV_INTERFACE     = "gpt-4.1-2025-04-14"   (agente da interface de ambiente)
    DEFAULT_LLM_EVAL_USER_SIMULATOR = "claude-opus-4-5"

`--user-llm` sobrescreve o simulador (por isso o nosso run funcionou), mas NAO existe flag
para os outros. O juiz das NL assertions e o que importa: `evaluator_nl_assertions.py:122`
usa `DEFAULT_LLM_NL_ASSERTIONS` direto, e nao ha try/except em volta da chamada — entao a
falha do juiz derruba a tarefa inteira e o tau2 tenta 3x antes de desistir (o "Retry 3/3 for
task N: litellm.NotFoundError ... The model `gpt-4.1-2025-04-14` does not exist").

Dano medido por dominio, pelo `reward_basis` das tarefas:
  * airline: 50/50 tarefas usam `[COMMUNICATE, DB]` -> o juiz NUNCA e chamado -> resultados validos
  * retail:  112/114 tarefas usam `NL_ASSERTION`   -> o juiz e chamado sempre -> resultados invalidos

Portanto: os numeros de AIRLINE valem; os de RETAIL nao. Note que olhar so a presenca de
`nl_assertions` no arquivo da tarefa da um falso alarme (aparece em 100% do airline). O que
decide e o `reward_basis`.

O que este script faz: troca as constantes por leituras de ambiente, com juiz padrao
`qwen38-base`. Assim a escolha do juiz fica visivel e trocavel sem editar codigo.

REGRA DE INDEPENDENCIA: o juiz NAO pode ser o modelo sob teste. Se o agente avaliado for o
`qwen38-base`, julgar com o proprio base e auto-avaliacao. Para a matriz completa, use um
adapter FORA do conjunto testado (ex. `qwen38-doh`) via `TAU3_JUDGE_LLM`.

Uso:
    python fix_tau3_judge.py --show
    TAU3_JUDGE_LLM=qwen38-doh python fix_tau3_judge.py
"""

import argparse
import importlib.util
import pathlib
import sys

MARK = "# ==== agent-native: juizes apontados para o nosso endpoint (BEGIN) ===="

# (constante, variavel de ambiente, valor padrao)
TARGETS = [
    ("DEFAULT_LLM_NL_ASSERTIONS", "TAU3_JUDGE_LLM", "qwen38-base"),
    ("DEFAULT_LLM_ENV_INTERFACE", "TAU3_INTERFACE_LLM", "qwen38-base"),
    ("DEFAULT_LLM_USER", "TAU3_USER_LLM", "qwen38-base"),
    ("DEFAULT_LLM_EVAL_USER_SIMULATOR", "TAU3_EVAL_SIM_LLM", "qwen38-base"),
]


def find_config() -> pathlib.Path:
    spec = importlib.util.find_spec("tau2")
    if spec is None or not spec.origin:
        sys.exit("tau2 nao e importavel neste interpretador")
    path = pathlib.Path(spec.origin).parent / "config.py"
    if not path.exists():
        sys.exit(f"config.py nao encontrado em {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    path = find_config()
    text = path.read_text(encoding="utf-8")

    if args.show:
        for line in text.splitlines()[:40]:
            if "DEFAULT_LLM_" in line and "=" in line:
                print("  " + line.strip())
        return

    if "import os" not in text.split("\n\n")[0]:
        text = "import os\n" + text

    # remove um bloco anterior (idempotente)
    if MARK in text:
        head, _, _ = text.partition(MARK)
        text = head.rstrip() + "\n"

    lines = []
    for const, env, default in TARGETS:
        lines.append(f'{const} = os.environ.get("{env}", "{default}")')
    block = f"\n\n{MARK}\n" + "\n".join(lines) + "\n"

    path.write_text(text.rstrip() + block, encoding="utf-8")
    print(f"corrigido: {path}")

    # prova: recarrega e confere
    for mod in [m for m in sys.modules if m.startswith("tau2")]:
        del sys.modules[mod]
    from tau2 import config as cfg

    ok = 0
    for const, env, default in TARGETS:
        val = getattr(cfg, const, None)
        flag = "OK " if val == default else "!! "
        ok += val == default
        print(f"  {flag}{const} = {val}")
    print(f"constantes apontadas para o endpoint: {ok}/{len(TARGETS)}")


if __name__ == "__main__":
    main()
