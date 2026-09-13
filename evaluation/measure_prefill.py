"""Perfil de prefill: TTFT e tok/s de prefill single-stream em funcao do contexto.

Por que este teste existe
-------------------------
Um demo publico do Qwen3.8-Flash-Next (MoE, quantizado, mlx-serve, M5 Max 128GB) reportou
"2.000 tok/s de prefill" a 16k de contexto. A pergunta natural e se o nosso Qwen3.8-27B
(dense, BF16, vLLM ROCm numa MI300X) chega la.

A comparacao direta NAO e valida, por tres eixos, e o script reporta os tres junto do numero:
  1. modelo  - o deles e MoE (ativa uma fracao dos pesos por token); o nosso e dense 27,8B
               (ativa os 27,8B inteiros). MoE tem prefill intrinsecamente mais barato.
  2. contexto- o deles mediu a 16k; o nosso modelo vai a 256k.
  3. stack   - mlx-serve (Apple Silicon) vs vLLM ROCm (MI300X).

O que o numero deles NAO pode ser comparado: prefill AGREGADO com muitas requisicoes
concorrentes, que e o que os logs do engine mostram (Avg prompt throughput). Este script
mede SINGLE-STREAM, que e a unica forma comparavel.

Metodo
------
Para cada nivel de contexto:
  - COLD: prompt unico (prefixo com salt aleatorio) -> o prefix cache nao pode acertar, entao
    o prefill e pago inteiro. E este o numero honesto.
  - WARM: o mesmo prompt enviado de novo -> mede o ganho do prefix cache (vLLM roda com
    enable_prefix_caching=True).
  - max_tokens=1 e stream=True: TTFT isola o prefill; nenhum decode contamina a medida.
  - REPEATS vezes, com salt diferente a cada repeticao; reporta mediana e min/max.

O tamanho REAL do prompt vem de usage.prompt_tokens na resposta, nao do que pedimos - o
tokenizer decide. A coluna `alvo` e so o alvo; `tokens` e o medido.

Uso (no droplet):
    export PREFILL_BASE_URL=http://localhost:8000/v1
    export PREFILL_API_KEY=$(cat /root/vllm_key)
    /root/bfcl-env/bin/python measure_prefill.py            # niveis padrao
    /root/bfcl-env/bin/python measure_prefill.py 1000 16000 # niveis escolhidos
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

BASE = os.environ.get("PREFILL_BASE_URL", "http://localhost:8000/v1")
KEY = os.environ.get("PREFILL_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
MODEL = os.environ.get("PREFILL_MODEL", "qwen38-base")
REPEATS = int(os.environ.get("PREFILL_REPEATS", "3"))
OUT = os.environ.get("PREFILL_OUT", "prefill_profile.json")

# Niveis padrao, em tokens-alvo. 256k e o teto do modelo; se o servidor recusar, o script
# registra o erro do nivel e segue.
LEVELS = [1000, 4000, 16000, 64000, 128000, 256000]

# ~1.3 tokens por palavra em ingles. O tamanho exato e corrigido depois por prompt_tokens.
FILLER_WORD = "context "


def build_prompt(target_tokens: int, salt: str) -> str:
    """Prompt com aproximadamente target_tokens. O salt vem PRIMEIRO para invalidar o prefix
    cache: sem isso o segundo nivel reusaria o prefixo do primeiro e mediria cache, nao prefill."""
    body = FILLER_WORD * int(target_tokens / 1.3)
    return (
        f"[{salt}] Ignore the filler below and reply with the single word OK.\n"
        f"{body}\n"
        f"Now reply with the single word OK."
    )


def timed_stream(prompt: str, max_tokens: int = 1) -> dict:
    """Um request com stream=True. Retorna ttft, tempo total e o usage reportado."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )

    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if ttft is None and obj.get("choices"):
                # primeiro chunk com conteudo OU com role ja significa que o prefill acabou
                ttft = time.time() - t0
    total = time.time() - t0

    return {
        "ttft_s": ttft if ttft is not None else total,
        "total_s": total,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def measure(target: int, salt: str) -> dict:
    """Um request para um prompt com este salt.

    O salt PRECISA ser unico por (nivel, repeticao). Duas armadilhas que ja produziram
    numeros fisicamente impossiveis (169k tok/s, 7x acima do pico da MI300X):
      1. salt fixo entre as N repeticoes -> da 2a em diante o prefix cache acerta.
      2. niveis construidos por repeticao de um mesmo token -> o prompt de 4k CONTEM o de
         1k como prefixo, entao o nivel maior reaproveita o trabalho do menor.
    Com salt unico no inicio, o hash do primeiro bloco muda e o vLLM invalida todos os
    blocos seguintes -> miss total, que e o que "cold" tem que significar.
    """
    prompt = build_prompt(target, salt)
    r = timed_stream(prompt)
    tok = r["prompt_tokens"]
    if tok and r["ttft_s"] > 0:
        r["prefill_tok_s"] = round(tok / r["ttft_s"], 1)
    else:
        r["prefill_tok_s"] = None
    return r


def main() -> None:
    levels = [int(a) for a in sys.argv[1:]] or LEVELS
    print(f"model={MODEL}  base={BASE}  repeats={REPEATS}")
    print(f"{'alvo':>8} {'tokens':>8} {'cold ttft':>10} {'cold tok/s':>11} {'warm ttft':>10} {'warm tok/s':>11}")
    print("-" * 66)

    report: dict = {"model": MODEL, "base_url": BASE, "repeats": REPEATS, "levels": []}

    for target in levels:
        entry: dict = {"target": target, "cold": [], "warm": [], "error": None}
        try:
            for i in range(REPEATS):
                # salt unico por nivel+repeticao: o 1o envio e cache miss (COLD de verdade),
                # o 2o usa o MESMO prompt, entao mede o ganho do prefix cache (WARM).
                salt = f"L{target}R{i}"
                entry["cold"].append(measure(target, salt))
                entry["warm"].append(measure(target, salt))
        except Exception as exc:  # noqa: BLE001 - reportar o erro do nivel e seguir
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["levels"].append(entry)
            print(f"{target:>8}  ERRO: {entry['error'][:60]}")
            continue

        ct = [c["ttft_s"] for c in entry["cold"]]
        cps = [c["prefill_tok_s"] for c in entry["cold"] if c["prefill_tok_s"]]
        wt = [w["ttft_s"] for w in entry["warm"]]
        wps = [w["prefill_tok_s"] for w in entry["warm"] if w["prefill_tok_s"]]
        tok = entry["cold"][0]["prompt_tokens"]

        entry["summary"] = {
            "prompt_tokens": tok,
            "cold_ttft_median_s": round(statistics.median(ct), 3),
            "cold_prefill_median_tok_s": round(statistics.median(cps), 1) if cps else None,
            "cold_prefill_min_tok_s": round(min(cps), 1) if cps else None,
            "cold_prefill_max_tok_s": round(max(cps), 1) if cps else None,
            "warm_ttft_median_s": round(statistics.median(wt), 3),
            "warm_prefill_median_tok_s": round(statistics.median(wps), 1) if wps else None,
        }
        report["levels"].append(entry)
        s = entry["summary"]
        print(f"{target:>8} {tok if tok else '?':>8} {s['cold_ttft_median_s']:>9.3f}s "
              f"{s['cold_prefill_median_tok_s']:>11} {s['warm_ttft_median_s']:>9.3f}s "
              f"{s['warm_prefill_median_tok_s']:>11}")

    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsalvo em {OUT}")


if __name__ == "__main__":
    main()
