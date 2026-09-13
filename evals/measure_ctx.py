"""Decode rate vs CONTEXT LENGTH, without shell argument limits.

Previous attempt passed a 90k-token prompt as a command-line argument -> "Argument list
too long". Everything here happens inside Python; nothing large touches argv.

Method (two calls per context, avoids fragile stream parsing):
  A) max_tokens=1    -> elapsed ~= prefill (TTFT)
  B) max_tokens=200  -> elapsed ~= prefill + decode
  decode_tok_s = 199 / (B - A)

Run: python measure_ctx.py <model> [ctx1,ctx2,...]
"""

import json, sys, time
from openai import OpenAI

KEY = open("/root/vllm_key").read().strip()
c = OpenAI(base_url="http://localhost:8000/v1", api_key=KEY)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen38-dxlam"
CTXS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "40000", "90000", "130000"])]

FILLER = ("the quick brown fox jumps over the lazy dog and then continues walking "
          "through the forest and over the hills ")


def build_prompt(ctx_tokens: int) -> str:
    if ctx_tokens <= 1000:
        return "Count from 1 to 1000, one per line."
    # Scale by the filler's REAL size: len(FILLER.split()) words per copy, ~1.3 tokens per
    # word. (Getting this wrong blows past max-model-len: "your prompt contains at least
    # 262144 input tokens" — that is what invalidated the first context attempt.)
    tokens_per_copy = len(FILLER.split()) * 1.3
    n = max(1, int(ctx_tokens / tokens_per_copy))
    return FILLER * n + "\n\nCount from 1 to 1000, one per line."


def call(prompt: str, max_tokens: int):
    t0 = time.time()
    r = c.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True},
    )
    return time.time() - t0, r.usage


for ctx in CTXS:
    prompt = build_prompt(ctx)
    try:
        t_a, u_a = call(prompt, 1)     # prefill only
        t_b, u_b = call(prompt, 200)   # prefill + decode
        pt = u_b.prompt_tokens
        dec = u_b.completion_tokens - 1
        dt = t_b - t_a
        rate = dec / dt if dt > 0 else 0.0
        print(json.dumps({
            "ctx_target": ctx, "prompt_tokens": pt,
            "prefill_s": round(t_a, 2), "prefill_tok_s": round(pt / t_a) if t_a > 0 else None,
            "decode_s": round(dt, 2), "decode_tok_s": round(rate, 1),
        }), flush=True)
    except Exception as e:
        print(json.dumps({"ctx_target": ctx, "ERROR": str(e)[:200]}), flush=True)
