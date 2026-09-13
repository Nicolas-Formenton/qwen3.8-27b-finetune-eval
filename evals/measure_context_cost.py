"""Measure decode rate vs CONTEXT LENGTH, robustly.

Method (avoids fragile streaming parsing):
  A) request with max_tokens=1      -> measures prefill (TTFT) for that context
  B) request with max_tokens=N      -> measures prefill+decode
  decode_time = t_B - t_A  ;  decode_rate = (N - 1) / decode_time

Both requests share the identical prefix, so the prompt is in prefix cache after A, meaning
B's prefill is cheap and t_B - t_A is dominated by decode. Run each context twice (warm) and
report the second pair.
"""
import json, time
from openai import OpenAI

KEY = open("/root/vllm_key").read().strip()
c = OpenAI(base_url="http://localhost:8000/v1", api_key=KEY)

# ~110 chars per sentence unit; repeat to reach the desired padding
UNIT = ("Distributed systems background notes: consistency, quorums, leases, and failure "
        "detectors are described here purely to pad the prompt to a target length. ")

def build(pad_units: int) -> str:
    return UNIT * pad_units + "\n\nCount from 1 upward, one number per line, no commentary."

def ask(prompt, max_tokens):
    t0 = time.time()
    r = c.chat.completions.create(
        model="qwen38-dxlam",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0,
        # ignore_eos is REQUIRED here: without it the model stops after a couple of tokens
        # ("1\n2") and the decode measurement is pure overhead noise.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True},
    )
    dt = time.time() - t0
    return dt, r.usage

def measure(label, pad_units, out=200):
    prompt = build(pad_units)
    ask(prompt, 1)                      # warm the prefix cache
    t_a, u_a = ask(prompt, 1)           # prefill only
    t_b, u_b = ask(prompt, out)         # prefill + decode
    dec = max(t_b - t_a, 1e-6)
    n = u_b.completion_tokens or out
    print(json.dumps({
        "label": label, "prompt_tokens": u_a.prompt_tokens, "out_tokens": n,
        "prefill_s": round(t_a, 2), "prefill_tok_s": round(u_a.prompt_tokens / max(t_a, 1e-6)),
        "decode_s": round(dec, 2), "decode_tok_s": round((n - 1) / dec, 1),
    }), flush=True)

if __name__ == "__main__":
    for label, units in (("small", 5), ("medium", 400), ("large", 1200), ("xlarge", 2400)):
        try:
            measure(label, units)
        except Exception as e:
            print(json.dumps({"label": label, "error": str(e)[:200]}), flush=True)
