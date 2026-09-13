"""Decode rate vs context length — measured INSIDE one streaming request.

Method notes (each one was a bug fixed):
  * Two separate calls (max_tokens=1 vs 200) and subtracting is INVALID: prefix caching
    makes the 2nd call's prefill cheaper, so the subtraction goes negative. Time the first
    token and the token arrivals inside ONE streaming call instead.
  * The engine's own "Avg generation throughput" divides by the logging window, so it
    under-reports badly -> never use it as the headline number.
  * Measurements are worthless if another request shares the GPU: wait for real idle first.
  * ignore_eos is REQUIRED or the model stops after a couple of tokens and decode timing
    becomes pure overhead noise.

Run:  python measure_stream.py [model] [ctx1,ctx2,...]
"""

import json, re, subprocess, sys, time
from openai import OpenAI

KEY = open("/root/vllm_key").read().strip()
c = OpenAI(base_url="http://localhost:8000/v1", api_key=KEY)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen38-dxlam"
CTXS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "90000"])]
REPS = 2

FILLER = ("the quick brown fox jumps over the lazy dog and then continues walking "
          "through the forest and over the hills ")
WORDS_PER_COPY = len(FILLER.split())


def build_prompt(ctx_tokens: int) -> str:
    if ctx_tokens <= 1000:
        return "Count from 1 to 1000, one per line."
    # Scale by the filler's REAL size (~1.3 tokens per word). Getting this wrong blows past
    # max-model-len and every long-context row errors out.
    n = max(1, int(ctx_tokens / (WORDS_PER_COPY * 1.3)))
    return FILLER * n + "\n\nCount from 1 to 1000, one per line."


def engine_state():
    out = subprocess.run(["docker", "logs", "vllm-lora"], capture_output=True, text=True)
    blob = (out.stdout or "") + (out.stderr or "")
    lines = [l for l in blob.split("\n") if "Engine 000: Avg" in l]
    running = None
    if lines:
        m = re.search(r"Running: (\d+) reqs", lines[-1])
        running = int(m.group(1)) if m else None
    posts = blob.count("POST /v1/chat/completions")
    return running, posts


def wait_idle(timeout=40):
    """Wait until the engine has no request in flight.

    Two traps learned the hard way:
      * a FRESH container has no "Engine 000: Avg" lines yet, so `running` parses as None.
        Requiring `running == 0` then never succeeds and burns the whole timeout before
        EVERY measurement (8 measurements x 150s = 20 min of pure waiting, zero requests
        issued). Treat "never seen a request" as idle.
      * keep the timeout short: this is a nicety, the actual measurement happens inside a
        single streaming call, so mild contention only skews the number slightly.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        r1, p1 = engine_state()
        time.sleep(10)
        r2, p2 = engine_state()
        if (r2 == 0 or r2 is None) and p2 == p1:
            return True
    return False


def measure(label, ctx_tokens, thinking, max_tokens=400):
    prompt = build_prompt(ctx_tokens)
    extra = {"ignore_eos": True}
    if not thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    first = None
    n = 0
    try:
        stream = c.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0, stream=True, extra_body=extra,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                n += 1
                if first is None:
                    first = time.time()
        end = time.time()
        ttft = (first - t0) if first else 0
        dec_n, dec_t = max(n - 1, 0), (end - first) if first else 0
        print(json.dumps({
            "label": label, "ctx_target": ctx_tokens, "thinking": thinking,
            "ttft_s": round(ttft, 2), "out_tokens": n,
            "decode_tok_s": round(dec_n / dec_t, 1) if dec_t > 0 else None,
        }), flush=True)
    except Exception as e:
        print(json.dumps({"label": label, "ctx_target": ctx_tokens,
                          "ERROR": str(e)[:200]}), flush=True)


print(json.dumps({"model": MODEL, "ctxs": CTXS, "reps": REPS}), flush=True)
for ctx in CTXS:
    for thinking in (False, True):
        for rep in range(REPS):
            if not wait_idle():
                print(json.dumps({"warn": "GPU not idle -- measurement may be contaminated",
                                  "ctx_target": ctx}), flush=True)
            measure(f"ctx{ctx}_think{'ON' if thinking else 'OFF'}_r{rep+1}", ctx, thinking)
