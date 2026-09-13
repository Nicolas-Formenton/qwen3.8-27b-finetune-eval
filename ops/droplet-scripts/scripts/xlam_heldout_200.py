import json, os, re, sys, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset

# n=200 held-out tool-call eval
#  - rows 6000+ = never seen in training (train used the first 6000 streamed rows)
#  - uniform reservoir sample with fixed seed (avoid a possibly atypical contiguous block)
#  - temperature 0 (deterministic), thinking disabled via chat_template_kwargs
#  - 12-way concurrency: vLLM batches, so wall-clock is far below sequential
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = 20260910
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 12
LABEL = sys.argv[1] if len(sys.argv) > 1 else "model"

os.environ["HF_TOKEN"] = open("/root/.hf_token").read().strip()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="eval", timeout=600)

# ---- reservoir sample of N unseen rows ----
rng = random.Random(SEED)
res = []
n_seen = 0
for i, r in enumerate(load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)):
    if i < 6000:
        continue
    idx = i - 6000
    n_seen += 1
    if idx < N:
        res.append(r)
    else:
        j = rng.randint(0, idx)
        if j < N:
            res[j] = r
print(f"sampled {len(res)} from {n_seen} unseen rows (seed={SEED})", flush=True)

def make_prompt(r):
    tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
    ans = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
    expected = ans[0]["name"] if isinstance(ans, list) else ans["name"]
    sys_tools = ("You are a helpful assistant with access to functions. When a tool is needed, reply with ONLY "
                 "a JSON list: [{\"name\": \"<fn>\", \"arguments\": {...}}].\nFunctions:\n" + json.dumps(tools))
    return sys_tools, r["query"], expected

def first_call_name(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    idx = text.rfind("[")
    tail = text[idx:] if idx != -1 else text
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', tail)
    return names[0] if names else None

def one(item):
    i, r = item
    sys_tools, query, expected = make_prompt(r)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": sys_tools + "\n\nQuery: " + query}],
            max_tokens=900, temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        out = f"__ERROR__ {e}"
    got = first_call_name(out)
    return {"i": i, "expected": expected, "got": got, "ok": got == expected,
            "time_s": round(time.time() - t0, 2), "query": query[:120], "head": out[:160]}

t0 = time.time()
rows = []
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = [ex.submit(one, (i, r)) for i, r in enumerate(res)]
    for done, f in enumerate(as_completed(futs), 1):
        rows.append(f.result())
        if done % 25 == 0:
            acc = sum(x["ok"] for x in rows) / len(rows)
            print(f"  {done}/{N} answered  acc={acc:.3f}  elapsed={round(time.time()-t0,1)}s", flush=True)
wall = round(time.time() - t0, 1)

ok = sum(r["ok"] for r in rows)
n = len(rows)
p = ok / n
z = 1.96
den = 1 + z * z / n
centre = (p + z * z / (2 * n)) / den
half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
multi = sum(1 for r in rows if r["got"] is None)

out = {
    "label": LABEL, "harness": "xlam_heldout_200 v1", "n": n, "correct": ok,
    "accuracy": round(p, 4),
    "wilson_ci95": [round(centre - half, 4), round(centre + half, 4)],
    "parse_failures": multi, "wall_s": wall, "concurrency": CONC,
    "seed": SEED, "temperature": 0, "thinking": False,
    "rows_per_s": round(n / wall, 2) if wall else None,
    "fails": [r for r in rows if not r["ok"]][:15],
    "all": rows,
}
path = f"/root/xlam200_{LABEL}.json"
json.dump(out, open(path, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("label", "n", "correct", "accuracy", "wilson_ci95", "parse_failures", "wall_s")}))
