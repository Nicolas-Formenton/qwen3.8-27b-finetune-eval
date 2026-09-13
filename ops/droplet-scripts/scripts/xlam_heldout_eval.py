import json, os, re, time
from openai import OpenAI
from datasets import load_dataset

os.environ["HF_TOKEN"] = open("/root/.hf_token").read().strip()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="eval")

def ask(sys_tools, query, max_tokens=900):
    try:
        r = client.chat.completions.create(model="qwen3.8-27b",
            messages=[{"role": "user", "content": sys_tools + "\n\nQuery: " + query}],
            max_tokens=max_tokens, temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"__ERROR__ {e}"

def first_call_name(text):
    # strip <think> block, then take JSON list starting at the LAST "[" (after reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    idx = text.rfind("[")
    tail = text[idx:] if idx != -1 else text
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', tail)
    return names[0] if names else None

# held-out rows: skip the 6000 used in training
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
rows = []
for i, r in enumerate(ds):
    if i < 6000:
        continue
    rows.append(r)
    if len(rows) >= 40:
        break

n_correct, n_total = 0, 0
fails = []
for r in rows:
    tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
    ans = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
    expected = ans[0]["name"] if isinstance(ans, list) else ans["name"]
    sys_tools = ("You are a helpful assistant with access to functions. When a tool is needed, reply with ONLY "
                 "a JSON list: [{\"name\": \"<fn>\", \"arguments\": {...}}].\nFunctions:\n" + json.dumps(tools))
    t0 = time.time()
    out = ask(sys_tools, r["query"])
    got = first_call_name(out)
    ok = got == expected
    n_total += 1
    n_correct += int(ok)
    if not ok:
        fails.append({"q": r["query"][:80], "expected": expected, "got": got, "head": out[:120]})
    if n_total % 15 == 0:
        print(f"  {n_total}/120 done, acc so far {round(n_correct/n_total,3)}", flush=True)

res = {"n": n_total, "correct": n_correct, "accuracy": round(n_correct/n_total, 4), "fails": fails[:10]}
print(json.dumps({"accuracy": res["accuracy"], "n": res["n"]}))
