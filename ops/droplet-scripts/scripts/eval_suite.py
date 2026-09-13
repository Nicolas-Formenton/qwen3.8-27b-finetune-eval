import json, re, time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="eval")

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "get_stock", "description": "Stock price for a ticker", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}]

def ask(messages, max_tokens=400):
    try:
        r = client.chat.completions.create(model="m", messages=messages, max_tokens=max_tokens, temperature=0.2)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"__ERROR__ {e}"

def jparse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

# (id, category, messages, check -> bool)
TASKS = [
    ("json1", "json", [{"role":"user","content":"Reply ONLY with valid JSON: {\"ok\": true, \"items\": [\"a\",\"b\",\"c\"]}"}],
     lambda t: (lambda j: j is not None and j.get("ok") is True and isinstance(j.get("items"), list) and len(j["items"])==3)(jparse(t))),
    ("json2", "json", [{"role":"user","content":"Envie APENAS JSON valido: {\"status\": \"ok\", \"total\": 42}"}],
     lambda t: (lambda j: j is not None and j.get("status")=="ok" and j.get("total")==42)(jparse(t))),
    ("bullets5", "format", [{"role":"user","content":"List exactly 5 reasons to fine-tune an LLM. One per line starting with '- '."}],
     lambda t: len([l for l in t.splitlines() if l.strip().startswith("-")])==5),
    ("bullets3pt", "format", [{"role":"user","content":"De exatamente 3 beneficios do ROCm, um por linha com '- '"}],
     lambda t: len([l for l in t.splitlines() if l.strip().startswith("-")])==3),
    ("codefence", "format", [{"role":"user","content":"Write a python function add(a,b) inside a ```python code fence. Nothing else."}],
     lambda t: "def add" in t and "```python" in t and t.count("```")>=2),
    ("single_sentence", "format", [{"role":"user","content":"Explain what a GPU is in ONE sentence."}],
     lambda t: t.count(".")<=2 and len(t.split())<=60),
    ("keyword_bandwidth", "content", [{"role":"user","content":"In one paragraph, explain why memory bandwidth matters for LLM inference. You MUST include the word 'bandwidth'."}],
     lambda t: "bandwidth" in t.lower()),
    ("keyword_hermes", "content", [{"role":"user","content":"One sentence mentioning the word 'agent' and the word 'tool'."}],
     lambda t: "agent" in t.lower() and "tool" in t.lower()),
    ("tool_weather", "tool", [{"role":"system","content":"You have tools: " + json.dumps(TOOLS) + ". When asked, reply with ONLY a JSON object {\"name\": <fn>, \"arguments\": {...}}."},
      {"role":"user","content":"What is the weather in Sao Paulo right now?"}],
     lambda t: (lambda j: j is not None and j.get("name")=="get_weather" and "city" in str(j.get("arguments")))(jparse(t))),
    ("tool_stock", "tool", [{"role":"system","content":"You have tools: " + json.dumps(TOOLS) + ". Reply ONLY JSON {\"name\": ..., \"arguments\": {...}}."},
      {"role":"user","content":"Check the current price of AAPL."}],
     lambda t: (lambda j: j is not None and j.get("name")=="get_stock" and "ticker" in str(j.get("arguments")))(jparse(t))),
    ("nopreamble", "format", [{"role":"user","content":"Answer directly with no preamble, no greeting: 2+2=?"}],
     lambda t: not re.match(r"(?i)^(sure|claro|of course|ok|sim|yes|here|aqui)", t) and t.strip()[:3]=="4"),
    ("tsv", "format", [{"role":"user","content":"Output a TSV with exactly 3 rows and 2 columns: city<TAB>population. No header, no code fence."}],
     lambda t: len([l for l in t.splitlines() if "\t" in l])==3),
    ("cap_words", "format", [{"role":"user","content":"Summarize LoRA in at most 20 words."}],
     lambda t: len(t.split())<=22),
    ("pt_short", "format", [{"role":"user","content":"Responda em portugues, no maximo 15 palavras: o que e VRAM?"}],
     lambda t: len(t.split())<=17 and any(c in t.lower() for c in "çãõáéíóú")),
    ("code_no_comment", "format", [{"role":"user","content":"Write python: x = [i*i for i in range(10)]. Output the code only."}],
     lambda t: "range(10)" in t and not t.startswith("```")),
    ("json_require_keys", "json", [{"role":"user","content":"JSON only: {\"model\": \"qwen\", \"vram_gb\": 192}"}],
     lambda t: (lambda j: j is not None and j.get("model")=="qwen" and j.get("vram_gb")==192)(jparse(t))),
    ("two_answers", "format", [{"role":"user","content":"Answer with exactly two short sentences."}],
     lambda t: 1 <= t.count(".") <= 2),
    ("explicit_tool_names", "tool", [{"role":"system","content":"Tools available: send_email, create_calendar_event."},{"role":"user","content":"Schedule a meeting tomorrow 10am. Reply only the tool name."}],
     lambda t: t.strip().lower() in ("create_calendar_event","create_calendar_event.")),
    ("no_markdown", "format", [{"role":"user","content":"Reply without markdown: the square root of 144."}],
     lambda t: "**" not in t and "`" not in t and "12" in t),
    ("bilingual_follow", "format", [{"role":"user","content":"Responda 'ok' se entendeu. So a palavra ok."}],
     lambda t: t.strip().lower()=="ok"),
]

def run_all(model_label):
    res = []
    for tid, cat, msgs, check in TASKS:
        t0 = time.time()
        ans = ask(msgs)
        dt = round(time.time() - t0, 1)
        passed = bool(check(ans))
        res.append({"id": tid, "cat": cat, "pass": passed, "time_s": dt, "answer_head": ans[:140]})
    ok = sum(r["pass"] for r in res)
    print(json.dumps({"model": model_label, "n": len(res), "passed": ok, "pass_rate": round(ok/len(res), 3),
                      "by_cat": {c: {"pass": sum(1 for r in res if r["cat"]==c and r["pass"]), "n": sum(1 for r in res if r["cat"]==c)} for c in set(r["cat"] for r in res)}}, indent=1))
    with open(f"/root/eval_{model_label}.json", "w") as f:
        json.dump({"model": model_label, "results": res}, f, indent=1)
    return res

if __name__ == "__main__":
    import sys
    run_all(sys.argv[1] if len(sys.argv) > 1 else "model")
