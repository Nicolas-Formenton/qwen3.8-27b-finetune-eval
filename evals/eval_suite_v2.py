"""agent-native instruction/tool suite v2 — the fixed version of our own harness.

What changed vs v1 (see docs/FAILURES.md #6) and why it matters:

  v1                                  v2
  ----------------------------------  --------------------------------------------
  thinking left ON (responses began   thinking explicitly disabled; <think> blocks
  with reasoning -> JSON checkers     stripped defensively in case the engine
  failed for every model)             ignores the flag
  compared only the FIRST function    compares function name AND expected argument
  NAME                                values (the v1 metric saturated at ~83%)
  no parallel-call task               parallel task requires BOTH calls
  no abstention task                  abstention task fails if any tool is emitted
  single sample per task              N repetitions with a seed, reports a rate

Design rule: every checker is a pure function of the answer string, so the whole
suite can be self-tested offline (see selftest_suite_v2.py) before any GPU is spent.
Task prompts are Portuguese/English mixed on purpose: instruction-following across
languages is part of what we are measuring.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("EVAL_MODEL", "qwen38-base")
REPEATS = int(os.environ.get("EVAL_REPEATS", "1"))
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "600"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "8"))
OUT_PATH = os.environ.get("EVAL_OUT", f"/root/evalv2_{MODEL}.json")

TOOLS_WEATHER_STOCK = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_stock", "description": "Stock price for a ticker",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
]

TOOLS_CALENDAR = [
    {"type": "function", "function": {"name": "send_email", "description": "Send an email",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create a calendar event",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "time": {"type": "string"}}, "required": ["title", "time"]}}},
]


# --------------------------------------------------------------------------- parsing

def strip_thinking(text: str) -> str:
    """Remove reasoning blocks. Engines that ignore enable_thinking still return these."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*Thinking Process:.*?(?=\{|\n\n)", "", text, flags=re.S)
    return text.strip()


def extract_calls(text: str) -> list[dict]:
    """Pull tool calls out of an answer, tolerant of JSON, Python-call and fenced forms.

    Returns a list of {"name": str, "arguments": dict}. Never raises: unparseable
    answers yield [] and the caller scores them as failures.
    """
    clean = strip_thinking(text)
    calls: list[dict] = []

    # 1) JSON object(s): prefer the LAST bracketed structure (after any reasoning)
    for candidate in _json_candidates(clean):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        calls = _normalise(data)
        if calls:
            return calls

    # 2) Python call syntax: fn(arg=value, ...)
    for name, argstr in re.findall(r"([A-Za-z_][\w.]*)\s*\(([^()]*)\)", clean):
        args = {}
        for key, val in re.findall(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]+)", argstr):
            args[key] = val.strip().strip("\"'")
        calls.append({"name": name, "arguments": args})
    return calls


def _json_candidates(text: str) -> list[str]:
    out = []
    for match in re.finditer(r"\[.*\]|\{.*\}", text, flags=re.S):
        out.append(match.group(0))
    out.reverse()                      # last occurrence first
    return out


def _normalise(data) -> list[dict]:
    items = data if isinstance(data, list) else [data]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function") or item.get("tool")
        if isinstance(name, dict):
            name = name.get("name")
        if not isinstance(name, str):
            continue
        args = item.get("arguments", item.get("parameters", item.get("args", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return calls


def parse_json(text: str):
    clean = strip_thinking(text)
    try:
        return json.loads(clean)
    except Exception:
        pass
    for candidate in _json_candidates(clean):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def arg_contains(call: dict, key: str, needle: str) -> bool:
    """Case-insensitive containment so 'Sao Paulo' matches 'São Paulo' variants loosely."""
    value = str(call.get("arguments", {}).get(key, "")).lower()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return norm(needle) in norm(value)


# --------------------------------------------------------------------------- checkers

def c_json_basic(text: str) -> bool:
    obj = parse_json(text)
    return isinstance(obj, dict) and obj.get("ok") is True and isinstance(obj.get("items"), list) and len(obj["items"]) == 3


def c_json_typed(text: str) -> bool:
    obj = parse_json(text)
    return isinstance(obj, dict) and obj.get("status") == "ok" and obj.get("total") == 42


def c_bullets5(text: str) -> bool:
    return len([l for l in strip_thinking(text).splitlines() if l.strip().startswith("- ")]) == 5


def c_codefence(text: str) -> bool:
    t = strip_thinking(text)
    return "def add" in t and "```python" in t and t.count("```") >= 2


def c_max20(text: str) -> bool:
    return 0 < len(strip_thinking(text).split()) <= 20


def c_no_preamble(text: str) -> bool:
    t = strip_thinking(text).strip()
    return not re.match(r"(?i)^(sure|claro|of course|ok|yes|here|aqui|the answer)", t) and t[:1] == "4"


def c_tsv_3x2(text: str) -> bool:
    return len([l for l in strip_thinking(text).splitlines() if l.count("\t") >= 1]) == 3


def c_tool_weather_args(text: str) -> bool:
    calls = extract_calls(text)
    return len(calls) == 1 and calls[0]["name"] == "get_weather" and arg_contains(calls[0], "city", "sao paulo")


def c_tool_stock_args(text: str) -> bool:
    calls = extract_calls(text)
    return len(calls) == 1 and calls[0]["name"] == "get_stock" and arg_contains(calls[0], "ticker", "AAPL")


def c_parallel_both(text: str) -> bool:
    names = {c["name"] for c in extract_calls(text)}
    return {"get_weather", "get_stock"} <= names


def c_abstain(text: str) -> bool:
    """No tool may be called: the answer must be plain prose containing 4."""
    return not extract_calls(text) and "4" in strip_thinking(text)


def c_right_tool(text: str) -> bool:
    calls = extract_calls(text)
    return len(calls) == 1 and calls[0]["name"] == "create_calendar_event"


# --------------------------------------------------------------------------- tasks

TASKS = [
    ("json_basic", "json",
     [{"role": "user", "content": 'Reply ONLY with valid JSON: {"ok": true, "items": ["a","b","c"]}'}],
     c_json_basic),
    ("json_typed", "json",
     [{"role": "user", "content": 'Envie APENAS JSON valido: {"status": "ok", "total": 42}'}],
     c_json_typed),
    ("bullets5", "format",
     [{"role": "user", "content": "List exactly 5 reasons to fine-tune an LLM. One per line starting with '- '."}],
     c_bullets5),
    ("codefence", "format",
     [{"role": "user", "content": "Write a python function add(a,b) inside a ```python code fence. Nothing else."}],
     c_codefence),
    ("max20", "format",
     [{"role": "user", "content": "Summarize LoRA in at most 20 words."}],
     c_max20),
    ("no_preamble", "format",
     [{"role": "user", "content": "Answer directly with no preamble, no greeting: 2+2=?"}],
     c_no_preamble),
    ("tsv_3x2", "format",
     [{"role": "user", "content": "Output a TSV with exactly 3 rows and 2 columns: city<TAB>population. No header, no code fence."}],
     c_tsv_3x2),
    ("tool_weather_args", "tool",
     [{"role": "system", "content": "You have tools: " + json.dumps(TOOLS_WEATHER_STOCK) +
       '. Reply with ONLY {"name": <fn>, "arguments": {...}}.'},
      {"role": "user", "content": "What is the weather in Sao Paulo right now?"}],
     c_tool_weather_args),
    ("tool_stock_args", "tool",
     [{"role": "system", "content": "You have tools: " + json.dumps(TOOLS_WEATHER_STOCK) +
       '. Reply with ONLY {"name": <fn>, "arguments": {...}}.'},
      {"role": "user", "content": "Check the current price of AAPL."}],
     c_tool_stock_args),
    ("parallel_both", "tool",
     [{"role": "system", "content": "You have tools: " + json.dumps(TOOLS_WEATHER_STOCK) +
       '. Reply with ONLY a JSON list of every call you need.'},
      {"role": "user", "content": "I need the weather in Sao Paulo AND the price of AAPL."}],
     c_parallel_both),
    ("abstain", "abstain",
     [{"role": "system", "content": "You have tools: " + json.dumps(TOOLS_WEATHER_STOCK) +
       '. Only call a tool when one is genuinely needed.'},
      {"role": "user", "content": "What is 2+2? Answer in plain text."}],
     c_abstain),
    ("right_tool", "tool",
     [{"role": "system", "content": "You have tools: " + json.dumps(TOOLS_CALENDAR) +
       '. Reply with ONLY {"name": <fn>, "arguments": {...}}.'},
      {"role": "user", "content": "Schedule a meeting for tomorrow at 10am."}],
     c_right_tool),
]


# --------------------------------------------------------------------------- runner

def ask(client, messages) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"__ERROR__ {exc}"


def run() -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("EVAL_API_KEY", "dummy"))
    jobs = [(tid, cat, msgs, check, rep) for (tid, cat, msgs, check) in TASKS for rep in range(REPEATS)]

    def one(job):
        tid, cat, msgs, check, rep = job
        t0 = time.time()
        answer = ask(client, msgs)
        return {"id": tid, "cat": cat, "rep": rep, "pass": bool(check(answer)),
                "time_s": round(time.time() - t0, 2), "answer_head": answer[:160]}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        rows = list(pool.map(one, jobs))

    cats = sorted({r["cat"] for r in rows})
    summary = {
        "model": MODEL,
        "base_url": BASE_URL,
        "repeats": REPEATS,
        "n": len(rows),
        "passed": sum(r["pass"] for r in rows),
        "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 4),
        "by_category": {c: {"pass": sum(1 for r in rows if r["cat"] == c and r["pass"]),
                            "n": sum(1 for r in rows if r["cat"] == c)} for c in cats},
        "by_task": {t: {"pass": sum(1 for r in rows if r["id"] == t and r["pass"]),
                        "n": sum(1 for r in rows if r["id"] == t)} for t, *_ in TASKS},
    }

    payload = {"summary": summary, "results": rows}
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
