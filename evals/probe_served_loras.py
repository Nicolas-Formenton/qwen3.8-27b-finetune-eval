"""Prove each served LoRA module is actually applied, before running any benchmark.

Yesterday's silent-merge bug produced a checkpoint bit-identical to base and the
benchmark still "ran". Serving the adapter at request time removes that failure
mode, but only if the module is really wired in: so compare the base output against
each adapter output on the same prompt and count how many differ.
"""

import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

MODELS = ["qwen38-base", "qwen38-oh", "qwen38-xlam", "qwen38-dbase", "qwen38-doh", "qwen38-dxlam"]

PROMPT = [
    {
        "role": "user",
        "content": (
            "You have access to functions. When a tool is needed, reply with ONLY a JSON list: "
            '[{"name": "<fn>", "arguments": {...}}].\n'
            "Functions:\n"
            '[{"name": "get_weather", "description": "Current weather", "parameters": '
            '{"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}, '
            '"required": ["city"]}}, '
            '{"name": "send_email", "description": "Send an email", "parameters": '
            '{"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}}, '
            '"required": ["to", "subject"]}}]\n\n'
            "Query: What is the weather in Curitiba in celsius, and email a summary to ana@example.com?"
        ),
    }
]


def ask(model, temperature=0.0, max_tokens=300):
    r = client.chat.completions.create(
        model=model,
        messages=PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (r.choices[0].message.content or "").strip()


out = {}
for m in MODELS:
    try:
        out[m] = ask(m)
    except Exception as e:
        out[m] = f"__ERROR__ {type(e).__name__}: {str(e)[:120]}"

base = out["qwen38-base"]
print(f"base output ({len(base)} chars): {base[:220]}\n")
print(f"{'model':<14} {'differs from base':<18} head")
for m in MODELS:
    if m == "qwen38-base":
        continue
    same = out[m] == base
    print(f"{m:<14} {str(not same):<18} {out[m][:110]!r}")

differing = [m for m in MODELS if m != "qwen38-base" and out[m] != base]
print(f"\nadapters producing different output than base: {len(differing)}/{len(MODELS)-1} -> {differing}")
print("LORA_MODULES_APPLIED" if len(differing) >= 3 else "!! SUSPECT: too many identical outputs")

with open("/root/lora_probe.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved /root/lora_probe.json")
