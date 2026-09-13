"""Disable Qwen reasoning on the BFCL FC path, and prove it.

Defect: `QwenFCHandler._format_prompt` builds the prompt by string concatenation and ends with
`<|im_start|>assistant\\n`, never emitting the empty `<think></think>` block that the model's own
`chat_template.jinja` emits when `enable_thinking=false`. The handler then requests the full
4096-token completion budget (see `_query_prompting`), so the model spends the whole budget on
reasoning before it ever writes the `<tool_call>` block. Measured effect on `all_scoring`: ~30s
per item instead of ~0.35s, i.e. ~17 GPU-hours per model instead of ~30 minutes.

The empty-then-closed thinking block below is copied verbatim from the model's own
chat_template.jinja (the `enable_thinking is false` branch), so the prompt is byte-identical to
what the model sees in normal serving.

Set BFCL_ENABLE_THINKING=1 to keep reasoning on (used to A/B the cost).

Usage: python fix_bfcl_thinking.py [--show]
"""

import argparse
import importlib.util
import pathlib
import sys

MARK = "# ---- allow-toolkit: thinking suppression (BEGIN) ----"
END = "# ---- allow-toolkit: thinking suppression (END) ----"

OLD = '        formatted_prompt += "<|im_start|>assistant\\n"\n'

NEW = (
    OLD
    + "        " + MARK + "\n"
    + '        if os.getenv("BFCL_ENABLE_THINKING", "").strip().lower() not in ("1", "true", "yes"):\n'
    + '            formatted_prompt += "<think>\\n\\n</think>\\n\\n"\n'
    + "        " + END + "\n"
)


def target() -> pathlib.Path:
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.origin:
        sys.exit("bfcl_eval is not importable in this interpreter")
    return pathlib.Path(spec.origin).parent / "model_handler" / "local_inference" / "qwen_fc.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    path = target()
    text = path.read_text(encoding="utf-8")

    if args.show:
        for i, line in enumerate(text.splitlines(), 1):
            if "assistant\\n" in line or MARK in line:
                print(f"{i}: {line}")
        return

    if MARK in text:
        print(f"ja corrigido: {path}")
    else:
        if text.count(OLD) != 1:
            sys.exit(f"ancora ambigua: {text.count(OLD)} ocorrencias de {OLD!r}")
        if "\nimport os\n" not in text:
            text = text.replace("import json\n", "import json\nimport os\n", 1)
        text = text.replace(OLD, NEW, 1)
        path.write_text(text, encoding="utf-8")
        print(f"corrigido: {path}")

    fresh = path.read_text(encoding="utf-8")
    print(f"import os presente: {int(chr(10) + 'import os' + chr(10) in fresh)}")
    print(f"bloco de supressao: {fresh.count(MARK)}")
    print(f"env gate: {int('BFCL_ENABLE_THINKING' in fresh)}")


if __name__ == "__main__":
    main()
