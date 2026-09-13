"""Offline validation of every checker in eval_suite_v2 — no GPU, no endpoint needed.

For each task we supply a hand-written answer that MUST pass and one that MUST fail.
The negative examples are deliberately adversarial in the way that matters:

  * tool_weather_args  -> correct function name, WRONG argument value
  * parallel_both      -> only one of the two required calls
  * abstain            -> emits a tool call when none is needed
  * no_preamble        -> right number, but with a greeting in front

Those four are exactly what v1 could not detect, so they are the proof that the
metric now has resolution. Exits non-zero on any mismatch, so it can gate a GPU run.
"""

import sys

from eval_suite_v2 import (
    c_abstain, c_bullets5, c_codefence, c_json_basic, c_json_typed, c_max20,
    c_no_preamble, c_parallel_both, c_right_tool, c_tool_stock_args,
    c_tool_weather_args, c_tsv_3x2,
)

LONG = ("LoRA is a parameter efficient fine tuning technique that keeps the base model "
        "weights frozen while training two small low rank matrices per targeted layer "
        "which dramatically cuts memory use and training time for large language models")

CASES = {
    "json_basic": (c_json_basic,
        '{"ok": true, "items": ["a", "b", "c"]}',
        '{"ok": false, "items": ["a", "b", "c"]}'),
    "json_typed": (c_json_typed,
        '{"status": "ok", "total": 42}',
        '{"status": "ok", "total": "42"}'),
    "bullets5": (c_bullets5,
        "- razão um\n- razão dois\n- razão três\n- razão quatro\n- razão cinco",
        "- razão um\n- razão dois\n- razão três\n- razão quatro"),
    "codefence": (c_codefence,
        "```python\ndef add(a, b):\n    return a + b\n```",
        "def add(a, b):\n    return a + b"),
    "max20": (c_max20,
        "LoRA trains small adapter matrices instead of all weights, cutting memory a lot.",
        LONG),
    "no_preamble": (c_no_preamble, "4", "Sure! The answer is 4."),
    "tsv_3x2": (c_tsv_3x2,
        "Sao Paulo\t12\nRio de Janeiro\t7\nTokyo\t14",
        "Sao Paulo\t12\nRio de Janeiro\t7"),
    "tool_weather_args": (c_tool_weather_args,
        '{"name": "get_weather", "arguments": {"city": "Sao Paulo"}}',
        '{"name": "get_weather", "arguments": {"city": "Rio de Janeiro"}}'),
    "tool_stock_args": (c_tool_stock_args,
        '{"name": "get_stock", "arguments": {"ticker": "AAPL"}}',
        '{"name": "get_stock", "arguments": {"ticker": "MSFT"}}'),
    "parallel_both": (c_parallel_both,
        '[{"name": "get_weather", "arguments": {"city": "Sao Paulo"}}, '
        '{"name": "get_stock", "arguments": {"ticker": "AAPL"}}]',
        '{"name": "get_weather", "arguments": {"city": "Sao Paulo"}}'),
    "abstain": (c_abstain,
        "The answer is 4.",
        '{"name": "get_weather", "arguments": {"city": "Sao Paulo"}}'),
    "right_tool": (c_right_tool,
        '{"name": "create_calendar_event", "arguments": {"title": "Meeting", "time": "tomorrow 10am"}}',
        '{"name": "send_email", "arguments": {"to": "team@example.com", "body": "meeting"}}'),
}

# thinking-mode robustness: a correct answer wrapped in a reasoning block must still pass
THINKING_CASES = [
    ("json_basic", c_json_basic, '<think>I should emit exactly the requested JSON.</think>\n{"ok": true, "items": ["a","b","c"]}'),
    ("tool_weather_args", c_tool_weather_args,
     '<think>The user wants Sao Paulo weather, so get_weather with city=Sao Paulo.</think>\n{"name": "get_weather", "arguments": {"city": "Sao Paulo"}}'),
]


def main() -> None:
    failures = []
    print(f"{'task':22s} {'pos':>5s} {'neg':>5s}")
    for task, (fn, positive, negative) in CASES.items():
        got_pos, got_neg = bool(fn(positive)), bool(fn(negative))
        flag = "" if (got_pos and not got_neg) else "  <-- PROBLEM"
        print(f"{task:22s} {str(got_pos):>5s} {str(got_neg):>5s}{flag}")
        if not got_pos:
            failures.append(f"{task}: correct answer rejected")
        if got_neg:
            failures.append(f"{task}: wrong answer accepted")

    print("--- thinking-block robustness ---")
    for task, fn, answer in THINKING_CASES:
        ok = bool(fn(answer))
        print(f"{task:22s} {str(ok):>5s}")
        if not ok:
            failures.append(f"{task}: correct answer rejected when wrapped in <think>")

    total = len(CASES) * 2 + len(THINKING_CASES)
    if failures:
        print(f"\nSUITE CHECKERS INVALID ({len(failures)}/{total} checks failed):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"\nSUITE CHECKERS VALIDATED ({total}/{total} checks)")
    print("discriminating cases proven: wrong-argument, single-of-two-parallel,")
    print("tool-called-when-abstention-required, preamble-before-answer, thinking-wrapped.")


if __name__ == "__main__":
    main()
