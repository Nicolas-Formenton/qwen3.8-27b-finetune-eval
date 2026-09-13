"""Consolidate the BFCL v4 grid into one results file + a readable matrix.

Reads the per-category score JSONs produced by BFCL on the droplet (copied into
evals/bfcl-scores/) and writes results/eval-04-bfcl-v4.json plus a markdown table.

Usage:  python evals/consolidate_bfcl.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCORES = HERE / "bfcl-scores"
OUT = REPO / "results" / "eval-04-bfcl-v4.json"

CATS = ["multiple", "parallel", "parallel_multiple", "irrelevance"]
# registry name -> (short label, kind) ; kind: base | sft | distill
VARIANTS = {
    "agent-native-qwen38-base": ("base", "base"),
    "agent-native-qwen38-oh": ("+OH", "sft"),
    "agent-native-qwen38-xlam": ("+xLAM", "sft"),
    "agent-native-qwen38-dbase": ("D-base", "distill"),
    "agent-native-qwen38-doh": ("D-OH", "distill"),
    "agent-native-qwen38-dxlam": ("D-xLAM", "distill"),
}
MODES = {"": "prompt", "-FC": "FC"}


def read_score(model: str, cat: str, mode_suffix: str) -> float | None:
    p = SCORES / f"{model}{mode_suffix}" / "non_live" / f"BFCL_v4_{cat}_score.json"
    if not p.exists():
        return None
    try:
        # BFCL writes these as JSONL: line 1 is the summary, following lines are
        # per-item verdicts (which are what makes item-level forensics possible).
        with p.open(encoding="utf-8") as fh:
            return json.loads(fh.readline())["accuracy"]
    except Exception:
        return None


def main() -> None:
    table: dict[str, dict[str, dict[str, float | None]]] = {}
    for model, (label, kind) in VARIANTS.items():
        table[label] = {"kind": kind, "prompt": {}, "FC": {}}
        for mode_suffix, mode in MODES.items():
            for cat in CATS:
                table[label][mode][cat] = read_score(model, cat, mode_suffix)

    payload = {
        "benchmark": "BFCL v4 (Berkeley Function Calling Leaderboard), run locally",
        "harness": "bfcl-eval (PyPI), prompt mode + FC mode, --partial-eval per category",
        "serving": "vLLM ROCm, base + 5 LoRA adapters loaded as modules (no merge)",
        "decoding": "greedy/temperature=0 (BFCL default for these handlers)",
        "result": table,
        "notes": [
            "prompt mode = the benchmark's own dialect ([func_name1(a=1)]); FC mode = native tool-call template.",
            "The two modes are NOT comparable to each other; the gap is itself the finding.",
            "irrelevance = abstention: correct behaviour is to NOT emit a tool call.",
            "Caveat: in prompt mode a model that fails to produce parseable calls can score 'correct' on "
            "irrelevance for the wrong reason (silence instead of an informed refusal). Read the two modes together.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    # markdown matrix for the README
    def fmt(v):
        return "--" if v is None else f"{v*100:.1f}"

    for mode in ("FC", "prompt"):
        print(f"\n### mode = {mode}\n")
        print("| variant | " + " | ".join(CATS) + " |")
        print("|---|" + "---|" * len(CATS))
        for label, data in table.items():
            row = " | ".join(fmt(data[mode][c]) for c in CATS)
            print(f"| {label} | {row} |")


if __name__ == "__main__":
    main()
