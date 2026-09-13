"""Which variant actually wins, with significance tests rather than eyeballing.

Reads the BFCL v4 score files (JSONL: line 1 = summary with accuracy + counts)
and reports:
  * per-category accuracy
  * the mean across the four categories (what a leaderboard aggregate approximates)
  * two-proportion z-tests for the comparisons that matter, so we only claim
    differences that survive their confidence interval

Usage:  python evals/score_variants.py
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES = HERE / "bfcl-scores"

CATS = ["multiple", "parallel", "parallel_multiple", "irrelevance"]
VARIANTS = {
    "agent-native-qwen38-base": "base",
    "agent-native-qwen38-oh": "+OH",
    "agent-native-qwen38-xlam": "+xLAM",
    "agent-native-qwen38-dbase": "D-base",
    "agent-native-qwen38-doh": "D-OH",
    "agent-native-qwen38-dxlam": "D-xLAM",
}
MODES = {"": "prompt", "-FC": "FC"}


def stats(model: str, cat: str, mode_suffix: str):
    p = SCORES / f"{model}{mode_suffix}" / "non_live" / f"BFCL_v4_{cat}_score.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        head = json.loads(fh.readline())
    return head["accuracy"], head["correct_count"], head["total_count"]


def z_test(k1, n1, k2, n2):
    """Two-proportion z-test. Returns (diff_pp, z, p_two_sided)."""
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 100 * (p1 - p2), float("inf"), 0.0
    z = (p1 - p2) / se
    # two-sided p via the complementary error function
    p = math.erfc(abs(z) / math.sqrt(2))
    return 100 * (p1 - p2), z, p


def main() -> None:
    table = {}
    for model, label in VARIANTS.items():
        table[label] = {}
        for mode_suffix, mode in MODES.items():
            for cat in CATS:
                s = stats(model, cat, mode_suffix)
                if s:
                    table[label].setdefault(mode, {})[cat] = s

    for mode in ("FC", "prompt"):
        print(f"\n===== mode = {mode} =====")
        print(f"{'variant':8s} " + " ".join(f"{c[:9]:>9s}" for c in CATS) + f" {'mean':>8s}  {'n':>4s}")
        means = {}
        for label, modes in table.items():
            if mode not in modes:
                continue
            row, accs = [], []
            for cat in CATS:
                a, k, n = modes[mode][cat]
                row.append(f"{a*100:9.1f}")
                accs.append(a)
            means[label] = sum(accs) / len(accs)
            n = modes[mode][CATS[0]][2]
            print(f"{label:8s} " + " ".join(row) + f" {means[label]*100:8.1f}  {n:4d}")
        best = max(means, key=means.get)
        print(f"  -> best mean: {best} ({means[best]*100:.1f}%)")

    # the comparisons that decide the verdict, in FC mode (the capability read)
    print("\n===== significance (FC mode, two-proportion z-test) =====")
    base = table["base"]["FC"]
    for label in ("+OH", "+xLAM", "D-base", "D-OH", "D-xLAM"):
        if label not in table:
            continue
        print(f"\n  base vs {label}:")
        for cat in CATS:
            _, k1, n1 = base[cat]
            _, k2, n2 = table[label]["FC"][cat]
            d, z, p = z_test(k1, n1, k2, n2)
            verdict = "SIGNIFICANT" if p < 0.05 else "not significant (within noise)"
            print(f"    {cat:18s} {d:+6.1f}pp   z={z:6.2f}  p={p:.4f}   {verdict}")

    # the headline claim: does the distill recover abstention?
    print("\n===== headline: abstention (irrelevance), FC mode =====")
    for a, b in (("base", "+xLAM"), ("+xLAM", "D-xLAM"), ("base", "D-xLAM"), ("base", "D-OH")):
        _, k1, n1 = table[a]["FC"]["irrelevance"]
        _, k2, n2 = table[b]["FC"]["irrelevance"]
        d, z, p = z_test(k2, n2, k1, n1)
        print(f"  {b:8s} vs {a:8s}: {d:+6.1f}pp  z={z:6.2f}  p={p:.6f}  "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")


if __name__ == "__main__":
    main()
