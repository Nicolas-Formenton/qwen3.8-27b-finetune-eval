"""Validate the BFCL harness end-to-end BEFORE spending GPU.

Regenerates both fabricated response files, counts the rows that actually differ,
then runs BFCL's own scorer on each and checks the reported accuracy matches the
prediction. Exits non-zero on mismatch so it can gate a GPU run.

  python validate_harness.py [category]

Prediction for the corrupt file is (N - changed) / N, where `changed` is counted
from the files themselves rather than from generator bookkeeping.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

CATEGORY = sys.argv[1] if len(sys.argv) > 1 else "simple_python"
MODEL = "agent-native-qwen38-base"
GROUP = "non_live"
ROOT = pathlib.Path(
    os.environ.get("BFCL_PROJECT_ROOT", r"evals/bfcl-project")
)
HERE = pathlib.Path(__file__).parent
PY = HERE / "bfcl-env" / "Scripts" / "python.exe"


def read_rows(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r["result"] for r in (json.loads(l) for l in fh if l.strip())}


def score_with_bfcl(result_name: str) -> float:
    """Point the standard result file at `result_name`, run bfcl evaluate, read the score."""
    src = ROOT / "result" / MODEL / GROUP / result_name
    std = ROOT / "result" / MODEL / GROUP / f"BFCL_v4_{CATEGORY}_result.json"
    if src != std:
        std.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    proc = subprocess.run(
        [str(HERE / "bfcl-env" / "Scripts" / "bfcl.exe"),
         "evaluate", "--model", MODEL, "--test-category", CATEGORY, "--partial-eval"],
        capture_output=True, text=True, cwd=str(HERE),
    )
    blob = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"Accuracy: ([\d.]+)%", blob)
    if not match:
        sys.exit(f"could not parse accuracy (exit {proc.returncode}) from:\n{blob}")
    return float(match.group(1)) / 100.0


def main() -> None:
    env = dict(os.environ, BFCL_PROJECT_ROOT=str(ROOT))
    print("regenerating fabricated responses ...")
    subprocess.run([str(PY), str(HERE / "selftest_scorer.py")], check=True, env=env, capture_output=True)

    clean = read_rows(ROOT / "result" / MODEL / GROUP / f"BFCL_v4_{CATEGORY}_result.json")
    corrupt = read_rows(ROOT / "result" / MODEL / GROUP / f"BFCL_v4_{CATEGORY}_result_corrupt.json")
    n = len(clean)
    changed = sum(1 for tid in clean if clean[tid] != corrupt[tid])
    print(f"n={n}  rows actually altered in the corrupt file: {changed}")

    expected_clean, expected_corrupt = 1.0, (n - changed) / n
    got_clean = score_with_bfcl(f"BFCL_v4_{CATEGORY}_result.json")
    got_corrupt = score_with_bfcl(f"BFCL_v4_{CATEGORY}_result_corrupt.json")

    print(f"clean    predicted {expected_clean:.4f}  reported {got_clean:.4f}")
    print(f"corrupt  predicted {expected_corrupt:.4f}  reported {got_corrupt:.4f}")

    ok = abs(got_clean - expected_clean) < 1e-9 and abs(got_corrupt - expected_corrupt) < 1e-9
    verdict = {
        "category": CATEGORY,
        "model": MODEL,
        "n": n,
        "rows_altered": changed,
        "expected_clean": expected_clean,
        "reported_clean": got_clean,
        "expected_corrupt": expected_corrupt,
        "reported_corrupt": got_corrupt,
        "harness_validated": ok,
    }
    json.dump(verdict, open(ROOT / "harness_validation.json", "w"), indent=1)
    print("HARNESS VALIDATED" if ok else "MISMATCH -- investigate before any GPU run")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
