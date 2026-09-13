"""Run BFCL's ast_checker case-by-case and summarise failures by error type.

Use when the self-test score differs from expectation: this tells you whether the
gap is in the fabricated responses or in the checker's semantics.
"""

import collections
import json
import os
import pathlib
import sys

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH, VERSION_PREFIX
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.utils import default_decode_ast_prompting

CAT = sys.argv[1] if len(sys.argv) > 1 else "simple_python"
MODE = sys.argv[2] if len(sys.argv) > 2 else "clean"
MODEL = "agent-native-qwen38-base"
ROOT = pathlib.Path(os.environ["BFCL_PROJECT_ROOT"])
GROUP = "non_live"


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


suffix = "_result.json" if MODE == "clean" else "_result_corrupt.json"
rows = load(ROOT / "result" / MODEL / GROUP / f"{VERSION_PREFIX}_{CAT}{suffix}")
test = {t["id"]: t for t in load(PROMPT_PATH / f"{VERSION_PREFIX}_{CAT}.json")}
gold = {g["id"]: g for g in load(POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{CAT}.json")}

errors = collections.Counter()
examples = collections.defaultdict(list)
n_ok = 0
for row in rows:
    tid = row["id"]
    entry = test[tid]
    try:
        decoded = default_decode_ast_prompting(row["result"], ReturnFormat.PYTHON)
    except Exception as exc:                                    # parse failure
        errors[f"parse:{type(exc).__name__}"] += 1
        examples[f"parse:{type(exc).__name__}"].append((tid, str(exc)[:120], row["result"][:120]))
        continue

    result = ast_checker(
        entry["function"],
        decoded,
        gold[tid]["ground_truth"],
        Language.PYTHON,
        CAT,
        MODEL,
    )
    if result["valid"]:
        n_ok += 1
    else:
        et = result["error_type"]
        errors[et] += 1
        if len(examples[et]) < 2:
            examples[et].append(
                (tid, (result["error"][0] if result["error"] else "")[:150], row["result"][:150])
            )

print(f"{CAT} [{MODE}]  correct={n_ok}/{len(rows)}  accuracy={n_ok/len(rows):.4f}")
print("--- failures by error_type ---")
for et, n in errors.most_common():
    print(f"  {et}: {n}")
    for tid, err, raw in examples[et]:
        print(f"      {tid}\n        err: {err}\n        out: {raw}")
