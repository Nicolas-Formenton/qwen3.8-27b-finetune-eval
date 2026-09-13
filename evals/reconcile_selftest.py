"""Reconcile the corrupt self-test: find cases we flagged as corrupted that the
checker nevertheless accepted. Explains any gap between our predicted accuracy
and BFCL's reported accuracy, so the self-test conclusion is precise.
"""

import json
import os
import pathlib

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH, VERSION_PREFIX
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.utils import default_decode_ast_prompting

CAT, MODEL, GROUP, EVERY = "simple_python", "agent-native-qwen38-base", "non_live", 4
ROOT = pathlib.Path(os.environ["BFCL_PROJECT_ROOT"])


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


rows = {r["id"]: r["result"] for r in load(ROOT / "result" / MODEL / GROUP / f"{VERSION_PREFIX}_{CAT}_result_corrupt.json")}
test = load(PROMPT_PATH / f"{VERSION_PREFIX}_{CAT}.json")
gold = {g["id"]: g for g in load(POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{CAT}.json")}

accepted = []
for i, case in enumerate(test):
    if i % EVERY:
        continue
    tid = case["id"]
    try:
        decoded = default_decode_ast_prompting(rows[tid], ReturnFormat.PYTHON)
        res = ast_checker(case["function"], decoded, gold[tid]["ground_truth"], Language.PYTHON, CAT, MODEL)
    except Exception as exc:
        res = {"valid": False, "error": [f"parse: {exc}"]}
    if res["valid"]:
        fn = case["function"][0]
        props = fn["parameters"].get("properties", {})
        accepted.append((tid, rows[tid], fn["parameters"].get("required", []), list(props)))

print(f"corrupted-batch cases accepted by the checker: {len(accepted)}")
for tid, out, req, props in accepted:
    optional = [p for p in props if p not in req]
    print(f"  {tid}\n    out={out}\n    required={req}  optional={optional}")
