import json, os, sys
from bfcl_eval.constants.eval_config import PROMPT_PATH, POSSIBLE_ANSWER_PATH

CAT = sys.argv[1] if len(sys.argv) > 1 else "simple_python"

def load(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

test = load(os.path.join(PROMPT_PATH, f"BFCL_v4_{CAT}.json"))
gold = load(os.path.join(POSSIBLE_ANSWER_PATH, f"BFCL_v4_{CAT}.json"))
print(f"=== {CAT}: test={len(test)} gold={len(gold)}")
print("--- test[0] keys:", list(test[0].keys()))
print("--- test[0].id:", test[0]["id"])
q = test[0]["question"]
print("--- question[0][0] role:", q[0][0]["role"])
print("--- question content head:", str(q[0][0]["content"])[:300].replace("\n", " | "))
print("--- functions[0] name:", test[0]["function"][0]["name"])
print("--- gold[0]:", json.dumps(gold[0])[:400])
