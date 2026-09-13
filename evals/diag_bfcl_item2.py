"""Measure the real prompt size, response length and latency per BFCL item (FC handler).

Answers: is the slowness the model (long generations) or the harness (retries)?
"""

import json
import sys
import time

from bfcl_eval.constants.eval_config import PROMPT_PATH
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

CAT = sys.argv[1] if len(sys.argv) > 1 else "irrelevance"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
REG = "agent-native-qwen38-base-FC"

with open(PROMPT_PATH / ("BFCL_v4_" + CAT + ".json")) as fh:
    lines = fh.readlines()

cfg = MODEL_CONFIG_MAPPING[REG]
h = cfg.model_handler(
    model_name=cfg.model_name,
    temperature=0.001,
    registry_name=REG,
    is_fc_model=cfg.is_fc_model,
)

for raw in lines[:N]:
    entry = json.loads(raw)
    d = h._pre_query_processing_prompting(entry)
    d = h.add_first_turn_message_prompting(d, entry["question"][0])
    msg = d.get("message") or d
    if isinstance(msg, dict) and "message" in msg:
        msg = msg["message"]
    prompt = h._format_prompt(msg, d.get("function") or [])
    t0 = time.time()
    r = h.client.completions.create(
        model=cfg.model_name,
        temperature=0.001,
        prompt=prompt,
        max_tokens=4096,
        timeout=600,
    )
    dt = time.time() - t0
    ch = r.choices[0]
    print(
        "%-30s prompt=%6dc out=%5d tok %6.1fs finish=%s"
        % (entry["id"][:28], len(prompt), r.usage.completion_tokens, dt, ch.finish_reason)
    )
    print("      head:", repr(ch.text[:170]))
