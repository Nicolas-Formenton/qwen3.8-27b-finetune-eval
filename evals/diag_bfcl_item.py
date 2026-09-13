"""Diagnose why BFCL FC items take ~60s each: dump the real prompt tail and the real response.

Builds the prompt exactly as BFCL's QwenFCHandler does for one `irrelevance` entry, prints its
tail (to confirm the <think></think> suppression is present) and the completion size, then
sends it and reports token counts + finish reason.
"""

import json
import sys

from bfcl_eval.constants.eval_config import PROMPT_PATH
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

CATEGORY = sys.argv[1] if len(sys.argv) > 1 else "irrelevance"
REG = sys.argv[2] if len(sys.argv) > 2 else "agent-native-qwen38-base-FC"

with open(PROMPT_PATH / f"BFCL_v4_{CATEGORY}.json") as fh:
    entry = json.loads(fh.readline())

cfg = MODEL_CONFIG_MAPPING[REG]
h = cfg.model_handler(
    model_name=cfg.model_name,
    temperature=0.001,
    registry_name=REG,
    is_fc_model=cfg.is_fc_model,
)

data = h._pre_query_processing_prompting(entry)
data = h.add_first_turn_message_prompting(data, entry["question"][0])

kinds = [k for k in data if k != "message"]
print("keys:", kinds)
raw = data["message"] if "message" in data else data
if isinstance(raw, dict):
    msgs, funcs = raw.get("message"), raw.get("function")
else:
    msgs, funcs = raw, data.get("function")

prompt = h._format_prompt(msgs, funcs or [])

print(f"prompt chars: {len(prompt)}")
print("--- tail 260 chars ---")
print(repr(prompt[-260:]))
print("--- suppression present? ---")
print("yes" if "<think>\n\n</think>\n\n" in prompt else "NO")

n_tok = len(h.tokenizer.tokenize(prompt))
print(f"prompt tokens (hf tokenizer): {n_tok}")

resp = h.client.completions.create(
    model=cfg.model_name,
    temperature=0.001,
    prompt=prompt,
    max_tokens=4096,
    timeout=600,
)
ch = resp.choices[0]
print(f"completion_tokens: {resp.usage.completion_tokens} | finish: {ch.finish_reason}")
print("--- response head 400 ---")
print(repr(ch.text[:400]))
