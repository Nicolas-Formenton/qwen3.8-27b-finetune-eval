"""Dump the EXACT prompt BFCL builds for a category, so we can see why models emit
placeholder names like func_name1 instead of the real function name.

No GPU needed: uses the local BFCL install and the same handler the droplet uses.
"""

import json
import sys

from bfcl_eval.constants.eval_config import PROMPT_PATH
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

CATEGORY = sys.argv[1] if len(sys.argv) > 1 else "parallel"
REGISTRY_NAME = sys.argv[2] if len(sys.argv) > 2 else "agent-native-qwen38-base"

path = PROMPT_PATH / f"BFCL_v4_{CATEGORY}.json"
entry = json.loads(open(path).readline())

cfg = MODEL_CONFIG_MAPPING[REGISTRY_NAME]
handler = cfg.model_handler(
    model_name=cfg.model_name,
    temperature=0.001,
    registry_name=REGISTRY_NAME,
    is_fc_model=cfg.is_fc_model,
)

inference_data = handler._pre_query_processing_prompting(entry)
inference_data = handler.add_first_turn_message_prompting(inference_data, entry["question"][0])
prompt = inference_data["prompt"]

print(f"=== category {CATEGORY} / {REGISTRY_NAME} (is_fc_model={cfg.is_fc_model}) ===")
print(f"prompt type: {type(prompt).__name__}")
text = prompt if isinstance(prompt, str) else json.dumps(prompt, indent=1)
print(f"chars: {len(text)}")
print("----- first 2500 chars -----")
print(text[:2500])
print("----- last 1200 chars -----")
print(text[-1200:])
print("----- placeholder scan -----")
for token in ("func_name1", "func_name2", "func_", "<tool_call>"):
    print(f"  {token!r}: {text.count(token)}")
