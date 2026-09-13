import sys, time, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = sys.argv[1]
ADAPTER = sys.argv[2]
OUT = sys.argv[3]

t0 = time.time()
print("loading base (CPU)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(BASE)
print("attaching adapter...", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER)
print("merging...", flush=True)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("MERGE_DONE", OUT, "in", round(time.time() - t0), "s", flush=True)
