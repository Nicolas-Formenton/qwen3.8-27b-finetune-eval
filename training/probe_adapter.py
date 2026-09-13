import json, os, glob
from safetensors import safe_open

AD = "/root/adapters/xlam-adapter"
BASE = "/root/models/qwen3.8-27b"

# adapter tensor keys
f = [x for x in sorted(os.listdir(AD)) if x.endswith(".safetensors")][0]
with safe_open(os.path.join(AD, f), framework="pt", device="cpu") as sf:
    ak = list(sf.keys())
print("adapter files:", os.listdir(AD))
print("adapter tensor count:", len(ak))
print("first 6 adapter keys:")
for k in ak[:6]:
    print("   ", k)

# base keys
idx = os.path.join(BASE, "model.safetensors.index.json")
bwm = json.load(open(idx))["weight_map"]
bkeys = set(bwm.keys())

# try to map adapter keys -> base keys
def strip(k):
    for pref in ["base_model.model.", "base_model."]:
        if k.startswith(pref):
            k = k[len(pref):]
    for suf in [".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight", ".lora_B.default.weight"]:
        if k.endswith(suf):
            k = k[: -len(suf)] + ".weight"
    return k

mapped = sum(1 for k in ak if strip(k) in bkeys)
print(f"\nmapped adapter keys -> base: {mapped}/{len(ak)}")
miss = [k for k in ak if strip(k) not in bkeys][:5]
print("unmapped examples:", miss)
# what do the corresponding base keys look like?
sample = strip(ak[0])
print("sample mapped key:", sample, "| in base:", sample in bkeys)
cands = [b for b in bkeys if sample.split(".lora")[0][-40:] in b][:5]
print("base candidates:", cands)
# cross-check: does the adapter key (as-is) exist in base?
print("\nadapter key as-is in base?", ak[0] in bkeys)
print("keys in base containing 'self_attn.q_proj.weight' (first 3):", [b for b in bkeys if "self_attn.q_proj.weight" in b][:3])
print("adapter keys containing 'self_attn.q_proj' (first 3):", [k for k in ak if "self_attn.q_proj" in k][:3])
