import json, os, sys
from safetensors import safe_open

AD = sys.argv[1]
BASE = "/root/models/qwen3.8-27b"
bwm = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))["weight_map"]

with safe_open(os.path.join(AD, "adapter_model.safetensors"), framework="pt", device="cpu") as sf:
    keys = list(sf.keys())

mods = {}
for k in keys:
    kk = k
    for pref in ("base_model.model.", "base_model."):
        if kk.startswith(pref):
            kk = kk[len(pref):]
    for suf in (".lora_A.weight", ".lora_B.weight"):
        if kk.endswith(suf):
            mods.setdefault(kk[: -len(suf)] + ".weight", set()).add(suf.split(".")[1])

with_lm = [m for m in mods if ".language_model." in m]
without_lm = [m for m in mods if ".language_model." not in m]
ok = [m for m in mods if m in bwm]
print(f"adapter modules: {len(mods)} | in base index: {len(ok)}")
print(f"contain .language_model.: {len(with_lm)} (in base: {sum(1 for m in with_lm if m in bwm)})")
print(f"WITHOUT .language_model.: {len(without_lm)} (in base: {sum(1 for m in without_lm if m in bwm)})")
print("examples without language_model:")
for m in without_lm[:8]:
    print("   ", m)
print("\nbase has 'model.layers.56.mlp.down_proj.weight'?", "model.layers.56.mlp.down_proj.weight" in bwm)
print("base keys starting with 'model.' but not 'model.language_model'/'model.visual':")
oth = [k for k in bwm if k.startswith("model.") and "language_model" not in k and "visual" not in k]
print(f"   {len(oth)} -> {oth[:6]}")
print("\nlayer index ranges: adapter without_lm layers =",
      sorted({int(m.split("layers.")[1].split(".")[0]) for m in without_lm})[:12],
      "... max", max([int(m.split("layers.")[1].split(".")[0]) for m in without_lm], default=None))
print("adapter with_lm layers count:", len({int(m.split("layers.")[1].split(".")[0]) for m in with_lm}))
