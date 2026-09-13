"""Verify a merged checkpoint against the adapter math:
   actual  = merged - base   (sampled rows)
   expect  = scale * (B @ A) (sampled rows)
If they agree, the adapter was applied exactly; if actual ~ 0 but expect != 0, the merge was a no-op.
"""
import json, os, sys, glob
import torch
from safetensors import safe_open

base_dir, adapter_dir, merged_dir, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STEP = 131

def index_of(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                wm[k] = os.path.basename(f)
    return wm

cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
scale = float(cfg.get("lora_alpha", 16)) / float(cfg.get("r", 16))

with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt", device="cpu") as sf:
    pairs = {}
    for k in sf.keys():
        kk = k
        for pref in ("base_model.model.", "base_model."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        for suf, part in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
            if kk.endswith(suf):
                mod = kk[: -len(suf)] + ".weight"
                pairs.setdefault(mod, {})[part] = sf.get_tensor(k).float()

mods = [m for m, v in pairs.items() if set(v) == {"A", "B"}]
mods.sort(key=lambda m: (int(m.split("layers.")[1].split(".")[0]) if ".layers." in m else 0, m))
sample = mods[:: max(1, len(mods) // 8)][:8]

def alt_names(k):
    """PEFT files name layers as model.layers.N.* while the HF checkpoint uses
    model.language_model.layers.N.*  -> try both spellings."""
    out = [k]
    if ".language_model.layers." in k:
        out.append(k.replace(".language_model.layers.", ".layers."))
    elif k.startswith("model.layers."):
        out.append(k.replace("model.layers.", "model.language_model.layers.", 1))
    return out

def pick(ix, k):
    for cand in alt_names(k):
        if cand in ix:
            return cand
    return None

bwm, mwm = index_of(base_dir), index_of(merged_dir)
print(f"=== {label}: scale={scale} modules={len(mods)} sampling {len(sample)}")
worst = 0.0
for m in sample:
    bk, mk = pick(bwm, m), pick(mwm, m)
    if bk is None or mk is None:
        print(f"   {m[-50:]:50s} MISSING in base/merged index"); continue
    with safe_open(os.path.join(base_dir, bwm[bk]), framework="pt", device="cpu") as sf:
        Wb = sf.get_tensor(bk)
    with safe_open(os.path.join(merged_dir, mwm[mk]), framework="pt", device="cpu") as sf:
        Wm = sf.get_tensor(mk)
    A, B = pairs[m]["A"], pairs[m]["B"]
    exp = (B @ A) * scale
    s = torch.arange(0, Wb.shape[0], STEP)
    actual = (Wm[s].float() - Wb[s].float())
    expected = exp[s]
    err = (actual - expected).abs().max().item()
    worst = max(worst, err)
    rat = (actual.abs().max().item() / max(expected.abs().max().item(), 1e-12))
    print(f"   {m[-50:]:50s} |actual|max={actual.abs().max().item():.3e} |expected|max={expected.abs().max().item():.3e} ratio={rat:.4f} err={err:.3e}")
print(f"VERDICT {label}: worst|actual-expected| = {worst:.3e} -> {'APPLIED CORRECTLY' if worst < 1e-3 else ('NO-OP / WRONG' if worst > 1e-6 else '???')}")
