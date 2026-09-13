import json, os, glob, torch
from safetensors import safe_open

BASE = "/root/models/qwen3.8-27b"
VAR = {
    "oh": "/root/models/merged-oh",
    "xlam": "/root/models/merged-xlam",
    "dbase": "/root/models/final-dbase",
    "doh": "/root/models/final-doh",
    "dxlam": "/root/models/final-dxlam",
}
KEYS = [
    "model.language_model.layers.20.mlp.down_proj.weight",
    "model.language_model.layers.20.linear_attn.in_proj_a.weight",
    "model.language_model.layers.31.mlp.gate_proj.weight",
]

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

def sampled(d, wm, key, step=131):
    sh = os.path.join(d, wm[key])
    with safe_open(sh, framework="pt", device="cpu") as sf:
        t = sf.get_tensor(key)
        rows = t.shape[0]
        sel = t[torch.arange(0, rows, step)]
        return sel.float()

base_wm = index_of(BASE)
print(f"{'model':6s} {'key':34s} {'mean|d|':>10s} {'max|d|':>10s} {'p99|d|':>10s} {'frac>1e-3':>10s} rows_checked")
for name, d in VAR.items():
    wm = index_of(d)
    for k in KEYS:
        if k not in wm:
            print(f"{name:6s} {k.split('.')[-2]+'.'+k.split('.')[-1]:34s} KEY MISSING")
            continue
        a = sampled(BASE, base_wm, k)
        b = sampled(d, wm, k)
        diff = (a - b).abs().flatten()
        print(f"{name:6s} {k.split('.')[-2]+'.'+k.split('.')[-1]:34s} {diff.mean():10.3e} {diff.max():10.3e} {diff.quantile(0.99):10.3e} {(diff>1e-3).float().mean():10.5f} {a.shape[0]}")
