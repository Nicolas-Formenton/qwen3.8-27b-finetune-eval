import json, os, glob
from safetensors import safe_open

BASE = "/root/models/qwen3.8-27b"
OTHERS = {
    "oh": "/root/models/merged-oh",
    "xlam": "/root/models/merged-xlam",
    "dbase": "/root/models/final-dbase",
    "doh": "/root/models/final-doh",
    "dxlam": "/root/models/final-dxlam",
}

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

def load_slice(d, wm, key, rows=2, cols=8):
    sh = os.path.join(d, wm[key]) if not os.path.isabs(wm[key]) else wm[key]
    with safe_open(sh, framework="pt", device="cpu") as sf:
        t = sf.get_tensor(key)
        return t[:rows, :cols].float().clone()

base_wm = index_of(BASE)
print("base keys:", len(base_wm))

KEYS = [
    "model.language_model.layers.20.linear_attn.in_proj_a.weight",
    "model.language_model.layers.20.mlp.down_proj.weight",
    "model.language_model.layers.10.self_attn.q_proj.weight",
]

for name, d in OTHERS.items():
    wm = index_of(d)
    missing = [k for k in base_wm if k not in wm]
    extra = [k for k in wm if k not in base_wm]
    print(f"\n=== {name}: {len(wm)} keys | missing vs base: {len(missing)} | extra: {len(extra)}")
    if missing:
        print("   ex missing:", missing[:5])
    if extra:
        print("   ex extra  :", extra[:3])
    for k in KEYS:
        if k in wm and k in base_wm:
            try:
                a = load_slice(BASE, base_wm, k)
                b = load_slice(d, wm, k)
                diff = (a - b).abs().max().item()
                print(f"   {k.split('.')[-2]+'.'+k.split('.')[-1]:28s} maxdiff={diff:.6g} base[0,:4]={[round(x,5) for x in a[0,:4].tolist()]} other={[round(x,5) for x in b[0,:4].tolist()]}")
            except Exception as e:
                print(f"   {k}: ERR {e}")
