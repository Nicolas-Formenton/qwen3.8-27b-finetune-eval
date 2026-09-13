import json, os
from safetensors import safe_open

KEY = "model.layers.20.self_attn.q_proj.weight"
dirs = {
    "base":  "/root/models/qwen3.8-27b",
    "oh":    "/root/models/merged-oh",
    "xlam":  "/root/models/merged-xlam",
    "dbase": "/root/models/final-dbase",
    "doh":   "/root/models/final-doh",
    "dxlam": "/root/models/final-dxlam",
}

def shard_for(d, key):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return os.path.join(d, wm[key])
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), framework="pt", device="cpu") as sf:
                if key in sf.keys():
                    return os.path.join(d, f)
    return None

for name, d in dirs.items():
    if not os.path.isdir(d):
        print(f"{name}: MISSING")
        continue
    try:
        sh = shard_for(d, KEY)
        if not sh:
            print(f"{name}: key not found in shards"); continue
        with safe_open(sh, framework="pt", device="cpu") as sf:
            t = sf.get_tensor(KEY)
            s = t[:1, :6].tolist()[0]
            print(f"{name:6s} shard={os.path.basename(sh):32s} shape={tuple(t.shape)} first6={[round(x,6) for x in s]}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
