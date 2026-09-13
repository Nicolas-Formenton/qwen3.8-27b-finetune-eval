import json, os, glob
from safetensors import safe_open

def keys_of(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return list(json.load(open(idx))["weight_map"].keys())
    f = sorted(glob.glob(os.path.join(d, "*.safetensors")))[0]
    with safe_open(f, framework="pt", device="cpu") as sf:
        return list(sf.keys())

for d in ["/root/models/qwen3.8-27b", "/root/models/merged-xlam"]:
    ks = keys_of(d)
    print("===", d, "|", len(ks), "keys")
    print("sample:", ks[:4])
    attn = [k for k in ks if "layers.20." in k][:6]
    print("layer20:", attn)
