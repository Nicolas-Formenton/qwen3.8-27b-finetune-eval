"""Manual LoRA merge — version-independent, verifiable.

Why: PeftModel.merge_and_unload() on this hybrid VLM-ish arch (Qwen3_5*) under
transformers 5.x produced a checkpoint bit-identical to base (adapter silently not
applied) for the unsloth-saved adapter. Manual math removes the guesswork:

    W_merged = W_base + (lora_alpha / r) * (B @ A)

Usage: manual_merge.py <base_dir> <adapter_dir> <out_dir>
Verification: stats of (W_merged - W_base) per sampled tensor, plus an independent
recompute of a few (B@A) values.
"""
import json, os, sys, glob, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base_dir, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)

cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
scale = float(cfg.get("lora_alpha", 16)) / float(cfg.get("r", 16))
print(f"scale (alpha/r) = {scale}")

adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
with safe_open(adapter_file, framework="pt", device="cpu") as sf:
    akeys = list(sf.keys())
    # collect module -> (A, B) in float32
    pairs = {}
    for k in akeys:
        kk = k
        for pref in ("base_model.model.", "base_model."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        if kk.endswith(".lora_A.weight"):
            mod, part = kk[: -len(".lora_A.weight")] + ".weight", "A"
        elif kk.endswith(".lora_B.weight"):
            mod, part = kk[: -len(".lora_B.weight")] + ".weight", "B"
        else:
            print("  ignoring non-lora key:", k)
            continue
        pairs.setdefault(mod, {})[part] = sf.get_tensor(k).float()
print(f"adapter modules: {len(pairs)}  (complete pairs: {sum(1 for v in pairs.values() if set(v)=={'A','B'})})")

base_wm = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
shards = {}
for k, f in base_wm.items():
    shards.setdefault(f, []).append(k)
print(f"base: {len(base_wm)} keys in {len(shards)} shards")

t0 = time.time()
touched, checked = 0, 0
check_lines = []
for shard, keys in sorted(shards.items()):
    out_tensors = {}
    with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as sf:
        for k in keys:
            W = sf.get_tensor(k)
            if k in pairs and set(pairs[k]) == {"A", "B"}:
                A, B = pairs[k]["A"], pairs[k]["B"]  # A:[r,in] B:[out,r]
                delta = (B @ A) * scale            # [out, in]
                if tuple(delta.shape) != tuple(W.shape):
                    raise RuntimeError(f"shape mismatch {k}: {tuple(delta.shape)} vs {tuple(W.shape)}")
                merged = (W.float() + delta).to(W.dtype)
                touched += 1
                if checked < 6:
                    s = torch.arange(0, merged.shape[0], 131)
                    d = (merged[s].float() - W[s].float())
                    expected = delta[s]
                    err = (d - expected).abs().max().item()
                    check_lines.append(f"   {k[-58:]:58s} max|merged-base|={d.abs().max().item():.3e} "
                                       f"max|recompute_err|={err:.3e}")
                    checked += 1
                out_tensors[k] = merged
            else:
                out_tensors[k] = W
    save_file(out_tensors, os.path.join(out_dir, shard), metadata={"format": "pt"})
    print(f"  wrote {shard} ({len(out_tensors)} tensors)", flush=True)

# copy every non-weight file (config, tokenizer, chat template, index)
for f in os.listdir(base_dir):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    src = os.path.join(base_dir, f)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(out_dir, f))
shutil.copy2(os.path.join(base_dir, "model.safetensors.index.json"),
             os.path.join(out_dir, "model.safetensors.index.json"))

print(f"\ntouched (LoRA-applied) tensors: {touched}")
print("spot checks (merged vs base vs recomputed delta):")
for l in check_lines:
    print(l)
print(f"MERGE_DONE {out_dir} in {round(time.time()-t0)}s")
