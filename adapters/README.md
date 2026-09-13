# Adapters

The five LoRA adapters are **not in this repository**. They total about 1.9 GB, which is past the
point where git is the right tool. They live on the Hugging Face Hub.

| Adapter | Branch | Trained on | Benchmarked | License |
|---|---|---|---|---|
| `xlam-adapter` | tool-calling SFT | xLAM-60k | BFCL + τ³ | **CC-BY-NC-4.0** |
| `dxlam-adapter` | distillation on the xLAM branch | teacher outputs | BFCL + τ³ | **CC-BY-NC-4.0** (inherited) |
| `oh-adapter` | instruction SFT | OpenHermes-2.5 | not yet | see dataset card |
| `distill-base-adapter` | distillation on the base | teacher outputs | not yet | see teacher terms |
| `doh-adapter` | distillation on the OH branch | teacher outputs | not yet | see dataset card + teacher |

Links to the Hub repositories are added here once uploaded.

## Why the two unmeasured adapters are published

`distill-base` and `distill-on-OH` are the **controls** for the main finding. The headline result
is that distillation repaired a refusal rate the tool-calling fine-tune had destroyed. Two
explanations fit that:

1. Distillation specifically undoes the damage the tool-calling SFT caused.
2. Distillation improves the model generally, and the refusal column just moved with it.

Separating those requires running distillation on a branch that was never tool-calling-tuned. The
budget ran out before those runs happened, so the adapters are published unmeasured and the gap is
named in the README rather than hidden.

## Details common to all five

- **Base model:** Qwen3.8-27B (Apache-2.0)
- **Rank:** LoRA r16
- **Served as:** vLLM LoRA modules over the API, one resident at a time
- **Files per adapter:** `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`,
  `chat_template.jinja`

## Merge warning

If you merge one of these into the base yourself, **verify the result at the weight level**.
`PeftModel.merge_and_unload()` silently returned a checkpoint bit-identical to the base model for
adapters saved by Unsloth, which produces a merged model that is simply the base model wearing a
different name. `training/verify_merge.py` checks that `(W_merged - W_base)` matches
`scale * (B @ A)`; `training/manual_merge.py` is the merge path that was verified to work.
