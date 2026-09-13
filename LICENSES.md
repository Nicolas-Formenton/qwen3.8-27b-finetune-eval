# License map

The code in this repository is MIT (see `LICENSE`). The **artifacts** carry their own terms,
and those come from the model and datasets that produced them, not from this repo. This file
exists so that nobody has to guess.

## Code

| What | License |
|---|---|
| All scripts in `training/`, `serving/`, `evaluation/`, `ops/` | MIT |

## Base model

| What | License | Note |
|---|---|---|
| Qwen3.8-27B | Apache-2.0 | Permissive. Fine-tunes and derivatives are allowed. |

## LoRA adapters

Every adapter was trained against the Apache-2.0 base, but **the training data decides the
adapter's terms**:

| Adapter | Training data | License | Commercial use |
|---|---|---|---|
| `oh-adapter` | OpenHermes-2.5 | see dataset card | verify before use |
| `xlam-adapter` | `Salesforce/xlam-function-calling-60k` | **CC-BY-NC-4.0** | **NO** |
| `distill-base-adapter` | distill set (teacher Kimi K2.6) | see teacher terms | verify before use |
| `doh-adapter` | distill set, on the OH branch | inherits OH + teacher terms | verify before use |
| `dxlam-adapter` | distill set, **on top of the xLAM branch** | **inherits CC-BY-NC-4.0** | **NO** |

Two consequences worth stating plainly:

1. **`xlam-adapter` and `dxlam-adapter` are non-commercial.** The xLAM dataset is CC-BY-NC, and
   `dxlam` is built on top of the xLAM branch, so it inherits that restriction. Each model card
   on the Hub repeats this.
2. **The distill adapters need the teacher's terms checked.** Distilling from a model does not
   automatically grant rights to the teacher's outputs. Verify the teacher's license before any
   use beyond research.

## Benchmarks

| Benchmark | Upstream | How it is used here |
|---|---|---|
| BFCL v4 | `ShishirPatil/gorilla` (Apache-2.0) | Installed from PyPI as `bfcl-eval`; results reported with its own scoring code. |
| τ³-bench | `sierra-research/tau2-bench` | Cloned and run unmodified except for the harness fixes documented in `docs/FAILURES.md`. |

## Datasets referenced

| Dataset | License | Used for |
|---|---|---|
| `Salesforce/xlam-function-calling-60k` | CC-BY-NC-4.0 | Tool-calling SFT |
| OpenHermes-2.5 | see upstream card | General instruction SFT |
| distill set (teacher Kimi K2.6) | see teacher terms | Distillation runs |

## What is NOT in this repository

The adapters (~1.9 GB total) live on the Hugging Face Hub, not in git. Download size and
provenance are described in `adapters/README.md`.
