# agent-native

Fine-tuning, serving, and evaluating a 27B LLM on datacenter GPUs, with the negative results and
the broken harnesses left in.

The interesting part is not that a 27B model runs. It is what the evaluation found.

## The finding

Teaching a model to call tools can destroy its ability to know when **not** to call one.

| Model | Non-Live Overall | Refusal detection |
|---|---|---|
| Qwen3.8-27B (base) | 87.87% | 67.92% |
| + SFT on tool-calling data | **89.69%** | **5.42%** |
| + distillation on top | 88.06% | **85.83%** |

The aggregate score barely moved: 87.87, 89.69, 88.06. The average hid the effect completely.
Looking at a single sub-metric instead, the refusal rate swings **15x**.

The fine-tune that scored best overall was also the one that would answer a question with a
tool call instead of declining, which is the failure mode that matters most in an agent.

Measured with **BFCL v4** (`non_live`, 240 items for the refusal column), fine-tune handler,
200+ items per category. Full table in `results/bfcl/`.

## What was built

| Stage | What |
|---|---|
| **Train** | LoRA r16 fine-tunes on Qwen3.8-27B: tool-calling SFT (xLAM-60k), instruction SFT (OpenHermes-2.5), then distillation from a larger teacher on each branch |
| **Serve** | vLLM on ROCm (1x AMD Instinct MI300X, 192 GB), LoRA modules swapped over the API, 256k context |
| **Evaluate** | BFCL v4 and τ³-bench, plus a TTFT-by-context profile of the attention backend |

Hardware: AMD Developer Cloud, $100 credit, one MI300X at about $2/hour. Every experiment here
fit in that budget, which is why some branches are trained but not benchmarked (see Limitations).

## Serving: 10x on the same GPU

The first working setup decoded at **4.3 tok/s** on a 104k-token conversation. The same session
now runs at **47 tok/s**, and the fix was configuration, not hardware:

| Lever | Effect |
|---|---|
| ROCm attention backend (AITER + `ROCM_AITER_FA`) | 6.8 → 50.3 tok/s at 90k context |
| CUDA graph mode (`FULL_DECODE_ONLY`, not `PIECEWISE`) | 20 → 65 tok/s on short prompts |
| One resident LoRA instead of five | +68% throughput |

Long context is where the attention backend decides everything: without it, the model is fine on
short prompts and unusable past 30k tokens.

### Prefill cost by context length

Single stream, `max_tokens=1`, cold means a unique prompt (no prefix-cache hit).

| Context | Cold | Warm (prefix cache) |
|---|---|---|
| 4k | 6,759 tok/s | 29,923 tok/s |
| 16k | **7,229 tok/s** | 63,408 tok/s |
| 64k | 3,804 tok/s | 95,784 tok/s |
| 256k | 1,244 tok/s | **170,827 tok/s** |

Cold prefill peaks around 16k and falls off after that, which is the attention term showing up.
Warm prefill is limited by the cache, not by compute, and stays fast to 256k.

## What broke

Nine harness bugs, each found by disbelieving a number, each fixed with a reproducible script in
`evaluation/harness_fixes/`. Two examples:

- **The BFCL readiness probe sends no auth header.** Against an authenticated endpoint it gets
  401 forever, never raises `ConnectionError`, and spins. It had burned **341,062 requests**
  before it was noticed.
- **A physically impossible measurement.** A first pass reported 169,189 tok/s of prefill. That
  would require 9.4 PFLOPS on a GPU with a 1.3 PFLOPS peak. The bug was a fixed salt across
  repeats, so the prefix cache was serving the work and the number was counting cached tokens as
  fresh ones.

The full list, with symptoms and fixes, is in [`docs/FAILURES.md`](docs/FAILURES.md). It is the
most useful file in this repository if you plan to run these benchmarks yourself.

## Limitations

Stated plainly, because they change how the results should be read.

1. **The second benchmark does not reach significance.** On τ³-bench the direction repeats
   (base ahead of the tool-calling fine-tune in both domains) but the best comparison lands at
   p = 0.071, which is 93% confidence, not 95%. It is supporting evidence, not proof. The BFCL
   result, with 200+ items per category, is the one worth quoting.
2. **Two branches are trained but not benchmarked.** `distill-base` and `distill-on-OH` exist
   because they are the controls that would separate two explanations for the repair: does
   distillation specifically undo tool-calling damage, or does it just help generally? The budget
   ran out before those ran. They are published so the gap is visible, not hidden.
3. **The refusal column comes from one benchmark.** BFCL measures refusal with 240 items. That is
   enough to see a 15x effect, not enough to characterize it finely.
4. **One seed, one hardware configuration.** No run-to-run variance study was done beyond the
   trial counts reported with each result.

## Repository layout

```
training/      LoRA training and the merge/verify tooling
serving/       vLLM launch script with the flags that matter, plus the toggle benchmark
evaluation/    Benchmarks, the prefill profiler, and the harness fixes
results/       Raw outputs: BFCL CSVs, τ³ simulation JSONs, prefill profile, serving baselines
docs/          FAILURES.md (the bug log), METHODOLOGY.md, RECREATE.md
ops/           Provisioning and teardown for the cloud GPU instance
evals/         Working tree from the development phase, kept for provenance
```

`evaluation/` holds the curated entry points, the ones the commands below use. `evals/` is the
scratch directory where the work actually happened, including the virtualenvs and upstream clones
(both gitignored). It is kept so the path from "first attempt" to "final script" is visible.

The adapters are on the Hugging Face Hub, not in git. See `adapters/README.md`.

## Reproducing

```bash
# 1. Provision a GPU instance (1x MI300X, ROCm image), then point the tooling at it.
#    ops/set_ip.sh rewrites ~/.ssh/config, the serving config, and any stored session URLs.
bash ops/set_ip.sh <instance-ip>
#    Full step-by-step rebuild, including the environments, is in docs/RECREATE.md.

# 2. Serve the base model with LoRA support and the working attention backend
MAXLEN=262144 bash serving/serve_lora.sh

# 3. Fine-tune (Unsloth, LoRA r16)
python training/train_xlam_unsloth.py

# 4. Benchmark. Apply the harness fixes first, or the numbers will be wrong.
python evaluation/harness_fixes/fix_bfcl_readiness.py
python evaluation/harness_fixes/fix_bfcl_thinking.py
bash evaluation/run_bfcl.sh

# 5. Profile prefill cost across context lengths
PREFILL_BASE_URL=http://localhost:8000/v1 python evaluation/measure_prefill.py
```

Two environment notes that cost real time to discover:

- Load `unsloth` **before** `unsloth_zoo`, or `UNSLOTH_IS_PRESENT` is missing and training crashes
  with a meta-tensor error.
- Verify a LoRA merge at the weight level. `PeftModel.merge_and_unload()` silently produced a
  checkpoint bit-identical to the base model for Unsloth-saved adapters. `training/verify_merge.py`
  checks `(W_merged - W_base)` against `scale * (B @ A)`.

## Licenses

Code is MIT. The adapters inherit the terms of their training data, and two of them are
**non-commercial** because the xLAM dataset is CC-BY-NC-4.0. Full map in
[`LICENSES.md`](LICENSES.md).

## Benchmarks used

- **BFCL v4** ([gorilla](https://github.com/ShishirPatil/gorilla)), Berkeley Function Calling
  Leaderboard. Reported with its own scoring code.
- **τ³-bench** ([tau2-bench](https://github.com/sierra-research/tau2-bench)). Run unmodified except
  for the harness fixes noted above.
