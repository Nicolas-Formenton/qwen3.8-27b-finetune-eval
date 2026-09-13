# Methodology

What was measured, how, and what would change the answer. Read this before quoting any number in
the README.

## Serving baseline

**Decode throughput** is measured from the vLLM engine's own reported generation throughput with a
single stream active, sampled from the engine log, alongside a wall-clock measurement of the same
request. The two are reported separately in `results/serving/` because they occasionally disagree
and the engine figure alone can flatter a run.

**Latency at context** is measured on a real conversation, not a synthetic prompt, because the
interesting failures (context-length misdetection, output-budget collapse) only appear with real
history attached.

## Prefill profile

Single stream, `max_tokens=1`, `stream=True`. TTFT isolates prefill; one output token means no
decode contaminates the measurement. Three repeats per level, median reported.

**Cold** means the prompt is unique, with a salt at the position that changes the hash of the
first block. This matters more than it sounds:

- **Pass 1 of this measurement was wrong.** The salt was fixed across repeats, so from the second
  repeat onward the prefix cache served the work. It reported 169,189 tok/s of prefill, which
  would require 9.4 PFLOPS on a GPU with a 1.3 PFLOPS peak. The number was caught by checking it
  against the hardware, not by checking it against expectations.
- Levels built by repeating a filler token also nest: the 4k prompt is a prefix of the 16k prompt,
  so an uncached 16k run silently reuses the 4k run's work. A unique salt per (level, repeat) is
  what makes cold mean cold.

**Warm** re-sends an identical prompt and reports what the prefix cache delivers. Warm numbers
describe a cache, not compute, and should not be compared against vendor prefill claims.

Every reported level includes the token count the server actually counted (`usage.prompt_tokens`),
not the target. The targets are what was asked for; the counts are what happened.

## BFCL v4

Run through `bfcl-eval` against our own endpoint with `REMOTE_OPENAI_BASE_URL`, handler
`QwenFCHandler` (`is_fc_model=True`), temperature 0.001, 16 threads, seed 300.

**Reasoning must be left on.** Measured on the 240-item refusal category: the same base model
scores 69.17% with reasoning and 17.08% without. With reasoning suppressed the model answers the
question instead of declining to call a tool, which is precisely what that category measures. Cost
is 3.4x wall time (2m15s against 44s). This is a property of the benchmark, not a preference.

**Two harness fixes are required** before any number is trustworthy, both in
`evaluation/harness_fixes/`. Without the readiness fix the runner spins on 401s. Without the
thinking fix the model spends its entire 4096-token completion budget on reasoning.

Only the `non_live` collection was run for all three models. The base model additionally ran
`multi_turn`.

## τ³-bench

Simulated user is the frozen base checkpoint for every variant, so the simulator does not move
with the agent. 20 tasks per domain, seed 300, `--max-steps 60`.

**Judge independence.** τ³ scores some task types with an LLM judge. The judge must not be the
model under test. A distilled adapter outside the test set serves as the judge
(`evaluation/harness_fixes/fix_tau3_judge.py` points the four hardcoded judge constants at it).
The judge name needs the `openai/` provider prefix; without it LiteLLM fails every call with
"LLM Provider NOT provided", which appears as task failures rather than as a config error.

**The judge is used per domain, and this is measurable.** Counting tasks that mention
`nl_assertions` gives a false alarm: 100% of airline tasks declare them. The field that decides is
`evaluation_criteria.reward_basis`:

| Domain | reward_basis | Judge invoked |
|---|---|---|
| airline | `[COMMUNICATE, DB]` | no |
| retail | `[NL_ASSERTION]` for 112 of 114 tasks | yes |

**Results do not reach significance.** The best comparison (base n=54 against the tool-calling
fine-tune n=51, airline) gives z=1.81, p=0.071. Two-proportion z-test, no correction for multiple
comparisons. Direction repeats across both domains and across 3- and 4-trial runs, but the
confidence intervals overlap. Treat it as supporting evidence.

To settle it would need roughly 80 to 100 scored runs per model, paired on identical task counts.
The runs here have unequal `n` because per-run timeouts truncated some blocks, so the pairs are
not matched.

## What would change the conclusions

1. **A paired τ³ run at n >= 80 per model.** Would turn the supporting evidence into a result, or
   kill it.
2. **The two unmeasured branches.** Distillation on the base and on the instruction-SFT branch
   separate "the repair is specific to tool-calling damage" from "distillation helps generally".
   Until those run, the mechanism behind the headline finding is a hypothesis.
3. **A quantized serving comparison.** Every number here is BF16. Quantized prefill claims are not
   comparable to these without measuring the accuracy cost alongside the speed.

## Reporting rules used throughout

- Report the median of repeats, not the best run.
- Report `n` next to every rate, and refuse to compare rates with materially different `n`.
- Report confidence intervals, and state the null result when there is one.
- Check physical plausibility before publishing: throughput against hardware peak, token counts
  against prompt size, merged weights against the delta that should have been applied.
- Name the harness bugs found, in `docs/FAILURES.md`, rather than quietly working around them.
