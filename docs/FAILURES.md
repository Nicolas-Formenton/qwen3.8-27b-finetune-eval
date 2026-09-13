# FAILURES.md — what broke, why, and how to detect it

Honest engineering log. Each entry: symptom → root cause → detection → fix.
These are the artifacts worth reading in a portfolio; the numbers are downstream.

---

## 1. Silent no-op LoRA merge (the expensive one)

**Symptom.** Six models scored statistically identically (83.5-84.0% on 200 held-out
tool-call questions). Suspicious: the per-question correctness vectors for `base` and
`xlam` were **bit-identical** (0 differences in 200).

**Root cause.** `PeftModel.merge_and_unload()` ran without error but produced a
checkpoint **bit-identical to base** (`max|merged - base| == 0.000e+00` on sampled
tensors). The adapter was saved by Unsloth with a regex `target_modules` and
`auto_mapping`, and under this transformers version the merge silently matched nothing.
Every downstream artifact that depended on it (the distill run stacked on top of the
"xLAM" merge) was therefore trained on the plain base — a duplicate of D-base.

**Detection.** Compare the merged checkpoint against the adapter math:

```
actual  = W_merged - W_base        (sampled rows)
expect  = (alpha/r) * (B @ A)      (same rows)
```

Agreement ⇒ the adapter was applied. `actual == 0` while the adapter is non-empty ⇒
no-op merge. A name-only comparison is not enough: module names can map perfectly
(992/992 keys) and the merge still be a no-op.

**Fix.** Manual merge (`W + scale * B @ A`), version-independent, with the
recomputation check printed for every run. 496/496 modules applied.

**Lesson.** Never trust a merge's exit code. Verify numerically, every time.

---

## 2. `Cannot copy out of meta tensor` in Unsloth

**Symptom.** `NotImplementedError: Cannot copy out of meta tensor; no data!` while
loading Qwen3.8-27B in Unsloth.

**Root cause (two, stacked).**
1. `unsloth_zoo.tokenizer_utils.fix_untrained_tokens()` walks the embedding matrix on a
   model whose embeddings start on the meta device (VLM-style config) and crashes.
2. `import unsloth_zoo` **before** `import unsloth` raises
   `ImportError: Please install Unsloth via pip install unsloth!` — `unsloth` is what
   sets the `UNSLOTH_IS_PRESENT` environment flag that `unsloth_zoo.__init__` requires.

**Fix.** Import `unsloth` first; then no-op the walker:

```python
from unsloth import FastModel                       # sets UNSLOTH_IS_PRESENT
import unsloth_zoo.tokenizer_utils
unsloth_zoo.tokenizer_utils.fix_untrained_tokens = lambda *a, **k: (None, None)
```

Also: use a plain `AutoTokenizer` for data rendering. FastModel's tokenizer is a VLM
processor whose `apply_chat_template` tries to decode plain JSON as base64 images
(`Invalid base64-encoded string ... cannot be 1 more than a multiple of 4`).

**Result.** Unsloth went from unusable (PEFT fallback at ~26 s/step) to **5.4 s/step**
(~4.7×), which is what made the rest of the runs affordable.

---

## 3. vLLM hangs after CUDA-graph capture on ROCm — and the workaround cost 4-5x

**Symptom.** Server loads the model, captures PIECEWISE CUDA graphs (51/51, 100%),
then never serves: `/v1/models` returns 000 forever, no error in the log.

**First fix (wrong).** `--enforce-eager`. The engine comes up, so it looks solved — but it
disables ALL graphs and cost **4.5-5.5x** throughput (base 14.8 tok/s, D-xLAM 9.5 tok/s).

**Real fix.** PIECEWISE is only the *default* mode, not the only one. `CUDAGraphMode` also has
`FULL`, `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE`. `FULL_DECODE_ONLY` captures full graphs for
**decode only** (what tok/s measures) and `NONE` for prefill, so it never enters the code path
that hangs:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Measured on the same stack, same model, same prompt:

| Graph mode | base tok/s | D-xLAM tok/s |
|---|---|---|
| `--enforce-eager` | 14.8 | 9.5 |
| **`FULL_DECODE_ONLY`** | **67.0** | **52.6** |

Capture completes in 24s (102 graphs, 0.53 GiB) and the API serves normally. Also note the
model loads 3x faster (load 21s vs 60s) because capture no longer stalls.

**Detection note.** "Server started" ≠ "server serving". Poll `/v1/models` for HTTP 200
with a timeout; a load-only check would have passed while the harness stalled.

**Lesson.** When a workaround costs 5x, it is worth re-reading the enum: the failing value was
one option among five, and the fix was a different option, not the absence of the feature.

---

## 4. `pkill -f <pattern>` kills your own SSH session

**Symptom.** `ssh host 'pkill -f eval_all.sh ...'` returns **exit 255** and does nothing,
because the remote shell's own command line contains the pattern.

**Fix.** Never `pkill -f` with a pattern that appears in the command you are running.
Either `pkill -x <exact-name>`, or resolve PIDs first and kill by PID from a launcher
script that is not itself named after the pattern.

---

## 5. BFCL's gold format overloads lists (learned during harness self-test)

**Symptom.** A fabricated "perfect" response set scored 86.25% instead of 100%.

**Root cause.** Three distinct semantics in `possible_answer`:
1. `[""]` (or `""` among alternatives) means **the argument is not passed** — passing
   `arg=''` explicitly is a *type error*, not a valid answer.
2. At parameter level a list is **acceptable alternatives**; for an array-typed
   parameter it is the **value itself**. Disambiguate with the function's JSON schema.
3. Nested structures carry their own alternative lists and need recursive unwrapping
   (`"conditions": [{"field": ["age"]}]` → `conditions={'field': 'age'}`).

**Detection.** The self-test: fabricate responses with a *known* accuracy and assert
the scorer reports that number. Two files — perfect (expect 1.0) and
first-scalar-argument-corrupted on every 4th case (expect 0.75). A name-only checker
would report 1.0 for both, so the corrupt file also proves argument-level checking
is live.

**Result.** `validate_harness.py` → predicted 1.0000/0.7500, reported 1.0000/0.7500.

---

## 6. Our own 200-question tool-call metric saturated

**Symptom.** All six variants scored 83.5-84.0% with identical error profiles
(23/200 unparseable, 9-10/200 wrong function).

**Root cause.** The metric only compared the **name of the first function call**.
No arguments, no parallel calls, no abstention, and thinking was left enabled so the
JSON frequently never appeared. Ceiling reached; a fine-tune cannot show up.

**Fix.** Move to BFCL v4 (arguments via AST + real execution, parallel calls,
multi-turn, irrelevance/abstention) and keep the custom suite only as a secondary
dialect-specific read, with thinking disabled and arguments compared.

**Lesson.** Before concluding "the fine-tune didn't change behaviour", check whether
the metric can detect it at all. A saturated metric and a null result look identical.

---

## 7. BFCL prompt mode: the model copies the placeholder name from the instruction

**Symptom.** First BFCL v4 run (`parallel`, 200 entries, xLAM adapter) scored **1.50%**.
The per-item errors show the model emitting calls that carry the *instruction's*
placeholder instead of the real function name:

```
model output : [func_name1(params={"b_field": 5, ...}), func_name1(params={...})]
expected     : [calculate_em_force(b_field=5, ...), calculate_em_force(...)]
```

Other items show the right values but a mangled name (`func_spotify_play` where the
documented function is `spotify.play`), and a few fail AST decoding because the model
emits our trained JSON dialect (`[{"calculate_bmi": {...}}]`) instead of Python call
syntax.

**Root cause.** BFCL's own system prompt specifies the output format using a
placeholder: *"you MUST put it in the format of `[func_name1(params_name1=..., ...),
func_name2(params)]`"*. A model that does not recognise the placeholder as
placeholders literally emits `func_name1`. Two contributing factors on our side:

* Our adapters were fine-tuned on JSON-in-content (`[{"name": ..., "arguments": ...}]`),
  which is a different dialect from BFCL prompt mode's Python-call syntax.
* The registry entries ship `underscore_to_dot=False`; models that write
  `spotify_play` for the dotted name need `underscore_to_dot=True` so the checker
  converts the expected name before comparing.

**Detection.** Dump the exact prompt BFCL builds (`dump_bfcl_prompt.py`) and grep the
model output for the literal token `func_name1`. If the instruction's placeholder string
appears in model output, the model is copying it — not a decoding bug in the harness.

**Fix / reading.** Report BFCL in **both** modes (prompt and `-FC`) because they measure
different things, and never compare a prompt-mode number against an FC-mode number. The
prompt-mode score is a genuine measurement of "can you follow this benchmark's dialect",
which is informative precisely because our SFT taught a different one.

**Lesson.** A near-zero score on a new harness is usually a *dialect* mismatch, not a
capability result. Check the prompt the harness actually sends before believing the
number, and get the base model's score on the same category as the control.


## 8. The agent chat "stops for no reason" mid-answer after the serving host changes IP

**Symptom.** The chat answers, then a later turn ends abruptly. The stored assistant message
is cut **mid-sentence** (`... - MG: Cleitinho (`) while recorded as `finish_reason=stop`, and
the turn registers as `status=complete`, so the UI shows a normal end. Looks like the model
gave up or "the app froze".

**Root cause.** Two independent facts combine:

1. The client stores `base_url` **per session** (`state.db` -> `sessions.model_config` JSON and
   `billing_base_url`), separate from the profile config. Recreating the serving host leaves
   those rows pointing at a dead IP, so the context-length probe fails and the client silently
   falls back to a hardcoded catalog window:
   ```
   Could not detect context length for model 'qwen38-dxlam' at http://129.212.188.58:8000/v1
   Using hardcoded context length 131,072 for model 'qwen38-dxlam' (catalog match on 'qwen')
   ```
2. The conversation then grows past that **believed** window (observed prompt: 148,286 tokens
   vs believed 131,072). With prompt > believed limit the output allowance collapses and the
   response is truncated at a few hundred tokens.

**Detection.** `finish_reason=stop` + content ending mid-word in the messages table, next to a
`catalog match` line in the client log. Both are cheap to check; neither is visible in the chat UI.

**Fix (two layers).**
* One-shot: repoint the session rows. `UPDATE sessions SET model_config=…, billing_base_url=…`
  rewriting any `http://<ipv4>:8000/v1` to the live URL (9 sessions were stale in one pass).
* Durable: do it inside the IP-change script (`ops/set_ip.sh` now has this as step 4), so
  recreating a host can never silently reintroduce the bug.
* Also declare `context_length` **per provider** (`providers.<name>.models.<model>.context_length`):
  a global `model.context_length` is discarded when the active provider differs from the
  configured default.

**Lesson.** A truncated answer with a clean `finish_reason` is a *client configuration* signal,
not a model-quality signal. When integrating a hosted model, treat every per-session cache of
the endpoint as a thing that goes stale.

## 9. tau2/tau3: o JUIZ das NL assertions esta hardcoded num modelo externo

**Sintoma.** Retries em serie e amostras minusculas. No log:
`Retry 3/3 for task N: litellm.NotFoundError ... The model \`gpt-4.1-2025-04-14\` does not exist`.
Um dominio de 20 tarefas fechava com 2 a 16 tarefas e o processo parecia ocioso (0,5% CPU) entre
tentativas.

**Causa.** `src/tau2/config.py` tem QUATRO modelos internos hardcoded, todos externos:

```
DEFAULT_LLM_USER                = "gpt-4.1-2025-04-14"
DEFAULT_LLM_NL_ASSERTIONS       = "gpt-4.1-2025-04-14"   <- o juiz
DEFAULT_LLM_ENV_INTERFACE       = "gpt-4.1-2025-04-14"
DEFAULT_LLM_EVAL_USER_SIMULATOR = "claude-opus-4-5"
```

`--user-llm` sobrescreve o simulador, mas nao existe flag para os outros. O
`evaluator_nl_assertions.py:122` usa `DEFAULT_LLM_NL_ASSERTIONS` direto e NAO ha try/except em
volta da chamada: a falha do juiz derruba a tarefa inteira.

**Diagnostico correto (a armadilha).** Olhar so a presenca de `nl_assertions` no arquivo da
tarefa da falso alarme: aparece em **100%** das tarefas do airline. O que decide e o
`evaluation_criteria.reward_basis`:

| Dominio | reward_basis | Juiz e chamado? |
|---|---|---|
| airline | 50x `[COMMUNICATE, DB]` | **nao** -> resultados validos |
| retail | 112 de 114 com `NL_ASSERTION` | **sim** -> resultados comprometidos |

**Correcao.** `evals/fix_tau3_judge.py` troca as quatro constantes por leitura de ambiente,
com juiz padrao no nosso endpoint. Idempotente e verifica 4/4 no fim.

**Regra de independencia.** O juiz NAO pode ser o modelo sob teste — senao e auto-avaliacao.
Para a matriz completa use um adapter FORA do conjunto testado, via `TAU3_JUDGE_LLM`.

**Licao.** Antes de acreditar num score de benchmark com juiz LLM, confirme QUAL modelo faz o
julgamento e se ele esta alcancavel. Um juiz ausente nao aparece como erro no resultado: aparece
como tarefas que "falharam", indistinguivel de incapacidade do modelo.
