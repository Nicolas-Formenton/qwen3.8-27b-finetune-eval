"""Upload the five LoRA adapters to the Hugging Face Hub, each with its own model card.

Run:  uv run --python 3.11 --with huggingface_hub python <this file>

Reads the write token from ~/.hf_write_token (never hardcoded, never committed).
Idempotent: re-running overwrites files and re-creates repos if missing.
"""

import pathlib
import sys

from huggingface_hub import HfApi

USER = "nickzin"
BASE_MODEL = "Qwen/Qwen3.8-27B"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]  # <repo>/ (file lives in <repo>/evaluation/)

# adapter dir -> (hub repo name, card metadata)
ADAPTERS = {
    "xlam-adapter": {
        "repo": "qwen3.8-27b-lora-xlam",
        "title": "Qwen3.8-27B + xLAM-60k LoRA (tool-calling SFT)",
        "branch": "Tool-calling SFT",
        "data": "`Salesforce/xlam-function-calling-60k`, 6000 rows",
        "license": "cc-by-nc-4.0",
        "benchmarked": True,
    },
    "dxlam-adapter": {
        "repo": "qwen3.8-27b-lora-dxlam",
        "title": "Qwen3.8-27B + xLAM SFT + distillation (the repaired branch)",
        "branch": "Distillation on top of the xLAM branch",
        "data": "teacher-generated distillation set",
        "license": "cc-by-nc-4.0",
        "benchmarked": True,
    },
    "oh-adapter": {
        "repo": "qwen3.8-27b-lora-oh",
        "title": "Qwen3.8-27B + OpenHermes-2.5 LoRA (instruction SFT)",
        "branch": "General instruction SFT",
        "data": "OpenHermes-2.5, 3000 rows",
        "license": "other",
        "benchmarked": False,
    },
    "distill-base-adapter": {
        "repo": "qwen3.8-27b-lora-distill-base",
        "title": "Qwen3.8-27B + distillation (control: no prior SFT)",
        "branch": "Distillation directly on the base model",
        "data": "teacher-generated distillation set",
        "license": "other",
        "benchmarked": False,
    },
    "doh-adapter": {
        "repo": "qwen3.8-27b-lora-doh",
        "title": "Qwen3.8-27B + OpenHermes SFT + distillation (control)",
        "branch": "Distillation on top of the instruction-SFT branch",
        "data": "teacher-generated distillation set",
        "license": "other",
        "benchmarked": False,
    },
}

CARD = """---
base_model: {base_model}
library_name: peft
license: {license}
tags:
- lora
- peft
- qwen
- tool-calling
- function-calling
---

# {title}

LoRA adapter (rank 16) trained on **{base_model}** as part of
[qwen3.8-27b-finetune-eval](https://github.com/Nicolas-Formenton/qwen3.8-27b-finetune-eval).

**Branch:** {branch}
**Training data:** {data}
**License:** {license_note}

{results}

## Serving

Served as a vLLM LoRA module, one resident at a time:

```bash
vllm serve Qwen/Qwen3.8-27B \\
  --served-model-name qwen38-base \\
  --enable-lora --max-lora-rank 16 --max-loras 1 \\
  --lora-modules {module_name}={local_path} \\
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The served name is what clients address. A LoRA module name is not a Hugging Face repo id, so the
client must be configured with the served name rather than the adapter path.

## Merge warning

Verify a merge at the **weight level** before trusting it. `PeftModel.merge_and_unload()` silently
produced a checkpoint bit-identical to the base model for these Unsloth-saved adapters, which
looks like a successful merge and behaves like no fine-tune at all. The check is
`(W_merged - W_base) == scale * (B @ A)` within a few bf16 ulps.
"""

RESULTS_MEASURED = """## Results (BFCL v4, non-live)

| Metric | Base | This adapter |
|---|---|---|
| Non-Live Overall | 87.87% | **{overall}** |
| Refusal detection | 67.92% | **{refusal}** |

Measured with `bfcl-eval`, fine-tune handler, reasoning enabled, 200+ items per category.
Numbers reproduce the table in the project README.
"""

RESULTS_NOT_MEASURED = """## Results

**Not benchmarked.** This adapter was trained as a control and the GPU budget ran out before it
was evaluated. It is published so the gap is visible rather than hidden.

Why it matters: the headline finding is that distillation repaired a refusal rate the tool-calling
fine-tune had destroyed. Separating "the repair is specific to tool-calling damage" from
"distillation helps generally" requires a distillation run on a branch that was never
tool-calling-tuned. That is what this adapter is for.
"""


def main() -> None:
    token = (pathlib.Path.home() / ".hf_write_token").read_text().strip()
    api = HfApi(token=token)
    print(f"authenticated as {api.whoami()['name']}")

    for local_name, meta in ADAPTERS.items():
        src = REPO_ROOT / "artifacts" / local_name
        if not src.is_dir():
            print(f"  SKIP {local_name}: not found")
            continue

        repo_id = f"{USER}/{meta['repo']}"
        print(f"\n=== {repo_id} ===")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        if meta["benchmarked"]:
            overall, refusal = ("89.69%", "5.42%") if "xlam-adapter" == local_name else ("88.06%", "85.83%")
            results = RESULTS_MEASURED.format(overall=overall, refusal=refusal)
        else:
            results = RESULTS_NOT_MEASURED

        if meta["license"] == "cc-by-nc-4.0":
            lic_note = ("**CC-BY-NC-4.0, non-commercial.** Inherited from the xLAM dataset"
                        if local_name != "dxlam-adapter"
                        else "**CC-BY-NC-4.0, non-commercial.** Inherited through the xLAM branch")
        else:
            lic_note = "See the dataset and teacher model cards; verify before commercial use"

        card = CARD.format(
            base_model=BASE_MODEL,
            title=meta["title"],
            branch=meta["branch"],
            data=meta["data"],
            license=meta["license"],
            license_note=lic_note,
            results=results,
            module_name=f"qwen38-{local_name.split('-')[0]}",
            local_path=f"/root/adapters/{local_name}",
        )

        # our card replaces whatever default the trainer wrote
        files = [f for f in src.iterdir() if f.is_file() and f.name != "README.md"]
        for f in files:
            size_mb = f.stat().st_size / 1e6
            print(f"  upload {f.name} ({size_mb:.1f} MB)")
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id)

        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print("  card uploaded")

    print("\nDONE")


if __name__ == "__main__":
    sys.exit(main())
