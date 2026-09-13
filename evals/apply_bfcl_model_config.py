"""Register the agent-native Qwen3.8-27B variants into BFCL's model registry.

Idempotent: the block is delimited by BEGIN/END markers; re-running replaces it.
Usage:  python apply_bfcl_model_config.py [--show]

Why a script instead of a fork: the droplet reinstalls bfcl-eval from PyPI each
time, so the registry injection has to be reproducible and reviewable.

model_name is the *served model name* that vLLM exposes. For LoRA runs that is
the --lora-modules name, so BFCL's request routes to the adapter. The tokenizer
is loaded from REMOTE_OPENAI_TOKENIZER_PATH (set to the local base dir), because
a LoRA module name is not a resolvable HuggingFace repo id.
"""

import argparse
import importlib.util
import pathlib
import sys

BEGIN = "# ==== agent-native models (BEGIN) ===="
END = "# ==== agent-native models (END) ===="

VARIANTS = [
    ("base", "Qwen3.8-27B base BF16", "qwen38-base"),
    ("oh", "Qwen3.8-27B + OpenHermes-2.5 LoRA r16 (SFT)", "qwen38-oh"),
    ("xlam", "Qwen3.8-27B + xLAM-60k LoRA r16 (SFT)", "qwen38-xlam"),
    ("dbase", "Qwen3.8-27B + distill Kimi K2.6 (on base)", "qwen38-dbase"),
    ("doh", "Qwen3.8-27B + distill Kimi K2.6 (on OH)", "qwen38-doh"),
    ("dxlam", "Qwen3.8-27B + distill Kimi K2.6 (on xLAM)", "qwen38-dxlam"),
]

BLOCK = f'''
{BEGIN}
_pending = {{}}
for _slug, _disp, _served in {VARIANTS!r}:
    for _sfx, _h, _fc, _tag in (
        ("", QwenHandler, False, "Prompt"),
        ("-FC", QwenFCHandler, True, "FC"),
    ):
        _pending["agent-native-qwen38-" + _slug + _sfx] = ModelConfig(
            model_name=_served,
            display_name=_disp + " (" + _tag + ")",
            url="https://github.com/Nicolas-Formenton/agent-native",
            org="agent-native",
            license="apache-2.0 (base) / CC-BY-NC-4.0 (xLAM-derived adapters)",
            model_handler=_h,
            input_price=None,
            output_price=None,
            is_fc_model=_fc,
            underscore_to_dot=False,
        )
local_inference_model_map.update(_pending)
MODEL_CONFIG_MAPPING.update(_pending)
del _pending, _slug, _disp, _served, _sfx, _h, _fc, _tag
{END}
'''


def find_model_config() -> pathlib.Path:
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.origin:
        sys.exit("bfcl_eval is not importable in this interpreter")
    path = pathlib.Path(spec.origin).parent / "constants" / "model_config.py"
    if not path.exists():
        sys.exit(f"model_config.py not found at {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print the file tail and exit")
    args = ap.parse_args()

    path = find_model_config()
    text = path.read_text(encoding="utf-8")

    if args.show:
        print(f"--- {path} (tail) ---")
        print("\n".join(text.splitlines()[-30:]))
        return

    if BEGIN in text:
        head = text.split(BEGIN)[0].rstrip() + "\n"
        action = "replaced"
    else:
        head = text.rstrip() + "\n"
        action = "appended"

    path.write_text(head + BLOCK, encoding="utf-8")
    print(f"{action}: {BEGIN} -> {path}")

    # prove the registry actually loads and contains our keys
    for mod in [m for m in sys.modules if m.startswith("bfcl_eval")]:
        del sys.modules[mod]
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    ours = sorted(k for k in MODEL_CONFIG_MAPPING if k.startswith("agent-native-"))
    print(f"registered {len(ours)} entries:")
    for k in ours:
        cfg = MODEL_CONFIG_MAPPING[k]
        print(f"  {k:38s} served={cfg.model_name:14s} handler={cfg.model_handler.__name__}")


if __name__ == "__main__":
    main()
