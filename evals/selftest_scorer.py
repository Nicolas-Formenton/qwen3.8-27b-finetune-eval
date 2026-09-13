"""Harness self-test: fabricate model responses with a KNOWN accuracy and check
that BFCL's scorer reports that number.

'Validate the instrument before spending GPU.' Exercises the registry entry,
prompt-mode decoding (ReturnFormat.PYTHON -> `func(arg=val)`), and the AST
checker (types, nested structures, optional args).

Writing a *correct* gold->response compiler needs care because BFCL's gold
format overloads lists: at parameter level a list means "acceptable
alternatives", but for an array-typed parameter it is the value itself. The
ambiguity is resolved with the function's JSON schema, which is why
`concrete()` takes the schema alongside the value.

Emits two files:
  <cat>_result.json          : perfect      -> expect accuracy 1.0
  <cat>_result_corrupt.json  : first scalar arg corrupted on every Nth case
The corrupt run proves argument-level checking is live (a name-only checker
would still report 1.0).
"""

import json
import os
import pathlib

from bfcl_eval.constants.eval_config import (
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    VERSION_PREFIX,
)

CAT = "simple_python"
GROUP = "non_live"
MODEL = "agent-native-qwen38-base"
CORRUPT_EVERY = 4
OMIT = ""          # gold uses "" to mean "this argument is not passed"

ROOT = pathlib.Path(
    os.environ.get(
        "BFCL_PROJECT_ROOT",
        r"evals/bfcl-project",
    )
)


def load(path: pathlib.Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def concrete(schema: dict, value):
    """Resolve one gold alternative into a concrete value, guided by the schema."""
    stype = (schema or {}).get("type")

    if isinstance(value, dict) or stype in ("dict", "object"):
        props = (schema or {}).get("properties", {})
        out = {}
        for key, sub in value.items():
            picked = pick(props.get(key, {}), sub)
            if picked is not None:
                out[key] = picked
        return out

    if stype in ("array", "list") and isinstance(value, list):
        items = (schema or {}).get("items", {})
        itype = (items or {}).get("type")
        out = []
        for item in value:
            if isinstance(item, list) and itype not in (None, "array", "list"):
                picked = pick(items, item)      # alternative-wrapped element
                if picked is not None:
                    out.append(picked)
            else:
                out.append(concrete(items, item))
        return out

    return value


def pick(schema: dict, alternatives):
    """Choose the first usable alternative (skipping the omit sentinel)."""
    options = alternatives if isinstance(alternatives, list) else [alternatives]
    for candidate in options:
        if candidate is OMIT or candidate is None:
            continue
        return concrete(schema, candidate)
    return None


def render(call: dict, schema_by_name: dict) -> str:
    (name, args_json), = call.items()
    params = (schema_by_name.get(name) or {}).get("parameters", {}).get("properties", {})
    parts = []
    for key, alternatives in args_json.items():
        value = pick(params.get(key, {}), alternatives)
        if value is None:
            continue
        parts.append(f"{key}={value!r}")
    return f"{name}({', '.join(parts)})"


def corrupt(call: dict, schema_by_name: dict) -> tuple[dict, bool]:
    """Flip the first concrete scalar argument to a clearly wrong value."""
    (name, args_json), = call.items()
    params = (schema_by_name.get(name) or {}).get("parameters", {}).get("properties", {})
    new_args, changed = {}, False
    for key, alternatives in args_json.items():
        value = pick(params.get(key, {}), alternatives)
        if value is None:
            continue
        if not changed and isinstance(value, (int, float, str)) and not isinstance(value, bool):
            value = value + 987654 if isinstance(value, (int, float)) else f"{value}_WRONG"
            changed = True
        new_args[key] = value
    return {name: new_args}, changed


def main() -> None:
    test = load(PROMPT_PATH / f"{VERSION_PREFIX}_{CAT}.json")
    gold = {g["id"]: g["ground_truth"] for g in load(POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{CAT}.json")}
    print(f"loaded {len(test)} test cases, {len(gold)} gold answers")

    clean, corrupt_rows, n_corrupted = [], [], 0
    for i, case in enumerate(test):
        schema_by_name = {f["name"]: f for f in case["function"]}
        calls = gold[case["id"]]

        good = "".join(render(c, schema_by_name) for c in calls)
        clean.append({"id": case["id"], "result": good})

        if i % CORRUPT_EVERY == 0:
            texts, any_changed = [], False
            for c in calls:
                bad, changed = corrupt(dict(c), schema_by_name)
                texts.append(render(bad, schema_by_name))
                any_changed = any_changed or changed
            corrupt_rows.append({"id": case["id"], "result": "".join(texts)})
            n_corrupted += int(any_changed)
        else:
            corrupt_rows.append({"id": case["id"], "result": good})

    out_dir = ROOT / "result" / MODEL / GROUP
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        (f"{VERSION_PREFIX}_{CAT}_result.json", clean),
        (f"{VERSION_PREFIX}_{CAT}_result_corrupt.json", corrupt_rows),
    ):
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {name}  ({len(rows)} rows)")

    expected = {
        "category": CAT,
        "model": MODEL,
        "n": len(test),
        "n_corrupted": n_corrupted,
        "expected_accuracy_clean": 1.0,
        "expected_accuracy_corrupt": round((len(test) - n_corrupted) / len(test), 4),
    }
    with open(ROOT / "selftest_expectation.json", "w") as fh:
        json.dump(expected, fh, indent=1)
    print(f"expect clean=1.0000 corrupt={expected['expected_accuracy_corrupt']:.4f} (corrupted {n_corrupted})")


if __name__ == "__main__":
    main()
