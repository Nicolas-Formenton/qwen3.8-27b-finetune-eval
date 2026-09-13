"""Summarize tau2-bench (tau3) simulation results from a simulations/ directory.

Reads every <sim>/results.json, pulls the per-simulation reward, and prints a table grouped
by run name. Also writes aggregate JSON so the numbers survive without re-parsing.
"""

import json
import os
import re
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else None


def reward_of(sim):
    ri = sim.get("reward_info") or {}
    for key in ("reward", "overall_reward"):
        if isinstance(ri.get(key), (int, float)):
            return float(ri[key])
    # some versions nest under reward_breakdown
    rb = ri.get("reward_breakdown") or {}
    if isinstance(rb, dict) and isinstance(rb.get("reward"), (int, float)):
        return float(rb["reward"])
    return None


rows = []
for name in sorted(os.listdir(ROOT)):
    rp = os.path.join(ROOT, name, "results.json")
    if not os.path.isfile(rp):
        continue
    try:
        d = json.load(open(rp, encoding="utf-8"))
    except Exception as e:
        rows.append({"run": name, "error": f"unreadable: {e}"})
        continue
    sims = d.get("simulations") or d.get("sims") or []
    rewards = [r for r in (reward_of(s) for s in sims) if r is not None]
    n_total = len(sims)
    info = d.get("info") or {}
    rows.append({
        "run": name,
        "domain": info.get("environment_info", {}).get("domain_name") or name.split("-")[0],
        "agent": info.get("agent_info", {}).get("llm") or "",
        "trials": info.get("num_trials"),
        "tasks": info.get("num_tasks"),
        "n_sims": n_total,
        "n_scored": len(rewards),
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "perfect": sum(1 for r in rewards if r >= 0.999),
    })

print(f"{'run':44s} {'trials':>6} {'tasks':>5} {'n':>5} {'mean':>7} {'perfect':>8}")
for r in rows:
    if "error" in r:
        print(f"{r['run']:44s}  ERROR {r['error']}")
        continue
    print(f"{r['run']:44s} {str(r['trials']):>6} {str(r['tasks']):>5} "
          f"{r['n_scored']:>5} {str(r['mean_reward']):>7} {r['perfect']:>8}")

if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsalvo em {OUT}")
