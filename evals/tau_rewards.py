"""Extract tau2/tau3-bench rewards from data/simulations/<domain>-<model>/results.json.

Handles the shapes the harness actually produces: `simulations` may contain nulls, rewards live
under reward_info.reward (older builds) or a top-level `reward`, and terminate reasons matter for
telling "agent failed" apart from "infra broke".
"""

import glob
import json
import os
import sys

ROOT = os.environ.get("SIM_ROOT", "/root/tools-tau3-bench/data/simulations")


def reward_of(sim):
    if not isinstance(sim, dict):
        return None
    ri = sim.get("reward_info")
    if isinstance(ri, dict) and isinstance(ri.get("reward"), (int, float)):
        return float(ri["reward"])
    for key in ("reward", "score"):
        v = sim.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, pattern))):
        p = os.path.join(d, "results.json")
        if not os.path.exists(p):
            continue
        try:
            r = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(d)}: JSON ilegivel ({e})")
            continue
        sims = r.get("simulations") if isinstance(r, dict) else r
        sims = sims or []
        vals, nulls = [], 0
        for s in sims:
            v = reward_of(s)
            if v is None:
                nulls += 1
            else:
                vals.append(v)
        if not vals:
            print(f"{os.path.basename(d)}: 0 rewards ({len(sims)} sims, {nulls} nulos)")
            continue
        perfect = sum(1 for x in vals if x >= 1.0)
        zero = sum(1 for x in vals if x == 0.0)
        rows.append((os.path.basename(d), len(vals), sum(vals) / len(vals), perfect, zero))
        print(f"{os.path.basename(d)}: n={len(vals)} avg={sum(vals)/len(vals):.3f} "
              f"exatas={perfect}/{len(vals)} zeradas={zero}" + (f" nulos={nulls}" if nulls else ""))
    if rows:
        print("\n=== consolidado ===")
        print(f"{'modelo/dominio':34s} {'n':>3s} {'avg':>6s} {'exatas':>9s}")
        for name, n, avg, perfect, _zero in rows:
            print(f"{name:34s} {n:3d} {avg:6.3f} {perfect:>4d}/{n:<4d}")


if __name__ == "__main__":
    main()
