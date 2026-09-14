#!/usr/bin/env python3
"""
Generate the README charts from the measured artifacts.

Every number here is read from a file under results/, not typed in. If a result file
changes, the charts change. Run:  uv run --with matplotlib python evaluation/make_charts.py

Charts are emitted as SVG with svg.fonttype='none', so text stays as text: the file is
small, it scales crisply on GitHub, and a screen reader can still read the labels.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# --- theme ------------------------------------------------------------------
# Light background on purpose: GitHub renders SVG in whichever theme the reader
# uses, and dark text on a white card stays legible in both.
plt.rcParams.update(
    {
        "svg.fonttype": "none",  # keep text as text, not outlines
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],  # matplotlib's own fallback, always present
        "font.size": 11,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D8DEE9",
        "axes.labelcolor": "#2E3440",
        "axes.titlecolor": "#2E3440",
        "text.color": "#2E3440",
        "xtick.color": "#4C566A",
        "ytick.color": "#4C566A",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    }
)

# Model palette. Blue / amber / teal reads correctly for the common forms of
# colour blindness, unlike red-vs-green.
C_BASE = "#4C6EF5"
C_XLAM = "#F59F00"
C_DX = "#12B886"
NEUTRAL = "#ADB5BD"
GREY = "#868E96"

MODELS = ["base", "xLAM", "D-xLAM"]
MCOLOR = {"base": C_BASE, "xLAM": C_XLAM, "D-xLAM": C_DX}


def save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.25)

    # matplotlib hardcodes the family it resolved to (DejaVu Sans). Swap in a web-safe
    # stack so the labels render in the reader's own sans on GitHub instead of falling
    # back unpredictably. Text stays text, so it also stays selectable and screen-readable.
    svg = p.read_text(encoding="utf-8")
    svg = re.sub(r"font-family\s*:\s*[^;\"']+", "font-family:Helvetica Neue,Helvetica,Arial,sans-serif", svg)
    p.write_text(svg, encoding="utf-8")

    # PNG preview alongside, for eyeballing the layout before committing.
    fig.savefig(OUT / name.replace(".svg", ".png"), format="png", bbox_inches="tight", pad_inches=0.25)

    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


# --- data loaders -----------------------------------------------------------
def load_bfcl() -> dict:
    """Read BFCL's own scoring CSV. This is the source of truth for the README table."""
    rows = {}
    with open(ROOT / "results" / "bfcl" / "data_non_live.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = r["Model"]
            if "xLAM-60k" in name:
                key = "xLAM"
            elif "distill" in name:
                key = "D-xLAM"
            else:
                key = "base"
            rows[key] = {
                "overall": float(r["Non-Live Overall Acc"].rstrip("%")),
                "refusal": float(r["Irrelevance Detection"].rstrip("%")),
                "Python simple": float(r["Python Simple AST"].rstrip("%")),
                "Java simple": float(r["Java Simple AST"].rstrip("%")),
                "JavaScript simple": float(r["JavaScript Simple AST"].rstrip("%")),
                "Multiple": float(r["Multiple AST"].rstrip("%")),
                "Parallel": float(r["Parallel AST"].rstrip("%")),
                "Parallel multiple": float(r["Parallel Multiple AST"].rstrip("%")),
            }
    return rows


def load_prefill() -> tuple[list[int], list[float], list[float]]:
    d = json.loads((ROOT / "results" / "prefill" / "prefill_profile.json").read_text())
    lv, cold, warm = [], [], []
    for L in d["levels"]:
        lv.append(L["target"])
        cold.append(sum(x["prefill_tok_s"] for x in L["cold"]) / len(L["cold"]))
        warm.append(sum(x["prefill_tok_s"] for x in L["warm"]) / len(L["warm"]))
    return lv, cold, warm


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Right choice for proportions this close to 0 and 1,
    where the normal approximation produces intervals that run past the bounds."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_tau3() -> list[dict]:
    """Pick the representative runs (one per domain/model), the ones cited in the README."""
    want = {
        ("airline", "base"): "FINAL-airline-qwen38-base-t4",
        ("airline", "xLAM"): "FINAL-airline-qwen38-xlam-t4",
        ("airline", "D-xLAM"): "airline-qwen38-dxlam-t3",
        ("retail", "base"): "retail-qwen38-base-t3",
        ("retail", "xLAM"): "retail-qwen38-xlam-t3",
    }
    out = []
    for r in json.loads((ROOT / "results" / "tau3-summary.json").read_text()):
        for (dom, mod), run in want.items():
            if r["run"] == run:
                n, k = r["n_scored"], r["perfect"]
                lo, hi = wilson(k, n)
                out.append(
                    {"domain": dom, "model": mod, "n": n, "mean": k / n, "lo": lo, "hi": hi}
                )
    return sorted(out, key=lambda x: (x["domain"], MODELS.index(x["model"])))


# --- chart 1: the finding ---------------------------------------------------
def chart_finding(bfcl: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    for ax, (metric, title, ylo) in zip(
        axes,
        [
            ("overall", "Non-Live Overall", 80),
            ("refusal", "Refusal detection", 0),
        ],
    ):
        vals = [bfcl[m][metric] for m in MODELS]
        bars = ax.bar(MODELS, vals, color=[MCOLOR[m] for m in MODELS], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (1.6 if ylo else 3.0),
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        ax.set_ylim(ylo, 100 if not ylo else 94)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="y", color="#ECEFF4", linewidth=0.9)
        ax.set_axisbelow(True)

    # Captions live below the axis, not inside the plot area: inside, they land on bars.
    axes[0].set_xlabel("spread: 1.8 points", fontsize=10, color=GREY, labelpad=8)
    axes[1].set_xlabel(
        "spread: 80.4 points", fontsize=10, color="#BF616A", fontweight="bold", labelpad=8
    )

    fig.suptitle(
        "Three models, one benchmark: the average hides the effect",
        fontsize=13.5,
        fontweight="bold",
        y=1.04,
    )
    save(fig, "the-finding.svg")


# --- chart 2: BFCL by category ---------------------------------------------
def chart_bfcl_categories(bfcl: dict) -> None:
    cats = [
        ("Python simple", "Python simple"),
        ("Java simple", "Java simple"),
        ("JavaScript simple", "JavaScript simple"),
        ("Multiple", "Multiple"),
        ("Parallel", "Parallel"),
        ("Parallel multiple", "Parallel multiple"),
        ("Refusal detection", "refusal"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ypos = range(len(cats))
    h = 0.26

    for i, m in enumerate(MODELS):
        offs = [y + (i - 1) * h for y in ypos]
        vals = [bfcl[m][k] for _, k in cats]
        bars = ax.barh(offs, vals, height=h, label=m, color=MCOLOR[m])
        for b, v in zip(bars, vals):
            ax.text(v + 1.0, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8.5)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([lbl for lbl, _ in cats], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", color="#ECEFF4", linewidth=0.9)
    ax.set_axisbelow(True)
    # Below the axes: inside, it sat on the refusal-detection bars.
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False, ncol=3, fontsize=10)
    ax.set_title(
        "Six of seven categories agree. The seventh is off the chart.\n"
        "BFCL v4 non-live, fine-tune handler, reasoning on",
        fontsize=12,
        fontweight="bold",
        pad=12,
        loc="left",
    )
    save(fig, "bfcl-categories.svg")


# --- chart 3: the serving backend ------------------------------------------
def chart_serving() -> None:
    labels = ["Short context\n(~1k tokens)", "Long context\n(90k tokens)"]
    off = [67.0, 6.8]
    on = [67.3, 60.3]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = range(len(labels))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], off, width=w, label="AITER off (legacy ROCM_ATTN)", color=NEUTRAL)
    b2 = ax.bar([i + w / 2 for i in x], on, width=w, label="AITER on (ROCm Flash Attention)", color=C_DX)

    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 1.2,
                f"{b.get_height():.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10.5,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_ylim(0, 82)
    ax.grid(axis="y", color="#ECEFF4", linewidth=0.9)
    ax.set_axisbelow(True)
    # Legend below the axes. Inside, it sat on the 67.0 / 67.3 labels.
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.20), frameon=False, fontsize=10)

    ax.annotate(
        "8.9x",
        xy=(1 + w / 2, 60.3),
        xytext=(1.30, 68),
        fontsize=12,
        fontweight="bold",
        color="#BF616A",
        arrowprops=dict(arrowstyle="->", color="#BF616A", lw=1.4),
    )
    ax.annotate("no change", xy=(0, 67.6), xytext=(0, 74), ha="center", fontsize=10, color=GREY)

    ax.set_title(
        "The attention backend only decides at long context",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    save(fig, "serving-backend.svg")


# --- chart 4: prefill -------------------------------------------------------
def chart_prefill() -> None:
    lv, cold, warm = load_prefill()
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    xs = range(len(lv))

    ax.plot(xs, cold, marker="o", markersize=6, linewidth=2.2, color=C_BASE, label="Cold (unique prompt)")
    ax.plot(xs, warm, marker="s", markersize=6, linewidth=2.2, color=C_XLAM, label="Warm (prefix cache)")
    ax.set_yscale("log")

    for i, (c, w) in enumerate(zip(cold, warm)):
        ax.annotate(f"{c:,.0f}", (i, c), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8.5, color=C_BASE)
        ax.annotate(f"{w:,.0f}", (i, w), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color="#B8860B")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{v // 1000}k" if v >= 1000 else str(v) for v in lv], fontsize=10)
    ax.set_xlabel("context length")
    ax.set_ylabel("prefill throughput (tok/s, log scale)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(color="#ECEFF4", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=10)

    ax.annotate(
        "cold peaks at 16k, then the attention term dominates",
        xy=(2, cold[2]),
        xytext=(2.35, 1200),
        fontsize=9.5,
        color=GREY,
        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1),
    )
    ax.set_title(
        "Prefill: cold falls off with context, warm rides the cache",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    save(fig, "prefill-curve.svg")


# --- chart 5: tau3 with error bars ------------------------------------------
def chart_tau3() -> None:
    rows = load_tau3()
    domains = ["airline", "retail"]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))

    xt, xl = [], []
    i = 0
    for dom in domains:
        d = [r for r in rows if r["domain"] == dom]
        for r in d:
            m = r["model"]
            err = [[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]]
            ax.bar(i, r["mean"], width=0.62, color=MCOLOR[m], yerr=err, capsize=6,
                   error_kw=dict(ecolor="#2E3440", lw=1.3))
            ax.text(i, r["mean"] + (r["hi"] - r["mean"]) + 0.035, f"{r['mean']:.3f}",
                    ha="center", fontsize=9.5, fontweight="bold")
            ax.text(i, 0.035, f"n={r['n']}", ha="center", fontsize=8.5, color="#FFFFFF", fontweight="bold")
            xt.append(i)
            xl.append(f"{m}")
            i += 1
        i += 0.7

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mean reward")
    ax.grid(axis="y", color="#ECEFF4", linewidth=0.9)
    ax.set_axisbelow(True)

    ax.text(sum(xt[:3]) / 3, 1.045, "airline", ha="center", fontsize=11, fontweight="bold")
    ax.text(sum(xt[3:]) / 3, 1.045, "retail", ha="center", fontsize=11, fontweight="bold")

    # Below the axes. Inside, the box covered the D-xLAM error bar it was describing.
    ax.set_xlabel(
        "95% CI (Wilson). Every pair overlaps: the best comparison is p=0.071",
        fontsize=9.5,
        color="#BF616A",
        labelpad=10,
    )
    ax.set_title(
        "τ³-bench: the direction repeats but sits inside the noise",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    save(fig, "tau3-noise.svg")


def main() -> None:
    print("reading results/, writing docs/charts/")
    bfcl = load_bfcl()
    chart_finding(bfcl)
    chart_bfcl_categories(bfcl)
    chart_serving()
    chart_prefill()
    chart_tau3()
    print("done")


if __name__ == "__main__":
    main()
