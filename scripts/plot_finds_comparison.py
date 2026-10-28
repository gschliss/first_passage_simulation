"""First passage vs n-find capture (L86) for the same walkers, geometry and dwells.

Reads summary.csv, paired_ratios.csv and the capture runs of first-passage run directories
(--first, one or more) and n-find run directories (--finds, one or more).

Usage:
    python scripts/plot_finds_comparison.py --first DIR [DIR ...] --finds DIR [DIR ...] --out FIG.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim.jobs import merge

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e6e6e6"
ROLES = (("switch", BLUE, "o", "switching (5% free)"), ("reference", ORANGE, "s", "free-only, same D_eff"))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load(dirs):
    s = pd.concat([pd.read_csv(d / "summary.csv") for d in dirs])
    p = pd.concat([pd.read_csv(d / "paired_ratios.csv") for d in dirs]).sort_values("bound_dwell")
    extra = []
    for d in dirs:
        m = merge(d / "runs_a2nm")[0]
        if "t_first_find" not in m:
            continue
        dwells = sorted(m[m.sweep == "switch"].bound_dwell.unique())
        c = m[m.outcome == "captured"]
        for i, g in c.groupby("sweep_index"):
            role = "switch" if i < len(dwells) else "reference"  # reference k pairs with switching point k
            x = g.t_end - g.t_first_find
            extra.append(dict(role=role, bound_dwell=dwells[i % len(dwells)], extra=x.mean(),
                              extra_se=x.std(ddof=1) / np.sqrt(len(x))))
    return s, p, pd.DataFrame(extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=Path, nargs="+", required=True)
    ap.add_argument("--finds", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import json
    info = json.loads((args.finds[0] / "reference.json").read_text())
    global ROLES
    ROLES = (ROLES[0], ("reference", ORANGE, "s", info.get("label", "free-only, same D_eff")))
    matched = info.get("matched", True)
    s1, p1, _ = load(args.first)
    s5, p5, extra = load(args.finds)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6))
    dw = np.sort(s5.bound_dwell.unique())
    ticks = (dw * 1e3, [f"{v * 1e3:g}" for v in dw])

    ax = axes[0]
    for role, color, mk, label in ROLES:
        for s, ls, fill, tag in ((s1, "--", "none", "first find"), (s5, "-", color, "5 finds")):
            r = s[s.role == role].sort_values("bound_dwell")
            ax.errorbar(r.bound_dwell * 1e3, r.t_mean, yerr=1.96 * r.t_se, fmt=mk, ls=ls, color=color, mfc=fill, ms=7,
                        capsize=3, lw=1.5, label=f"{label}: {tag}")
    ax.set_xscale("log"); ax.set_xticks(*ticks); ax.minorticks_off(); ax.set_ylim(bottom=0)
    ax.set_xlabel("mean bound dwell of the switching walker (ms)"); ax.set_ylabel("mean capture time (s, 95% CI)")
    ax.set_title("A. Capture time of captured molecules:\nfirst find (dashed) vs 5th find (solid)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); style(ax)

    ax = axes[1]
    for role, color, mk, label in ROLES:
        r = extra[extra.role == role].sort_values("bound_dwell")
        ax.errorbar(r.bound_dwell * 1e3, r.extra, yerr=1.96 * r.extra_se, fmt=mk + "-", color=color, ms=7, capsize=3,
                    lw=1.5, label=label)
    ax.set_xscale("log"); ax.set_xticks(*ticks); ax.minorticks_off(); ax.set_yscale("log")
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("mean time from 1st to 5th find (s, 95% CI)")
    ax.set_title("B. Cost of the 4 extra finds", loc="left"); ax.legend(frameon=False, fontsize=9); style(ax)

    ax = axes[2]
    for p, ls, fill, tag in ((p1, "--", "none", "first find"), (p5, "-", BLUE, "5 finds")):
        ax.errorbar(p.bound_dwell * 1e3, p.total, yerr=[p.total - p.total_lo, p.total_hi - p.total], fmt="o", ls=ls,
                    color=BLUE, mfc=fill, ms=7, capsize=3, lw=1.5, label=tag)
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_xticks(*ticks); ax.minorticks_off(); ax.set_ylim(bottom=0)
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("switching / free-only" + (" at equal D_eff" if matched else "") + " (mean time)")
    ax.set_title("C. Paired ratio over 10 shared layouts (95% CI)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left"); style(ax)

    ax = axes[3]
    for role, color, mk, label in ROLES:
        for s, ls, fill, tag in ((s1, "--", "none", "first find"), (s5, "-", color, "5 finds")):
            r = s[s.role == role].sort_values("bound_dwell")
            half = 1.96 * np.sqrt(r.captured * (1 - r.captured) / r.n)
            ax.errorbar(r.bound_dwell * 1e3, r.captured, yerr=half, fmt=mk, ls=ls, color=color, mfc=fill, ms=7,
                        capsize=3, lw=1.5, label=f"{label}: {tag}")
    ax.set_xscale("log"); ax.set_xticks(*ticks); ax.minorticks_off(); ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("fraction captured (95% CI); the rest escape")
    ax.set_title("D. Fraction that reach capture", loc="left"); ax.legend(frameon=False, fontsize=8, loc="center left")
    style(ax)

    fig.suptitle("First passage vs capture on the 5th receptor find (3D, ring-2 targets, 1000 receptors per cell, "
                 "D_free 1, D_bound 0.5 µm²/s, 5% free, re-armed at 10 nm; 200 molecules per condition)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
