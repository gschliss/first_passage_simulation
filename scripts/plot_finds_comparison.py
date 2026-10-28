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
    extra, curves = [], {}
    for d in dirs:
        m = merge(d / "runs_a2nm")[0]
        dwells = sorted(m[m.sweep == "switch"].bound_dwell.unique())
        for k, dwell in enumerate(dwells):  # capture times per walker; reference k pairs with switching point k
            curves[dwell] = {"switch": m[m.sweep_index == k], "reference": m[m.sweep_index == len(dwells) + k]}
        if "t_first_find" not in m:
            continue
        c = m[m.outcome == "captured"]
        for i, g in c.groupby("sweep_index"):
            role = "switch" if i < len(dwells) else "reference"  # reference k pairs with switching point k
            x = g.t_end - g.t_first_find
            extra.append(dict(role=role, bound_dwell=dwells[i % len(dwells)], extra=x.mean(),
                              extra_se=x.std(ddof=1) / np.sqrt(len(x))))
    return s, p, pd.DataFrame(extra), dict(sorted(curves.items()))


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
    s1, p1, _, c1 = load(args.first)
    s5, p5, extra, c5 = load(args.finds)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    dw = np.sort(s5.bound_dwell.unique())
    fig = plt.figure(figsize=(max(24, 3.6 * len(dw)), 11.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.32)
    top = outer[0].subgridspec(1, len(dw), wspace=0.1)
    t_hi = max(max(g.t_end.max() for g in c.values()) for c in list(c1.values()) + list(c5.values()))
    t = np.logspace(1, np.log10(t_hi), 400)
    for k, dwell in enumerate(dw):
        ax = fig.add_subplot(top[0, k])
        for role, color, _, label in ROLES:
            for c, ls, tag in ((c1, "--", "first find"), (c5, "-", "5 finds")):
                if dwell not in c:
                    continue
                g = c[dwell][role]
                times = np.sort(g.t_end[g.outcome == "captured"].to_numpy())
                f = np.searchsorted(times, t, side="right") / len(g)
                half = 1.96 * np.sqrt(f * (1 - f) / len(g))
                ax.fill_between(t, f - half, f + half, color=color, alpha=0.12, lw=0)
                ax.plot(t, f, color=color, ls=ls, lw=2, label=f"{label}: {tag}")
        ax.set_xscale("log"); ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1.02)
        ax.set_xlabel("time since release (s)")
        if k == 0:
            ax.set_ylabel("fraction captured")
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"bound {dwell * 1e3:g} ms (bands: 95% CI)", loc="left", fontsize=10)
        style(ax)
    bottom = outer[1].subgridspec(1, 4, wspace=0.25)
    axes = [fig.add_subplot(bottom[0, j]) for j in range(4)]
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
                 "D_free 1, D_bound 0.5 µm²/s, 5% free, re-armed at 10 nm; 200 molecules per condition)", fontsize=11, y=0.94)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
