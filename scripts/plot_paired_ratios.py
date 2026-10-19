"""Combine paired dimension-experiment runs (L57-L59) into one figure across bound dwells.

Each run directory must come from scripts/dimension_experiment.py with --layouts > 1 and
the same layouts, free fraction and geometry, so the points are comparable.

Usage:
    python scripts/plot_paired_ratios.py results/dimension/lattice2d_ring2_rec2_dw5-16-50_paired10 \\
        results/dimension/lattice2d_ring2_rec2_dw100-300_paired10 --out results/figures/paired_ratios_lattice2d.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e6e6e6"
D_BOUND = 0.5  # µm²/s, for the slide-length axis (L20)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load(dirs):
    paired, summary, cal, check = [], [], [], []
    for d in dirs:
        paired.append(pd.read_csv(d / "paired_ratios.csv"))
        summary.append(pd.read_csv(d / "summary.csv"))
        c = pd.read_csv(d / "calibration.csv")
        cal.append(c[c.role == "switch"])
        check.append(pd.read_csv(d / "calibration_reference.csv"))
    sort = lambda df: df.sort_values("bound_dwell").reset_index(drop=True)  # noqa: E731
    return sort(pd.concat(paired)), pd.concat(summary), sort(pd.concat(cal)), sort(pd.concat(check))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    paired, s, cal, check = load(args.run_dirs)
    dw = paired.bound_dwell.to_numpy() * 1e3  # ms
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for off, col, mk, color, label in ((0.97, "total", "o", BLUE, "total capture time"),
                                       (1.03, "local", "D", INK, "local search (at a target cell)")):
        ax.errorbar(dw * off, paired[col], yerr=[paired[col] - paired[f"{col}_lo"], paired[f"{col}_hi"] - paired[col]],
                    fmt=mk + "-", color=color, ms=8, capsize=3, lw=1.5, label=label)
    ax.axhline(1, color=MUTED, lw=1)
    ax.annotate("1 = switching no different from plain diffusion", (dw.max(), 1), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(dw, [f"{v:g}" for v in dw]); ax.minorticks_off()
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("switching / free-only at equal D_eff (mean time)")
    ax.set_title(f"A. Paired over {int(paired.n_layouts.iloc[0])} shared receptor layouts (95% CI)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    top = ax.secondary_xaxis("top", functions=(lambda ms: np.sqrt(2 * D_BOUND * np.maximum(ms, 1e-9) / 1e3),
                                               lambda lam: lam**2 / (2 * D_BOUND) * 1e3))
    top.set_xlabel("typical slide per binding, √(2·D_bound·τ_bound) (µm)", color=MUTED)
    top.set_xticks(np.sqrt(2 * D_BOUND * dw / 1e3), [f"{v:.2f}" for v in np.sqrt(2 * D_BOUND * dw / 1e3)])
    top.xaxis.set_minor_locator(plt.NullLocator())
    top.tick_params(colors=MUTED)
    style(ax)

    def by_role(ax, col, se, ylabel, title):
        for off, role, color, mk, label in ((0.97, "switch", BLUE, "o", "switching (5% free)"),
                                            (1.03, "reference", ORANGE, "s", "free-only, same D_eff")):
            r = s[s.role == role].sort_values("bound_dwell")
            ax.errorbar(r.bound_dwell * 1e3 * off, r[col], yerr=1.96 * r[se], fmt=mk + "-", color=color, ms=8,
                        capsize=3, lw=1.5, label=label)
        ax.set_xscale("log"); ax.set_xticks(dw, [f"{v:g}" for v in dw]); ax.minorticks_off()
        ax.set_xlabel("mean bound dwell of the switching walker (ms)")
        ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left"); ax.legend(frameon=False, fontsize=9, loc="lower left")
        style(ax)

    by_role(axes[0, 1], "local_mean", "local_se", "mean time from first arrival to capture (s, 95% CI)",
            "B. Local search: finding the receptor at a target cell")
    by_role(axes[1, 0], "t_mean", "t_se", "mean capture time (s, 95% CI)", "C. Total capture time")

    ax = axes[1, 1]
    ax.errorbar(cal.bound_dwell * 1e3 * 0.97, cal.d_eff, yerr=1.96 * cal.d_eff_se, fmt="o", color=BLUE, ms=8, capsize=3,
                label="switching (5% free)")
    ax.errorbar(check.bound_dwell * 1e3 * 1.03, check.d_eff, yerr=1.96 * check.d_eff_se, fmt="s", color=ORANGE, ms=8,
                capsize=3, label="free-only, same D_eff")
    ax.set_xscale("log"); ax.set_xticks(dw, [f"{v:g}" for v in dw]); ax.minorticks_off()
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("measured D_eff (µm²/s, 95% CI)"); ax.set_ylim(0, 0.8)
    ax.set_title("D. Long-range spreading is matched", loc="left"); ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)

    fig.suptitle("Switching vs non-switching diffusion at equal D_eff (2D, 5% free, ring-2 targets with 2 receptors, "
                 "2 nm; 200 molecules per condition)", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(paired.to_string(index=False))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
