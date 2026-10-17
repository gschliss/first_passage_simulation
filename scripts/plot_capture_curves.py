"""Cumulative capture curves across the points of one sweep, from a merged run directory.

Usage:
    python scripts/merge.py results/runs/lattice2d --chunks-per-point 2
    python scripts/plot_capture_curves.py results/runs/lattice2d --sweep free_fraction

Panels: A, B cumulative fraction captured vs time (linear and log time) with 95%
pointwise binomial bands; C mean and median capture time vs the swept quantity;
D share of captures made while membrane-bound.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ordinal single-hue ramp (dataviz reference palette, blue steps 250 -> 650): light = low.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
SWEPT = {"free_fraction": ("free_fraction", "fraction free", lambda v: f"{v:.0%} free"),
         "dwell": ("bound_dwell", "mean bound dwell (s)", lambda v: f"{v * 1e3:.0f} ms bound")}


def ramp(n):
    if n <= len(RAMP):
        return RAMP[-n:] if n < 3 else [RAMP[int(round(k))] for k in np.linspace(0, len(RAMP) - 1, n)]
    raise ValueError("more points than ramp steps; facet instead")


def cumulative(m, t):
    times = np.sort(m.t_end[m.outcome == "captured"].to_numpy())
    f = np.searchsorted(times, t, side="right") / len(m)
    half = 1.96 * np.sqrt(f * (1 - f) / len(m))
    return f, f - half, f + half


def spread(ys, gap=0.045, top=1.0):
    """Nudge label heights apart (keeping order, never above ``top``) so labels don't overlap."""
    order = np.argsort(ys)[::-1]  # highest first
    out = np.array(ys, dtype=float)
    ceiling = top
    for k in order:
        out[k] = min(out[k], ceiling)
        ceiling = out[k] - gap
    return out


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--sweep", default="free_fraction", choices=sorted(SWEPT))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    molecules = pd.read_parquet(args.run_dir / "merged.parquet")
    col, axis_label, fmt = SWEPT[args.sweep]
    m_all = molecules[molecules.sweep == args.sweep]
    if m_all.empty:
        raise SystemExit(f"no molecules from sweep {args.sweep!r} in {args.run_dir}")
    groups = sorted(m_all.groupby(col), key=lambda kv: kv[0])
    colors = ramp(len(groups))
    t_cap = m_all.t_end[m_all.outcome == "captured"]
    t_hi = float(np.percentile(t_cap, 99.5)) if len(t_cap) else float(m_all.t_end.max())

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [1.25, 1]})
    rows = []
    for ax, scale in ((axes[0, 0], "linear"), (axes[0, 1], "log")):
        t = np.linspace(0, t_hi, 600) if scale == "linear" else np.logspace(0, np.log10(m_all.t_end.max()), 600)
        ends = []
        for (value, m), color in zip(groups, colors):
            f, lo, hi = cumulative(m, t)
            ax.fill_between(t, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(t, f, color=color, lw=2, label=f"{fmt(value)} (n = {len(m)})")
            ends.append(f[-1])
        if scale == "linear":
            for (value, _), color, y in zip(groups, colors, spread(ends)):
                ax.annotate(fmt(value), (t[-1], y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK)
                ax.plot([t[-1] * 1.003, t[-1] * 1.012], [y, y], color=color, lw=2)
        ax.set_xscale(scale)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("time since release (s)" + (", log scale" if scale == "log" else ""))
        ax.set_ylabel("fraction of molecules captured")
        style(ax)
    axes[0, 0].set_xlim(0, t_hi * 1.12)
    axes[0, 0].set_title("A. Cumulative capture (bands: 95% CI)", loc="left", fontsize=11)
    axes[0, 1].set_title("B. Same, log time: early arrivals", loc="left", fontsize=11)
    axes[0, 1].legend(loc="upper left", fontsize=9, frameon=False)

    for value, m in groups:
        cap = m[m.outcome == "captured"]
        rows.append(dict(value=value, n=len(m), captured=len(cap) / len(m),
                         escaped=(m.outcome == "escaped").mean(), timeout=(m.outcome == "timeout").mean(),
                         mean=cap.t_end.mean(), se=cap.t_end.std(ddof=1) / np.sqrt(len(cap)),
                         median=cap.t_end.median(), q10=cap.t_end.quantile(0.1), q90=cap.t_end.quantile(0.9),
                         from_bound=(cap.final_state == "bound").mean(),
                         from_bound_se=np.sqrt((cap.final_state == "bound").mean() * (1 - (cap.final_state == "bound").mean()) / len(cap))))
    s = pd.DataFrame(rows)

    ax = axes[1, 0]
    ax.fill_between(s.value, s.q10, s.q90, color=RAMP[1], alpha=0.25, lw=0, label="10th–90th percentile")
    ax.errorbar(s.value, s["mean"], yerr=1.96 * s.se, fmt="o-", color=RAMP[-1], lw=2, ms=8, capsize=3, label="mean (95% CI)")
    ax.plot(s.value, s["median"], "s--", color=RAMP[2], lw=2, ms=8, label="median")
    ax.set_xscale("log")
    ax.set_xticks(s.value, [fmt(v).split()[0] for v in s.value])
    ax.minorticks_off()
    ax.set_xlabel(axis_label)
    ax.set_ylabel("capture time (s)")
    ax.set_title("C. Capture time", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    ax = axes[1, 1]
    ax.errorbar(s.value, s.from_bound, yerr=1.96 * s.from_bound_se, fmt="o-", color=RAMP[-1], lw=2, ms=8, capsize=3)
    for _, r in s.iterrows():
        ax.annotate(f"{r.from_bound:.0%}", (r.value, r.from_bound), xytext=(8, 6), textcoords="offset points",
                    ha="left", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(s.value, [fmt(v).split()[0] for v in s.value])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(axis_label)
    ax.set_ylabel("share of captures made while membrane-bound")
    ax.set_title("D. How molecules are captured (95% CI)", loc="left", fontsize=11)
    style(ax)

    meta = (f"{len(m_all)} molecules; {int(m_all.outcome.eq('escaped').sum())} escaped, "
            f"{int(m_all.outcome.eq('timeout').sum())} still searching at t_max")
    fig.suptitle(f"Capture across the '{args.sweep}' sweep ({args.run_dir.name}): {meta}", fontsize=12, color=INK)
    fig.tight_layout()
    out = args.out or args.run_dir / f"capture_curves_{args.sweep}.png"
    fig.savefig(out, dpi=150)
    s.to_csv(out.with_suffix(".csv"), index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
