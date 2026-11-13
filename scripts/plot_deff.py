"""Effective diffusion of the switching walker along one cline (paper experiments 1-2, L121).

    python scripts/plot_deff.py results/paper_2026_11_02/01_confinement/deff/* \\
        --x free_fraction --out paper_figures/2026_11_02/01_confinement/deff_fraction_free.png

Reads each run's `calibration.csv` (written by `dimension_experiment.py --calibration-only`):
D_eff from the time-averaged MSD at the long lag, in a receptor-free 12-ring cluster, with the
short-lag value alongside. The free fraction is read from the run name (`_f<pct>`; no suffix
means 5%), the dwell from the CSV. The microscopic D = 0.5 is drawn for scale, and, where the
100%-free point is among the runs, the free-only walker's D_eff (the packing factor times D).
Also writes <out>.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vizstyle as vz

D_MICRO = 0.5


def load(dirs):
    rows = []
    for d in dirs:
        f = d / "calibration.csv"
        if not f.exists():
            print(f"  skipping {d.name}: no calibration.csv")
            continue
        c = pd.read_csv(f)
        c = c[c.role == "switch"].iloc[0]
        m = re.search(r"_f([\d.]+)_", d.name + "_")
        rows.append(dict(run=d.name, free_fraction=float(m.group(1)) / 100 if m else 0.05,
                         bound_dwell=float(c.bound_dwell), d_eff=c.d_eff, d_eff_se=c.d_eff_se,
                         d_eff_short=c.d_eff_short, n=int(c.n), escaped=int(c.escaped), t_max=c.t_max))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--x", required=True, choices=("free_fraction", "bound_dwell"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--compare", type=Path, nargs="+", default=[],
                   help="a second set of runs (e.g. the 12-ring cluster) drawn faded for comparison (L122)")
    p.add_argument("--free-only", type=Path, default=None,
                   help="a 100%%-free calibration run whose D_eff is drawn as the free-only level")
    args = p.parse_args()

    vz.use()
    t = load(args.run_dirs)
    if t.empty:
        raise SystemExit("no calibration runs found")
    t = t.sort_values(args.x)
    t.to_csv(args.out.with_suffix(".csv"), index=False)

    free_only = None
    if args.free_only is not None:
        f = load([args.free_only])
        free_only = float(f.d_eff.iloc[0]) if len(f) else None
    elif (t.free_fraction == 1.0).any():
        free_only = float(t[t.free_fraction == 1.0].d_eff.iloc[0])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.errorbar(t[args.x], t.d_eff, yerr=1.96 * t.d_eff_se, color=vz.SWITCH, marker="o", lw=2,
                capsize=3, label="switching: D_eff (long lag)")
    if args.compare:
        c = load(args.compare).sort_values(args.x)
        if len(c):
            ax.errorbar(c[args.x], c.d_eff, yerr=1.96 * c.d_eff_se, color=vz.MUTED, marker="o", ms=5,
                        lw=1.2, capsize=2, alpha=0.8, label="comparison set (e.g. 12-ring cluster)")
            t = t.merge(c[[args.x, "d_eff", "escaped"]].rename(columns={"d_eff": "d_eff_compare",
                        "escaped": "escaped_compare"}), on=args.x, how="left")
            t.to_csv(args.out.with_suffix(".csv"), index=False)
    ax.plot(t[args.x], t.d_eff_short, color=vz.SWITCH, ls=(0, (3, 2)), lw=1.2, alpha=0.7,
            label="switching: short-lag value")
    ax.axhline(D_MICRO, color=vz.MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("microscopic D = 0.5", xy=(1, D_MICRO), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8.5, color=vz.SECONDARY)
    if free_only is not None:
        ax.axhline(free_only, color=vz.REFERENCE, lw=1.4)
        ax.annotate(f"free-only D_eff = {free_only:.3f}", xy=(0, free_only), xycoords=("axes fraction", "data"),
                    xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=vz.REFERENCE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(vz.AXIS_LABEL[args.x])
    ax.set_ylabel("effective diffusion D_eff (µm²/s)")
    vz.style(ax)
    ax.legend(loc="lower right", fontsize=9)
    if args.title:
        ax.set_title(args.title, loc="left", fontsize=11, color=vz.INK)
    print("Saved " + ", ".join(str(q) for q in vz.save(fig, args.out)))


if __name__ == "__main__":
    main()
