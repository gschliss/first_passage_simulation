"""Cumulative capture curves along one 1-D cline, showing where t_20 comes from (L107).

    python scripts/plot_capture_cline.py results/dimension/<run dirs...> \\
        --x free_fraction --out docs/figures/capture_free_fraction.png

Same run directories and same axis as `plot_line.py`, which plots only the t_20 levels this
reads off. Here the whole curve is drawn, so the statistic can be seen rather than taken on
trust: each curve is the fraction of **all** molecules captured by time t, and t_20 is simply
where it crosses 0.20.

Two things the curves show that a t_20 level cannot:

  * Each curve plateaus at that condition's capture fraction, not at 1. Molecules that escape
    the cluster or are still searching at `t_max` never arrive. Where the plateau is below
    0.20 there is no t_20 at all, which is why cells go blank in the grids (L91).
  * How well t_20 stands in for the rate. The curves are close to a single rising front, so
    one crossing time summarises them; where a curve is unusually shallow, 0.2/t_20 flatters
    it and the later part of the capture is slower than that.

The axis value orders a single-hue ramp, light = low. The reference walker never binds, so
along a free-fraction or dwell cline its curves are the same run repeated; that is detected
and drawn once rather than as identical overlapping curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fpsim.jobs import merge

from plot_line import FIELD, LOG_X, dig, quantile_time
import vizstyle as vz

Q = 0.20


def ecdf(times, n_all):
    """Step curve of captured-so-far / all molecules, and the time it crosses Q."""
    t = np.sort(np.asarray(times, dtype=float))
    y = np.arange(1, len(t) + 1) / n_all
    return t, y, quantile_time(t, n_all, Q)


def collect(dirs, x):
    out = []
    for d in dirs:
        cfgs = sorted(d.glob("runs_*/config.json"))
        if not cfgs:
            print(f"  skipping {d.name}: no config.json")
            continue
        cfg = json.loads(cfgs[0].read_text())["config"]
        cfg.setdefault("receptors", {}).setdefault("finds_to_capture", 1)
        mol, _, _ = merge(cfgs[0].parent)
        dwells = dig(cfg, ("sweeps", "switch", "bound_dwell"))
        dwells = [dwells] if not isinstance(dwells, list) else dwells
        k = max(len(dwells), 1)
        raw = dig(cfg, FIELD[x])
        for i in range(k):
            xv = dwells[i] if x == "bound_dwell" else (raw[i] if isinstance(raw, list) else raw)
            row = {"x": xv}
            for role, idx in (("switch", i), ("reference", k + i)):
                a = mol[(mol["sweep"] == role) & (mol.sweep_index == idx)]
                if a.empty:
                    break
                cap = a[a.outcome == "captured"].t_end.to_numpy()
                row[role] = ecdf(cap, len(a))
                row[role + "_cap"] = len(cap) / len(a)
            else:
                out.append(row)
    return sorted(out, key=lambda r: r["x"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--x", required=True, choices=sorted(FIELD))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    vz.use()
    rows = collect(args.run_dirs, args.x)
    if not rows:
        raise SystemExit("no usable runs found")

    shades = [vz.SEQ(v) for v in np.linspace(0.18, 0.95, len(rows))]
    # The free-only walker ignores the free fraction and the dwell, so along those clines its
    # curves are one run repeated; say so once instead of overplotting identical steps.
    ref_same = all(np.array_equal(r["reference"][0], rows[0]["reference"][0]) for r in rows)

    # sharex as well as sharey: the panels are there to be compared, and letting each
    # autoscale its own time axis would make the slower walker look faster.
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True, sharex=True,
                             gridspec_kw=dict(wspace=0.07))
    for ax, role, name in ((axes[0], "switch", "Switching"),
                           (axes[1], "reference", "Non-switching (free-only, D = 0.5)")):
        drawn = rows[:1] if (role == "reference" and ref_same) else rows
        for r, colour in zip(drawn, shades if len(drawn) > 1 else [vz.SEQ(0.62)]):
            t, y, t20 = r[role]
            ax.step(t, y, where="post", color=colour, lw=1.9,
                    label=vz.AXIS_TICK[args.x](r["x"]) if len(drawn) > 1 else "every point (identical)")
            if np.isfinite(t20):
                ax.plot([t20], [Q], "o", color=colour, ms=7,
                        markeredgecolor=vz.SURFACE, markeredgewidth=1.3, zorder=5)
        ax.axhline(Q, color=vz.INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("time (s)")
        ax.set_title(name, loc="left", fontsize=11, color=vz.INK, pad=8)
        ax.set_ylim(0, 1.02)
        vz.style(ax)
        ax.legend(loc="upper left", fontsize=9, title=vz.AXIS_LABEL[args.x],
                  title_fontsize=9, alignment="left")

    axes[0].set_ylabel("fraction of all molecules captured")
    axes[0].annotate("t_20: the 20% crossing", xy=(axes[0].get_xlim()[0], Q), xytext=(4, -12),
                     textcoords="offset points", fontsize=9, color=vz.SECONDARY, va="top")
    if args.title:
        fig.suptitle(args.title, fontsize=12.5, x=0.077, ha="left", y=1.02)
    fig.text(0.077, -0.035,
             "Each curve levels off at that condition's capture fraction, not at 1: escapes and "
             "molecules still searching at t_max never arrive. A curve that plateaus below 0.20 "
             "has no t_20.", ha="left", fontsize=9.5, color=vz.SECONDARY)

    vz.save(fig, args.out)
    print(f"Saved {args.out} (+ .pdf)  ({len(rows)} points"
          + (", reference identical across the cline" if ref_same else "") + ")")


if __name__ == "__main__":
    main()
