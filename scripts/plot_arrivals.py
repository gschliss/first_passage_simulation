"""Arrival curves by ring when every cell carries receptors (L117).

    python scripts/plot_arrivals.py results/dimension/<run dirs...> \\
        --x free_fraction --out docs/figures/arrivals/arrivals_free_fraction.png

With `--receptors-on all`, every cell but the source absorbs, so a molecule "arrives" at
whichever cell takes its last required find, and most are soaked up by ring 1. Each curve
is the fraction of **all** released molecules absorbed on a ring-k cell by time t; it
levels off at that ring's share, so the curves for rings 1, 2 and 3 plus escapes and
timeouts add up to 1 at t_max. No t_20 is read off (L117): the whole curve is the result.

Top three rows: rings 1, 2 and 3, switching on the left and the free-only reference on the
right, one curve per cline point (light = low axis value). The two panels of a row have
separate y scales (switching shares beyond ring 1 can be orders of magnitude smaller). Bottom row: where every molecule
ended up. The reference never binds, so along a free-fraction or dwell cline it is one run
repeated; that is detected and drawn once. Also writes <out>.csv with the counts behind
every curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim.config import from_dict
from fpsim.jobs import merge
from fpsim.lattice import make_cluster

from plot_line import FIELD, dig
import vizstyle as vz

RINGS = (1, 2, 3)
FATES = ["ring 1", "ring 2", "ring 3", "ring 4+", "escaped", "still searching"]


def collect(dirs, x, refs=None):
    """``refs``: {t_max: reference arm} used where a run has no reference chunks of its own
    (switching-only runs, L118)."""
    rows = []
    for d in dirs:
        cfgs = sorted(d.glob("runs_*/config.json"))
        if not cfgs:
            print(f"  skipping {d.name}: no config.json")
            continue
        cfg = json.loads(cfgs[0].read_text())["config"]
        ring_of = make_cluster(from_dict(cfg).geometry).ring
        mol, _, problems = merge(cfgs[0].parent)
        dwells = dig(cfg, ("sweeps", "switch", "bound_dwell"))
        dwells = dwells if isinstance(dwells, list) else [dwells]
        k = max(len(dwells), 1)
        raw = dig(cfg, FIELD[x])
        for i in range(k):
            xv = dwells[i] if x == "bound_dwell" else (raw[i] if isinstance(raw, list) else raw)
            row = {"x": xv, "run": d.name}
            for role, idx in (("switch", i), ("reference", k + i)):
                a = mol[(mol["sweep"] == role) & (mol.sweep_index == idx)]
                if a.empty and role == "reference" and refs:
                    t_max = cfg["run"]["t_max"]
                    if t_max not in refs:
                        raise SystemExit(f"{d.name}: no --reference-runs entry with t_max {t_max:g}")
                    row[role] = refs[t_max]
                    continue
                if a.empty:
                    break
                cap = a.outcome == "captured"
                ring = np.where(cap, ring_of[a.final_cell.to_numpy()], -1)
                row[role] = dict(n=len(a), t=a.t_end.to_numpy(), ring=ring,
                                 escaped=int((a.outcome == "escaped").sum()),
                                 timeout=int((a.outcome == "timeout").sum()))
            else:
                rows.append(row)
    return sorted(rows, key=lambda r: r["x"])


def fates(r):
    ring = r["ring"]
    counts = [int((ring == 1).sum()), int((ring == 2).sum()), int((ring == 3).sum()),
              int((ring >= 4).sum()), r["escaped"], r["timeout"]]
    return np.array(counts) / r["n"], counts


def table(rows):
    out = []
    for r in rows:
        for role in ("switch", "reference"):
            m = r[role]
            frac, counts = fates(m)
            rec = dict(x=r["x"], walker=role, n=m["n"], run=r["run"])
            for name, c, f in zip(FATES, counts, frac):
                rec[f"n_{name.replace(' ', '_').replace('+', 'plus')}"] = c
                rec[f"frac_{name.replace(' ', '_').replace('+', 'plus')}"] = f
            for k in RINGS:
                t = m["t"][m["ring"] == k]
                rec[f"median_t_ring{k}"] = float(np.median(t)) if len(t) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--x", required=True, choices=("free_fraction", "bound_dwell"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--reference-runs", type=Path, nargs="+", default=[],
                   help="complete runs whose free-only arm stands in for runs that have none, matched "
                        "by t_max (L118)")
    p.add_argument("--t-min", type=float, default=10.0, help="left edge of the log time axis (s)")
    args = p.parse_args()

    vz.use()
    refs = {}
    for d in args.reference_runs:
        r = collect([d], "free_fraction")[0]
        refs[json.loads(sorted(d.glob("runs_*/config.json"))[0].read_text())["config"]["run"]["t_max"]] = r["reference"]
    rows = collect(args.run_dirs, args.x, refs)
    if not rows:
        raise SystemExit("no usable runs found")
    table(rows).to_csv(args.out.with_suffix(".csv"), index=False)

    shades = [vz.SEQ(v) for v in np.linspace(0.18, 0.95, len(rows))]
    ref_same = all(np.array_equal(r["reference"]["t"], rows[0]["reference"]["t"]) for r in rows)
    t_hi = max(max(r[role]["t"].max() for role in ("switch", "reference")) for r in rows)

    fig = plt.figure(figsize=(12.4, 14.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.9], hspace=0.42, wspace=0.08)
    for row_i, k in enumerate(RINGS):
        t0 = min(args.t_min, 0.1) if k == 1 else args.t_min  # ring 1 fills within seconds
        axes = [fig.add_subplot(gs[row_i, 0]), None]
        axes[1] = fig.add_subplot(gs[row_i, 1], sharex=axes[0])  # own y: switching ring-3 shares can be ~1e-5
        for ax, role, name in ((axes[0], "switch", "Switching"),
                               (axes[1], "reference", "Non-switching (free-only, D = 0.5)")):
            drawn = rows[:1] if (role == "reference" and ref_same) else rows
            for r, colour in zip(drawn, shades if len(drawn) > 1 else [vz.SEQ(0.62)]):
                m = r[role]
                t = np.sort(m["t"][m["ring"] == k])
                y = np.arange(1, len(t) + 1) / m["n"]
                lab = (vz.AXIS_TICK[args.x](r["x"]) if len(drawn) > 1 else "every point (identical)")
                ax.step(np.r_[t0, t], np.r_[0, y], where="post", color=colour, lw=1.9,
                        label=f"{lab}  ({len(t)})")
            ax.set_xscale("log")
            ax.set_xlim(t0, t_hi * 1.2)
            ax.set_title(f"{name}: arrivals at ring {k}", loc="left", fontsize=11, color=vz.INK, pad=6)
            vz.style(ax)
            ax.legend(loc="upper left", fontsize=8.5, title=vz.AXIS_LABEL[args.x] + "  (count)",
                      title_fontsize=8.5, alignment="left")
            if row_i == len(RINGS) - 1:
                ax.set_xlabel("time (s)")
        axes[0].set_ylabel(f"fraction of all molecules\nabsorbed at ring {k}")
        for a in axes:
            a.set_ylim(bottom=0)

    # Where every molecule ended up: one stacked bar per point and walker.
    ax = fig.add_subplot(gs[3, :])
    colours = [vz.SEQ(0.25), vz.SEQ(0.55), vz.SEQ(0.9), vz.MUTED, vz.REFERENCE, vz.GRID]
    labels, x0 = [], 0
    for r in rows:
        for role in ("switch", "reference"):
            frac, _ = fates(r[role])
            bottom = 0.0
            for f, c in zip(frac, colours):
                ax.bar(x0, f, bottom=bottom, color=c, width=0.8,
                       edgecolor=vz.SECONDARY if role == "reference" else "none", linewidth=0.6)
                bottom += f
            labels.append(f"{vz.AXIS_TICK[args.x](r['x'])}\n{'sw' if role == 'switch' else 'ref'}")
            x0 += 1
        x0 += 0.5
    ax.set_xticks([i + 0.5 * (i // 2) for i in range(len(labels))], labels, fontsize=8.5)
    ax.set_ylabel("fraction of molecules")
    ax.set_ylim(0, 1)
    ax.set_title("Where every molecule ended up (sw = switching, ref = free-only, outlined)",
                 loc="left", fontsize=11, color=vz.INK, pad=6)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colours], FATES, ncol=6,
              loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=9)
    vz.style(ax)

    if args.title:
        fig.suptitle(args.title, fontsize=12.5, x=0.077, ha="left", y=0.995)
    vz.save(fig, args.out)
    print(f"Saved {args.out} (+ .pdf, .csv)  ({len(rows)} points"
          + (", reference identical across the cline" if ref_same else "") + ")")


if __name__ == "__main__":
    main()
