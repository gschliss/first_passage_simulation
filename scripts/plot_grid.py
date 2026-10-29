"""Heatmap of t_20 over a 2D parameter sweep, switching beside non-switching (L96).

    python scripts/plot_grid.py results/dimension/<run dirs...> \\
        --x free_fraction --y target_ring --out results/figures/grid_free_x_ring.png

Each run directory is one cell: its coordinates are read from the run's own
config.json, so the grid is assembled from whatever runs exist and missing cells are
left blank. Both panels share one sequential colour scale, light = fast, so the two
walkers can be compared by eye.

t_20 is the time by which 20% of ALL molecules have been captured (L91). Escapes count
as never arriving, so a cell where fewer than 20% are ever captured has no t_20 and is
drawn blank; the printed capture fraction is the readout there instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vizstyle as vz

FIELD = {
    "free_fraction": ("sweeps", "switch", "free_fraction"),
    "bound_dwell": ("sweeps", "switch", "bound_dwell"),
    "target_ring": ("geometry", "target_ring"),
    "n_per_cell": ("receptors", "n_per_cell"),
    "finds_to_capture": ("receptors", "finds_to_capture"),
}


def dig(cfg, path):
    if path[0] == "sweeps":
        sw = next((s for s in cfg["sweeps"] if s["name"] == path[1]), None)
        v = None if sw is None else sw.get(path[2])
        return v[0] if isinstance(v, list) and len(v) == 1 else v
    node = cfg
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def load(dirs, x, y):
    rows = []
    for d in dirs:
        paired, cfgs = d / "paired_ratios.csv", sorted(d.glob("runs_*/config.json"))
        if not paired.exists() or not cfgs:
            print(f"  skipping {d.name}: not finished")
            continue
        cfg = json.loads(cfgs[0].read_text())["config"]
        cfg.setdefault("receptors", {}).setdefault("finds_to_capture", 1)
        for _, r in pd.read_csv(paired).iterrows():
            rows.append(dict(r, dir=d.name, **{x: dig(cfg, FIELD[x]), y: dig(cfg, FIELD[y])}))
    return pd.DataFrame(rows)


def panel(ax, piv, cap, title, norm, x, y, show_y):
    v = piv.to_numpy(dtype=float)
    ax.imshow(np.ma.masked_invalid(v), cmap=vz.SEQ, norm=norm, origin="lower", aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                # white on the dark half of the ramp, ink on the light half
                dark = norm(v[i, j]) > 0.55
                ax.text(j, i, vz.fmt_time(v[i, j]), ha="center", va="center", fontsize=10,
                        color="#ffffff" if dark else vz.INK)
            else:
                frac = cap.to_numpy(dtype=float)[i, j] if cap is not None else np.nan
                note = "n/a" if not np.isfinite(frac) else f"n/a\n{frac:.0%} cap"
                ax.text(j, i, note, ha="center", va="center", fontsize=8.5, color=vz.MUTED)
    ax.set_xticks(range(piv.shape[1]), [vz.AXIS_TICK[x](c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [vz.AXIS_TICK[y](r) for r in piv.index] if show_y else [])
    ax.set_xlabel(vz.AXIS_LABEL[x])
    if show_y:
        ax.set_ylabel(vz.AXIS_LABEL[y])
    ax.set_title(title, loc="left", fontsize=11, color=vz.INK, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--x", required=True, choices=sorted(FIELD))
    p.add_argument("--y", required=True, choices=sorted(FIELD))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    vz.use()
    df = load(args.run_dirs, args.x, args.y)
    if df.empty:
        raise SystemExit("no finished runs found")
    if df[[args.x, args.y]].isna().any().any():
        raise SystemExit(f"some runs do not define {args.x}/{args.y}")
    dup = df.duplicated([args.x, args.y], keep=False)
    if dup.any():
        # Run dirs are one per cell, so a repeat means the caller's glob swept in runs that
        # differ along some third axis (L96) and the figure would silently show one of them.
        names = ", ".join(sorted(df[dup]["dir"].unique())) if "dir" in df else ""
        raise SystemExit(f"{dup.sum()} runs share a cell; widen the key or narrow the glob: {names}")

    piv = {c: df.pivot(index=args.y, columns=args.x, values=c)
           for c in ("t20_switch", "t20_ref", "captured_switch", "captured_ref")}
    vals = np.concatenate([piv["t20_switch"].to_numpy(dtype=float).ravel(),
                           piv["t20_ref"].to_numpy(dtype=float).ravel()])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        raise SystemExit("every cell is missing t_20")
    norm = plt.matplotlib.colors.LogNorm(vmin=vals.min(), vmax=vals.max())

    fig, axes = plt.subplots(1, 2, figsize=(5.2 + 1.9 * piv["t20_switch"].shape[1], 1.0 + 0.95 * piv["t20_switch"].shape[0]),
                             gridspec_kw=dict(wspace=0.08))
    panel(axes[0], piv["t20_switch"], piv["captured_switch"], "Switching", norm, args.x, args.y, True)
    panel(axes[1], piv["t20_ref"], piv["captured_ref"], "Non-switching (free-only, D = 0.5)",
          norm, args.x, args.y, False)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=vz.SEQ), ax=axes, orientation="horizontal",
                      fraction=0.08, pad=0.18, aspect=46)
    cb.set_label("t_20: time for 20% of all molecules to be captured (s) — lighter is faster", fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    vz.log_ticks(cb.ax.xaxis, norm.vmin, norm.vmax, want=8)

    if args.title:
        fig.suptitle(args.title, fontsize=12, x=0.125, ha="left", y=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
