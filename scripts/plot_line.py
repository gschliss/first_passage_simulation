"""One 1D line of the L91 sweep: t_20 for the switching walker and for the free-only
reference, as levels rather than a ratio (L96).

    python scripts/plot_line.py results/dimension/<run dirs...> \\
        --x free_fraction --out results/figures/line_free_fraction.png

Each run directory contributes one point per switching sweep point. The axis value is read
from the run's own config.json, so a line is assembled from whatever runs exist.

t_20 is the time by which 20% of ALL molecules have been captured (L91): escapes count as
never arriving, so it is undefined wherever fewer than 20% are ever captured. Those points
are simply absent from the top panel, which is why the bottom panel carries the capture
fraction — it is the readout that always exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim.jobs import merge

import vizstyle as vz

FIELD = {
    "free_fraction": ("sweeps", "switch", "free_fraction"),
    "bound_dwell": ("sweeps", "switch", "bound_dwell"),
    "target_ring": ("geometry", "target_ring"),
    "n_per_cell": ("receptors", "n_per_cell"),
    "finds_to_capture": ("receptors", "finds_to_capture"),
}
LOG_X = {"free_fraction", "bound_dwell", "n_per_cell", "finds_to_capture"}
Q = 0.20


def quantile_time(times, n_total, q=Q):
    t = np.sort(np.asarray(times, dtype=float))
    k = int(np.ceil(q * n_total))
    return float(t[k - 1]) if 0 < k <= len(t) else np.nan


def dig(cfg, path):
    if path[0] == "sweeps":
        sw = next((s for s in cfg["sweeps"] if s["name"] == path[1]), None)
        return None if sw is None else sw.get(path[2])
    node = cfg
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def layout_table(mol, sweep, idx):
    a = mol[(mol["sweep"] == sweep) & (mol.sweep_index == idx)]
    c = a[a.outcome == "captured"]
    chunks = np.sort(a.chunk.unique())
    return [(int(ch), (a.chunk.to_numpy() == ch).sum(), c.t_end.to_numpy()[c.chunk.to_numpy() == ch])
            for ch in chunks]


def level(rows, sel):
    sel = list(sel)
    n_all = sum(rows[i][1] for i in sel)
    times = np.concatenate([rows[i][2] for i in sel]) if len(sel) else np.array([])
    return quantile_time(times, n_all), (len(times) / n_all if n_all else np.nan)


def collect(dirs, x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
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
            xv = (dwells[i] if x == "bound_dwell" else (raw[i] if isinstance(raw, list) else raw))
            sw, rf = layout_table(mol, "switch", i), layout_table(mol, "reference", k + i)
            if not sw or not rf:
                continue
            n_lay = min(len(sw), len(rf))
            est = {"switch": level(sw, range(n_lay)), "reference": level(rf, range(n_lay))}
            boot = {"switch": [], "reference": []}
            for _ in range(n_boot):
                sel = rng.integers(0, n_lay, n_lay)
                boot["switch"].append(level(sw, sel)[0])
                boot["reference"].append(level(rf, sel)[0])
            for role in ("switch", "reference"):
                b = np.asarray(boot[role], dtype=float)
                lo, hi = (np.nanpercentile(b, [2.5, 97.5]) if np.isfinite(b).any() else (np.nan, np.nan))
                out.append(dict(x=xv, role=role, t20=est[role][0], lo=lo, hi=hi,
                                captured=est[role][1], n_layouts=n_lay, dir=d.name))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--x", required=True, choices=sorted(FIELD))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    vz.use()
    df = collect(args.run_dirs, args.x).sort_values("x")
    if df.empty:
        raise SystemExit("no usable runs found")
    dup = df.duplicated(["x", "role"], keep=False)
    if dup.any():
        # Two runs at one x differ along some axis this line does not show (L96).
        raise SystemExit(f"duplicate points; narrow the glob: {', '.join(sorted(df[dup]['dir'].unique()))}")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for role in ("switch", "reference"):
        r = df[df.role == role]
        ok = r[r.t20.notna()]
        ax.errorbar(ok.x, ok.t20, yerr=[ok.t20 - ok.lo, ok.hi - ok.t20],
                    fmt="o-" if role == "switch" else "s-", color=vz.ROLE[role],
                    ms=8, capsize=0, lw=2, elinewidth=1.4, label=vz.LABEL[role],
                    markeredgecolor=vz.SURFACE, markeredgewidth=1.5)
        # direct label at the right-hand end, so identity is not colour-alone
        if len(ok):
            last = ok.iloc[-1]
            ax.annotate(vz.LABEL[role].split(" (")[0], (last.x, last.t20), xytext=(9, 0),
                        textcoords="offset points", color=vz.ROLE[role], fontsize=9.5, va="center")
        for _, m in r[r.t20.isna()].iterrows():  # no t_20: under 20% ever captured
            ax.annotate(f"no t_20\n({m.captured:.0%} captured)", (m.x, ax.get_ylim()[1]),
                        ha="center", va="top", fontsize=8.5, color=vz.ROLE[role])

    ax.set_yscale("log")
    if args.x in LOG_X:
        ax.set_xscale("log")
    xs = sorted(df.x.unique())
    ax.set_xticks(xs, [vz.AXIS_TICK[args.x](v) for v in xs])
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    vz.log_ticks(ax.yaxis, *ax.get_ylim())
    ax.set_xlabel(vz.AXIS_LABEL[args.x])
    ax.set_ylabel("t_20 (s)")
    ax.set_title(args.title or f"Time for 20% of molecules to find a receptor, vs {vz.AXIS_LABEL[args.x]}",
                 loc="left", color=vz.INK, pad=10)
    ax.legend(loc="best", fontsize=9.5)
    ax.margins(x=0.12)
    vz.style(ax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
