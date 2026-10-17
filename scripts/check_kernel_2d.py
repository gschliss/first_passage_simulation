"""Statistical validation of the 2D kernel (build step 2; L31, L37).

  A. Free fraction vs accuracy setting at each sweep point (target 5%).
  B. Fraction of free trips ending on a different cell: adaptive vs fixed vs the
     independent pure-Python hop demo (scripts/demo_cell_hopping.py).
  C. Adaptive / fixed ratios of free fraction and spreading (mean squared displacement).
  D. Cost: steps per molecule per simulated second, adaptive vs fixed.

Writes results/checks/kernel2d_checks.csv and results/figures/kernel2d_checks.png.

Usage:
    python scripts/check_kernel_2d.py            # ~15-20 min on 8 cores
    python scripts/check_kernel_2d.py --quick    # smaller samples, for a smoke test
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim import load_config
from fpsim.config import Stepping
from fpsim.simulate import simulate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "lattice2d.yaml"
COLORS = {"5 ms": "#1f77b4", "71 ms": "#ff7f0e", "1000 ms": "#2ca02c"}


def label(dwell):
    return f"{dwell * 1e3:.0f} ms"


def summarize(r, **extra):
    m = r.molecules
    f = m.time_free / m.t_end
    disp2 = ((m.x1 - m.x0) ** 2 + (m.y1 - m.y0) ** 2) / m.t_end  # µm²/s per molecule
    free_periods = m.n_trips + (m.start_state == "free")
    p = m.n_cell_changes.sum() / free_periods.sum()
    return dict(
        dwell=r.meta["sweep_point"]["bound_dwell"], mode=r.meta["stepping_mode"],
        distance_fraction=r.meta["config"]["stepping"]["distance_fraction"],
        n=len(m), t_max=r.meta["t_max"],
        free=f.mean(), free_se=f.std(ddof=1) / np.sqrt(len(m)),
        cell_change=p, cell_change_se=np.sqrt(p * (1 - p) / free_periods.sum()),
        msd_rate=disp2.mean(), msd_rate_se=disp2.std(ddof=1) / np.sqrt(len(m)),
        steps_per_s=(m.n_steps / m.t_end).mean(), wall_s=r.meta["wall_time_s"], **extra,
    )


def run(cfg, i, n, t_max, **stepping):
    c = dataclasses.replace(cfg, stepping=Stepping(**{**dataclasses.asdict(cfg.stepping), **stepping}))
    r = simulate(c, i, n_molecules=n, t_max=t_max)
    row = summarize(r, check="accuracy" if stepping.get("mode", "adaptive") == "adaptive" else "fixed")
    print(f"  {label(row['dwell']):>7} {row['mode']:8s} frac={row['distance_fraction']:<6} "
          f"free {row['free']:.4f}±{row['free_se']:.4f}  cell-change {row['cell_change']:.4f}  "
          f"MSD/t {row['msd_rate']:.3f}  steps/s {row['steps_per_s']:.3g}  ({row['wall_s']:.0f}s)", flush=True)
    return row


def demo_reference(t_max, seeds):
    """Cell-change fraction from the independent pure-Python demo at 1000 ms dwell."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import demo_cell_hopping as demo

    centers = demo.hex_centers(2)
    center = min(range(len(centers)), key=lambda k: np.hypot(*centers[k]))
    trips = changes = 0
    for s in seeds:
        rng = random.Random(s)
        arr = np.array(demo.simulate(centers, center, rng.uniform(0, 2 * np.pi), t_max, rng))
        trips += int(np.sum(np.diff(arr[:, 3]) == -1))
        changes += len(demo.find_hops(list(map(tuple, arr))))
    p = changes / trips
    print(f"  demo (pure Python, 1000 ms): cell-change {p:.4f} from {trips} trips", flush=True)
    return dict(dwell=1.0, mode="demo", check="demo", cell_change=p, cell_change_se=np.sqrt(p * (1 - p) / trips))


def plot(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    acc = df[df.check == "accuracy"]

    ax = axes[0, 0]
    for k, (dwell, g) in enumerate(acc.groupby("dwell")):
        g = g.sort_values("distance_fraction", ascending=False)
        x = np.arange(len(g)) + (k - 1) * 0.08
        ax.errorbar(x, g.free * 100, yerr=g.free_se * 100, fmt="o-", capsize=3, color=COLORS[label(dwell)], label=label(dwell))
    fracs = sorted(acc.distance_fraction.unique(), reverse=True)
    ax.set_xticks(range(len(fracs)), [f"{f}" + (" (default)" if f == 0.125 else "") for f in fracs])
    ax.axhline(5, color="k", ls="--", lw=1, label="target 5%")
    ax.set_xlabel("distance_fraction (smaller = finer steps)")
    ax.set_ylabel("time spent free (%)")
    ax.set_title("A. Free fraction: within ~2% of 5% at the default (0.125);\n0.25 is ~3% low at 5 ms")
    ax.legend()

    ax = axes[0, 1]
    default = df[(df.check == "accuracy") & (df.distance_fraction == 0.125)].assign(series="adaptive")
    fixed = df[df.check == "fixed"].assign(series="fixed")
    demo = df[df.check == "demo"].assign(series="demo (pure Python)")
    for k, (name, g, marker) in enumerate([("adaptive", default, "o"), ("fixed", fixed, "s"), ("demo (pure Python)", demo, "^")]):
        for _, row in g.iterrows():
            x = [0.005, 0.0707, 1.0].index(round(row.dwell, 4)) + (k - 1) * 0.12
            ax.errorbar(x, row.cell_change, yerr=row.cell_change_se, fmt=marker, capsize=3, color=f"C{k}",
                        label=name if _ == g.index[0] else None)
    ax.set_yscale("log")
    ax.set_xticks(range(3), ["5 ms", "71 ms", "1000 ms"])
    ax.set_xlabel("mean bound dwell")
    ax.set_ylabel("fraction of free trips ending on a new cell")
    ax.set_title("B. Cell-to-cell hopping: three implementations agree within ~1.5 SE")
    ax.legend()

    ax = axes[1, 0]
    rows = []
    for dwell in sorted(fixed.dwell.unique()):
        a = df[(df.check == "adaptive_vs_fixed") & (df.dwell == dwell)].iloc[0]
        f = fixed[fixed.dwell == dwell].iloc[0]
        for metric in ("free", "msd_rate"):
            ratio = a[metric] / f[metric]
            se = ratio * np.hypot(a[f"{metric}_se"] / a[metric], f[f"{metric}_se"] / f[metric])
            rows.append((dwell, metric, ratio, se))
    for k, metric in enumerate(("free", "msd_rate")):
        sub = [r for r in rows if r[1] == metric]
        x = np.arange(len(sub)) + (k - 0.5) * 0.15
        ax.errorbar(x, [r[2] for r in sub], yerr=[r[3] for r in sub], fmt="o", capsize=3, color=f"C{k + 3}",
                    label={"free": "free fraction", "msd_rate": "spreading (MSD per time)"}[metric])
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(sub)), [label(r[0]) for r in sub])
    ax.set_ylabel("adaptive / fixed")
    ax.set_title("C. Adaptive / fixed (1 = agreement): consistent, but fixed runs are short,\nso this is a weak test until larger fixed runs (cluster)")
    ax.legend()

    ax = axes[1, 1]
    dw = sorted(fixed.dwell.unique())
    a_cost = [df[(df.check == "adaptive_vs_fixed") & (df.dwell == d)].steps_per_s.iloc[0] for d in dw]
    f_cost = [fixed[fixed.dwell == d].steps_per_s.iloc[0] for d in dw]
    x = np.arange(len(dw))
    ax.bar(x - 0.2, a_cost, 0.4, label="adaptive", color="C0")
    ax.bar(x + 0.2, f_cost, 0.4, label="fixed", color="C1")
    for xi, a_, f_ in zip(x, a_cost, f_cost):
        ax.text(xi, max(a_, f_) * 1.3, f"{f_ / a_:.0f}× fewer steps", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x, [label(d) for d in dw])
    ax.set_ylabel("steps per molecule per simulated second")
    ax.set_title("D. Cost")
    ax.legend(loc="lower left")

    fig.suptitle("2D kernel checks: free/bound switching with 10 nm binding layer (no receptors, no escape)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--replot", action="store_true", help="redraw the figure from the saved CSV")
    args = parser.parse_args()
    out_csv = PROJECT_ROOT / "results" / "checks" / "kernel2d_checks.csv"
    out_png = PROJECT_ROOT / "results" / "figures" / "kernel2d_checks.png"
    if args.replot:
        plot(pd.read_csv(out_csv), out_png)
        print(f"Saved {out_png}")
        return
    cfg = load_config(CONFIG)
    q = 0.1 if args.quick else 1.0
    rows = []

    print("A. accuracy sweep (adaptive)")
    idx = {d: cfg.find_point("dwell", bound_dwell=d) for d in (0.005, 0.0707, 1.0)}
    for i, t_max in [(idx[0.005], 50.0), (idx[0.0707], 50.0), (idx[1.0], 200.0)]:
        for frac in (0.25, 0.125, 0.0625):
            rows.append(run(cfg, i, int(1000 * q), t_max * q, distance_fraction=frac))

    print("C/D. adaptive vs fixed (same molecules and duration)")
    for i, n, t_max in [(idx[0.005], 200, 2.0), (idx[0.0707], 200, 5.0), (idx[1.0], 100, 10.0)]:
        r = run(cfg, i, max(int(n * q), 10), t_max * q)
        r["check"] = "adaptive_vs_fixed"
        rows.append(r)
        rows.append(run(cfg, i, max(int(n * q), 10), t_max * q, mode="fixed"))

    print("B. independent demo")
    rows.append(demo_reference(300.0 * q, seeds=[0, 1]))

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    plot(df, out_png)
    print(f"Saved {out_csv}\nSaved {out_png}")


if __name__ == "__main__":
    main()
