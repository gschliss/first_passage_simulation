"""Adaptive vs fixed time steps on capture and escape (L31), small test geometry.

Uses scaled-down cells (2 µm radius, 1 receptor per cell, ring-1 targets) so captures
and escapes both happen within seconds; the physics rules are the same as the model.
Compares cumulative capture and escape over time for:
  adaptive (default, distance_fraction 0.125), adaptive halved (0.0625), fixed steps.

Writes results/checks/capture2d_*.parquet and results/figures/capture2d_checks.png.

Usage:
    python scripts/check_capture_2d.py           # ~15 min on 8 cores
    python scripts/check_capture_2d.py --replot
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim.config import Stepping, from_dict
from fpsim.io import load_result, save_result
from fpsim.simulate import simulate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "checks"
SMALL = from_dict({
    "name": "small_check",
    "geometry": {"cell_radius": 2.0, "gap": 0.5, "target_ring": 1, "rings_beyond_target": 1},
    "receptors": {"n_per_cell": 1},
})
VARIANTS = {
    "adaptive (0.125)": dict(mode="adaptive", distance_fraction=0.125),
    "adaptive halved (0.0625)": dict(mode="adaptive", distance_fraction=0.0625),
    "fixed": dict(mode="fixed", distance_fraction=0.125),
}
POINTS = [(SMALL.find_point("dwell", bound_dwell=0.005), "5 ms"), (SMALL.find_point("dwell", bound_dwell=1.0), "1000 ms")]
N, T_MAX = 200, 20.0


def stem(i, name):
    return OUT / f"capture2d_sweep{i}_{name.split()[0]}{'_half' if 'halved' in name else ''}"


def run_all():
    for i, label in POINTS:
        for name, st in VARIANTS.items():
            cfg = dataclasses.replace(SMALL, stepping=Stepping(**st))
            r = simulate(cfg, i, n_molecules=N, t_max=T_MAX)
            save_result(r, stem(i, name))
            c = r.molecules.outcome.value_counts()
            print(f"{label:>8} {name:26s} captured {c['captured']:4d} escaped {c['escaped']:4d} "
                  f"timeout {c['timeout']:4d}  wall {r.meta['wall_time_s']:.0f}s", flush=True)


def cumulative(m, outcome, t):
    times = np.sort(m.t_end[m.outcome == outcome].to_numpy())
    return np.searchsorted(times, t, side="right") / len(m)


def plot(out):
    t = np.linspace(0, T_MAX, 400)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (i, label) in zip(axes[:2], POINTS):
        for k, name in enumerate(VARIANTS):
            m = load_result(stem(i, name)).molecules
            for outcome, ls in (("captured", "-"), ("escaped", "--")):
                f = cumulative(m, outcome, t)
                ax.plot(t, f, ls, color=f"C{k}", lw=1.8, label=f"{name}, {outcome}")
                se = np.sqrt(f * (1 - f) / len(m))
                ax.fill_between(t, f - se, f + se, color=f"C{k}", alpha=0.12, lw=0)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("fraction of molecules")
        ax.set_title(f"{label} dwell: cumulative capture (solid) and escape (dashed)")
        ax.legend(fontsize=7.5)
    ax = axes[2]
    rows = []
    for i, label in POINTS:
        for name in VARIANTS:
            m = load_result(stem(i, name)).molecules
            cap = m[m.outcome == "captured"]
            rows.append(dict(point=label, variant=name, captured=len(cap) / len(m),
                             captured_se=np.sqrt(len(cap) / len(m) * (1 - len(cap) / len(m)) / len(m)),
                             from_bound=(cap.final_state == "bound").mean() if len(cap) else np.nan))
    df = pd.DataFrame(rows)
    for k, name in enumerate(VARIANTS):
        d = df[df.variant == name]
        x = np.arange(len(POINTS)) + (k - 1) * 0.2
        ax.errorbar(x, d.captured, yerr=d.captured_se, fmt="o", color=f"C{k}", capsize=3, label=f"{name}: captured by {T_MAX:.0f} s")
        ax.plot(x, d.from_bound, "x", color=f"C{k}", ms=9, label=f"{name}: share captured while bound")
    ax.set_xticks(range(len(POINTS)), [p[1] for p in POINTS])
    ax.set_ylim(0, 1)
    ax.set_title("Totals (error bars: ±1 SE)")
    ax.legend(fontsize=7)
    fig.suptitle(f"Capture/escape: adaptive vs halved vs fixed steps, small test geometry "
                 f"(2 µm cells, 1 receptor/cell, ring-1 targets), {N} molecules each", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    if not args.replot:
        run_all()
    out = PROJECT_ROOT / "results" / "figures" / "capture2d_checks.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
