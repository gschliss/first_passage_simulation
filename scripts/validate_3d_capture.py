"""Validate 3D receptor capture at 2 nm against theory and step size (L68, L69).

  1. Free capture vs Berg–Purcell: free walkers around one isolated cell (R = 10 µm,
     1000 Fibonacci receptors) start 50 nm off the membrane and are lost at 2 µm;
     capture probability vs the Berg–Purcell prediction, at 2 nm and 10 nm.
  2. Free capture vs step size: start within 50 nm of a receptor, 5 nm off the membrane,
     lost at 100 nm (small, so fixed tiny steps are affordable): default adaptive,
     2× and 4× finer, and fixed steps.
  3. Bound search vs theory and step size: molecules bound to the cell that never
     unbind; mean time to reach a receptor vs the narrow-escape cell approximation,
     adaptive (default, 2× finer) and fixed steps.
  4. The real 3D lattice run (L68): the free-only reference at 100 ms rerun with 2×
     finer steps on the same 10 layouts and molecule seeds; local search must not change.

Usage:
    NUMBA_NUM_THREADS=6 python scripts/validate_3d_capture.py
    python scripts/validate_3d_capture.py --replot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim import load_config
from fpsim.config import Receptors, Stepping, Sweep
from fpsim.jobs import Task, merge, run_task
from fpsim.receptors import fibonacci_sphere, make_grid
from fpsim.validation import (bound_isolated, bp_capture_probability, free_isolated, narrow_escape_time,
                              perfect_capture_probability)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "validation" / "capture3d"
R, N, LAYER, D_FREE, D_BOUND, DT_CAP = 10.0, 1000, 0.010, 1.1, 0.5, 0.01
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e6e6e6"
L68_RUN = PROJECT_ROOT / "results" / "dimension" / "lattice3d_ring2_rec1000_dw50-100-300_paired10"


def ci(p, n):
    h = 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n)
    return p - h, p + h


def free(a, r0_off, lost_at, n, frac=0.125, fixed=False, seed0=0, rho=0.0):
    base = fibonacci_sphere(N)
    g = make_grid(base)
    dt_fixed = (0.125 * a) ** 2 / (2 * D_FREE) / 4  # 4x finer than the finest adaptive floor at the default
    seeds = np.arange(seed0, seed0 + n, dtype=np.int64)
    t0 = time.time()
    c, t, ns = free_isolated(seeds, R, R + lost_at, R + r0_off, D_FREE, LAYER, a, base, g.start, g.index, g.n, g.box,
                             fixed, dt_fixed, frac, DT_CAP, rho)
    return dict(a_nm=a * 1e3, r0_off=r0_off, lost_at=lost_at, rho=rho, n=n, method="fixed" if fixed else f"adaptive {frac:g}",
                u=c.mean(), steps=ns.mean(), wall=time.time() - t0)


def bound(a, n, frac=0.125, fixed=False):
    base = fibonacci_sphere(N)
    g = make_grid(base)
    dt_fixed = (0.125 * a) ** 2 / (2 * D_BOUND)
    seeds = np.arange(n, dtype=np.int64)
    t0 = time.time()
    c, t, ns = bound_isolated(seeds, R, D_BOUND, a, base, g.start, g.index, g.n, g.box, fixed, dt_fixed, frac, DT_CAP,
                              500.0)
    return dict(a_nm=a * 1e3, n=n, method="fixed" if fixed else f"adaptive {frac:g}", captured=c.mean(),
                T=t.mean(), T_se=t.std(ddof=1) / np.sqrt(n), steps=ns.mean(), wall=time.time() - t0)


def lattice_check(n_layouts=10, per_layout=20):
    """Rerun L68's free-only 100 ms reference with 2x finer steps on the same layouts and seeds."""
    ref = json.loads((L68_RUN / "reference.json").read_text())
    cfg = load_config(PROJECT_ROOT / "configs" / "lattice3d.yaml")
    dwells = (0.05, 0.1, 0.3)
    base_cfg = dataclasses.replace(
        cfg, name=cfg.name + "_ring2_rec1000_dw50-100-300_paired10",
        geometry=dataclasses.replace(cfg.geometry, target_ring=2),
        receptors=Receptors(n_per_cell=1000, capture_radius=0.002),
        sweeps=(Sweep("switch", free_fraction=0.05, bound_dwell=dwells),
                Sweep("reference", free_fraction=1.0, d_free=tuple(ref["d_free_reference"]))))
    fine = dataclasses.replace(base_cfg, name=base_cfg.name + "_fine",
                               stepping=dataclasses.replace(base_cfg.stepping, distance_fraction=0.0625))
    run_dir = OUT / "lattice_fine"
    idx = len(dwells) + 1  # reference point paired with the 100 ms switching walker
    for c in range(n_layouts):
        run_task(fine, Task(idx, c), per_layout, run_dir, layout_seed=c)
        print(f"   layout {c} done", flush=True)
    rows = []
    for name, d in (("adaptive 0.125 (L68 run)", L68_RUN / "runs_a2nm"), ("adaptive 0.0625", run_dir)):
        m = merge(d)[0]
        m = m[m.sweep_index == idx]
        cap = m[m.outcome == "captured"]
        loc = cap.t_end - cap.t_first_target
        rows.append(dict(method=name, n=len(m), captured=len(cap) / len(m), t_mean=cap.t_end.mean(),
                         t_se=cap.t_end.std(ddof=1) / np.sqrt(len(cap)), local=loc.mean(),
                         local_se=loc.std(ddof=1) / np.sqrt(len(cap)), transport=cap.t_first_target.mean()))
    return pd.DataFrame(rows)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot(c1, c2, c3, c4, out):
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    x = np.arange(len(c1))
    lo, hi = zip(*[ci(r.u, r.n) for r in c1.itertuples()])
    ax.errorbar(x - 0.1, c1.u, yerr=[c1.u - lo, np.array(hi) - c1.u], fmt="o", color=BLUE, ms=9, capsize=4,
                label="simulation (95% CI)")
    ax.plot(x + 0.1, c1.bp, "D", color=INK, ms=9, label="Berg–Purcell theory")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{a:g} nm target" for a in c1.a_nm])
    ax.set_ylabel("capture probability before loss (log)")
    ax.set_title("1. Free walkers around one cell: simulation vs Berg–Purcell\n"
                 f"start 50 nm off the membrane, lost at 2 µm (fully absorbing cell: {c1.perfect.iloc[0]:.2f})",
                 loc="left", fontsize=10)
    ax.legend(frameon=False)
    style(ax)

    ax = axes[0, 1]
    x = np.arange(len(c2))
    lo, hi = zip(*[ci(r.u, r.n) for r in c2.itertuples()])
    ax.errorbar(x, c2.u, yerr=[c2.u - lo, np.array(hi) - c2.u], fmt="o", color=BLUE, ms=9, capsize=4)
    ax.axhline(c2.u.iloc[-1], color=MUTED, ls="--", lw=1, label="fixed tiny steps")
    ax.set_xticks(x, c2.method)
    ax.set_ylim(0, max(hi) * 1.3)
    ax.set_ylabel("capture probability before loss (95% CI)")
    ax.set_title("2. Free capture vs step size (2 nm; start within 50 nm of a receptor,\n5 nm off the membrane; lost at 100 nm)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    style(ax)

    ax = axes[1, 0]
    x = np.arange(len(c3))
    ax.errorbar(x, c3.T, yerr=1.96 * c3.T_se, fmt="o", color=BLUE, ms=9, capsize=4, label="simulation (95% CI)")
    for k, (a, theory) in enumerate(c3.groupby("a_nm").theory.first().items()):
        sel = np.flatnonzero(c3.a_nm == a)
        ax.hlines(theory, sel.min() - 0.3, sel.max() + 0.3, color=INK, lw=1.5, label="narrow-escape theory" if k == 0 else None)
    ax.set_xticks(x, [f"{r.a_nm:g} nm\n{r.method}" for r in c3.itertuples()], fontsize=9)
    ax.set_ylabel("mean time to reach a receptor while bound (s)")
    ax.set_title("3. Bound walkers on one cell (never unbind): simulation vs theory", loc="left", fontsize=10)
    ax.legend(frameon=False)
    style(ax)

    ax = axes[1, 1]
    if c4 is not None:
        x = np.arange(2)
        for off, col, se, color, label in ((-0.12, "local", "local_se", ORANGE, "local search"),
                                           (0.12, "t_mean", "t_se", INK, "total capture time")):
            ax.errorbar(x + off, c4[col], yerr=1.96 * c4[se], fmt="o", color=color, ms=9, capsize=4, label=label)
        ax.set_xticks(x, c4.method)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("mean time (s, 95% CI)")
        ax.legend(frameon=False)
    ax.set_title("4. Real 3D lattice run: free-only reference at 100 ms,\nsame layouts and seeds, 2× finer steps", loc="left", fontsize=10)
    style(ax)

    fig.suptitle("Validation of 3D capture at 2 nm (free-only walker and bound walker)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replot", action="store_true")
    p.add_argument("--skip-lattice", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.replot:
        print("1. free capture vs Berg–Purcell", flush=True)
        rows = []
        for a, n in ((0.002, 20000), (0.010, 10000)):
            r = free(a, 0.05, 2.0, n)
            r.update(bp=bp_capture_probability(R, R + 2.0, R + 0.05, N, a), perfect=perfect_capture_probability(R, R + 2.0, R + 0.05))
            rows.append(r)
            print(f"   {r}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "check1.csv", index=False)

        print("2. free capture vs step size", flush=True)
        rows = [free(0.002, 0.005, 0.1, 20000, frac=f, rho=0.05) for f in (0.125, 0.0625, 0.03125)]
        rows.append(free(0.002, 0.005, 0.1, 4000, fixed=True, rho=0.05))
        for r in rows:
            print(f"   {r}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "check2.csv", index=False)

        print("3. bound search vs theory and step size", flush=True)
        rows = []
        for a in (0.002, 0.010):
            for kw in (dict(frac=0.125), dict(frac=0.0625), dict(fixed=True)):
                r = bound(a, 2000 if not kw.get("fixed") else 400, **kw)
                r["theory"] = narrow_escape_time(R, N, a, D_BOUND)
                rows.append(r)
                print(f"   {r}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "check3.csv", index=False)

        if not args.skip_lattice:
            print("4. real lattice run, 2x finer steps", flush=True)
            lattice_check().to_csv(OUT / "check4.csv", index=False)

    c4 = pd.read_csv(OUT / "check4.csv") if (OUT / "check4.csv").exists() else None
    for name in ("check1", "check2", "check3", "check4"):
        if (OUT / f"{name}.csv").exists():
            print(f"\n{name}:\n" + pd.read_csv(OUT / f"{name}.csv").to_string(index=False))
    out = PROJECT_ROOT / "results" / "figures" / "validation_capture3d.png"
    plot(pd.read_csv(OUT / "check1.csv"), pd.read_csv(OUT / "check2.csv"), pd.read_csv(OUT / "check3.csv"), c4, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
