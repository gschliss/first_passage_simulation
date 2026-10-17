"""Pilot run of the full model (2D or 3D): search times, capture vs escape, and cost.

Runs three points of the config's "dwell" sweep with a long t_max, records a few
trajectories for pictures, and saves everything under results/pilot/.

Usage:
    python scripts/pilot.py --config configs/lattice2d.yaml --n 200 --t-max 5000
    python scripts/pilot.py --config configs/lattice3d.yaml --n 200 --t-max 5000
    python scripts/pilot.py --config configs/lattice3d.yaml --replot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from fpsim import load_config, make_cluster
from fpsim.io import load_result, save_result
from fpsim.simulate import simulate
from fpsim.viz.cluster import draw_cells, escape_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "pilot"
PILOT_DWELLS = (0.005, 0.0707, 1.0)  # s; points of the "dwell" sweep used for the pilot (L40)


def pilot_points(cfg):
    return [cfg.find_point("dwell", bound_dwell=d) for d in PILOT_DWELLS]


def stem(cfg, i):
    return OUT / f"pilot_{cfg.name}_sweep{i}"


def run_all(cfg, n, t_max, n_traj):
    for i, dwell in zip(pilot_points(cfg), PILOT_DWELLS):
        r = simulate(cfg, i, n_molecules=n, t_max=t_max, n_trajectories=n_traj)
        save_result(r, stem(cfg, i))
        m = r.molecules
        c = m.outcome.value_counts()
        molecule_s = m.t_end.sum()
        print(f"dwell {dwell * 1e3:6.0f} ms: captured {c['captured']}, escaped {c['escaped']}, timeout {c['timeout']} | "
              f"steps per molecule-s {m.n_steps.sum() / molecule_s:.3g} | wall {r.meta['wall_time_s']:.0f}s "
              f"for {molecule_s:.3g} molecule-s", flush=True)


def draw_background(ax, cfg, cluster, r, lost, X, Y):
    if cfg.geometry.dim == 2:
        ax.contourf(X, Y, lost.astype(float), levels=[0.5, 1.5], colors=["#fde0c5"])
        draw_cells(ax, cluster, cfg.geometry.target_ring, labels=False)
        cells = np.array(r.meta["receptor_cells"]); angles = np.array(r.meta["receptor_angles"])
        rec = cluster.centers[cells][:, None, :] + cluster.cell_radius * np.stack([np.cos(angles), np.sin(angles)], -1)
        ax.plot(rec[..., 0].ravel(), rec[..., 1].ravel(), "k.", ms=3)
    else:  # x-y projection: target and source cells as outlines
        c = cluster.centers
        for i in np.flatnonzero(cluster.ring == cfg.geometry.target_ring):
            ax.add_patch(plt.Circle(c[i, :2], cluster.cell_radius, fill=False, ec="#b22222", lw=0.5, alpha=0.5))
        ax.add_patch(plt.Circle(c[cluster.source, :2], cluster.cell_radius, fc="#a6c8f4", ec="#1f4e9c"))


def plot(cfg, out):
    cluster = make_cluster(cfg.geometry)
    results = [load_result(stem(cfg, i)) for i in pilot_points(cfg)]
    t_max = results[0].meta["t_max"]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    # Top: example paths at each sweep point
    extent = (cluster.rings + 1.6) * cluster.spacing
    X, Y = np.meshgrid(np.linspace(-extent, extent, 700), np.linspace(-extent, extent, 700))
    lost = escape_mask(cluster, X, Y) if cfg.geometry.dim == 2 else None
    for k, (r, dwell) in enumerate(zip(results, PILOT_DWELLS)):
        ax = fig.add_subplot(gs[0, k])
        draw_background(ax, cfg, cluster, r, lost, X, Y)
        m = r.molecules
        for j, tr in enumerate(r.trajectories):
            ok = np.isfinite(tr[:, 0])
            pts = tr[ok, :2]
            if len(pts) < 2:
                continue
            seg = np.stack([pts[:-1], pts[1:]], 1)
            col = plt.get_cmap("tab10")(j % 10)
            ax.add_collection(LineCollection(seg, colors=[col], linewidths=0.7, alpha=0.85))
            end = m.outcome.iloc[j]
            ax.plot(*pts[-1], {"captured": "*", "escaped": "X", "timeout": "o"}[end], ms=11 if end == "captured" else 8,
                    color=col, mec="k", zorder=5)
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent); ax.set_aspect("equal"); ax.axis("off")
        proj = "" if cfg.geometry.dim == 2 else " (x–y projection; red = ring-3 targets)"
        ax.set_title(f"bound dwell {dwell * 1e3:.0f} ms: {len(r.trajectories)} example molecules{proj}\n"
                     "★ captured, ✕ escaped, ● still searching", fontsize=10)

    # Bottom left: cumulative capture and escape
    ax = fig.add_subplot(gs[1, 0])
    t = np.logspace(-1, np.log10(t_max), 300)
    for k, (r, dwell) in enumerate(zip(results, PILOT_DWELLS)):
        m = r.molecules
        for outcome, ls in (("captured", "-"), ("escaped", "--")):
            times = np.sort(m.t_end[m.outcome == outcome].to_numpy())
            ax.plot(t, np.searchsorted(times, t, side="right") / len(m), ls, color=f"C{k}",
                    label=f"{dwell * 1e3:.0f} ms, {outcome}")
    ax.set_xscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fraction of molecules")
    ax.set_title("Cumulative capture (solid) and escape (dashed)")
    ax.legend(fontsize=8)

    # Bottom middle: outcome totals
    ax = fig.add_subplot(gs[1, 1])
    labels = [f"{d * 1e3:.0f} ms" for d in PILOT_DWELLS]
    bottoms = np.zeros(len(results))
    for outcome, color in (("captured", "#2ca02c"), ("escaped", "#d9480f"), ("timeout", "#999999")):
        vals = np.array([(r.molecules.outcome == outcome).mean() for r in results])
        ax.bar(labels, vals, bottom=bottoms, color=color, label=outcome)
        for x, (b, v) in enumerate(zip(bottoms, vals)):
            if v > 0.04:
                ax.text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", color="white", fontsize=10)
        bottoms += vals
    ax.set_ylabel(f"fraction of molecules by t = {t_max:.0f} s")
    ax.set_title("Outcomes")
    ax.legend(loc="upper right", fontsize=8)

    # Bottom right: cost
    ax = fig.add_subplot(gs[1, 2])
    rate = [r.molecules.n_steps.sum() / r.molecules.t_end.sum() for r in results]
    core_s = [r.meta["wall_time_s"] * 8 / r.molecules.t_end.sum() for r in results]
    ax.bar(np.arange(len(results)) - 0.2, rate, 0.4, label="steps per molecule-second", color="C0")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(results)) + 0.2, np.array(core_s) * 1e3, 0.4, label="core-ms per molecule-second", color="C1")
    ax.set_xticks(range(len(results)), labels)
    ax.set_ylabel("steps per molecule-second", color="C0")
    ax2.set_ylabel("core-milliseconds per molecule-second (8 cores)", color="C1")
    ax.set_title("Cost")

    n = len(results[0].molecules)
    fig.suptitle(f"Pilot, {cfg.geometry.dim}D model ({cfg.name}): {n} molecules per sweep point, t_max = {t_max:.0f} s", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "lattice2d.yaml")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--t-max", type=float, default=5000.0)
    parser.add_argument("--n-traj", type=int, default=6)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.replot:
        run_all(cfg, args.n, args.t_max, args.n_traj)
    out = PROJECT_ROOT / "results" / "figures" / f"pilot_{cfg.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(cfg, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
