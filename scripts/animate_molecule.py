"""Animate one replayed molecule searching the 3D lattice (L103).

    python scripts/replay_molecule.py <run>/runs_a2nm --chunk 0 --molecule 15 --interval 0.05
    python scripts/animate_molecule.py results/trajectories/<name> --out results/figures/<name>.mp4

Takes the `.npy`/`.json` pair that `replay_molecule.py` writes. Only the cells the molecule
actually touches are drawn — the full cluster is 323 spheres at 86% packing, which from
outside is just a solid block.

Colour follows the report's validated pair (`vizstyle`): the molecule is orange while
diffusing freely in the gaps, blue while bound to a membrane. That matches the report's
sense, where orange is the free-only walker and blue the switching one. The legend says so
outright, so nothing rests on colour alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from fpsim.config import Geometry
from fpsim.lattice import make_cluster

import vizstyle as vz

BOUND, FREE = vz.SWITCH, vz.REFERENCE
TRAIL = 260          # frames of bright trail behind the molecule
HOLD_S = 2.0         # seconds held on the capture frame at the end


def sphere(ax, centre, r, color, lw, n=13):
    """A wireframe, not a surface: matplotlib does not depth-sort surfaces against lines, so a
    shaded sphere swallows the very trajectory the animation is about."""
    u, v = np.mgrid[0:2 * np.pi:n * 1j, 0:np.pi:(n // 2 + 1) * 1j]
    ax.plot_wireframe(centre[0] + r * np.cos(u) * np.sin(v),
                      centre[1] + r * np.sin(u) * np.sin(v),
                      centre[2] + r * np.cos(v),
                      color=color, linewidth=lw, zorder=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectory", type=Path, help="path without extension, as replay_molecule.py printed it")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--frames", type=int, default=900, help="target frame count (the path is subsampled)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    base = args.trajectory.parent / args.trajectory.name.removesuffix(".npy").removesuffix(".json")
    traj = np.load(base.parent / f"{base.name}.npy")
    scene = json.loads((base.parent / f"{base.name}.json").read_text())
    row, geo = scene["molecule_row"], scene["geometry"]
    # `spacing` is a derived property, so keep only the real fields. Trajectory columns are
    # (dim + 2), which pins the dimension even for a scene file written before it was stored.
    keep = {f.name for f in dataclasses.fields(Geometry)}
    geo = dict(geo, dim=traj.shape[1] - 2)
    centres = make_cluster(Geometry(**{k: v for k, v in geo.items() if k in keep})).centers
    if geo["dim"] != 3:
        raise SystemExit(f"this animation is 3D only; trajectory is {geo['dim']}D")

    step = max(1, len(traj) // args.frames)
    f = traj[::step]
    t = np.arange(len(traj))[::step] * scene["interval"]
    xyz, state = f[:, :3], f[:, 3]
    visited = sorted({int(c) for c in traj[:, 4] if np.isfinite(c)})
    final_cell = int(row["final_cell"])

    vz.use()
    fig = plt.figure(figsize=(9, 8.6))
    ax = fig.add_axes([0.0, 0.14, 1.0, 0.78], projection="3d")
    bar = fig.add_axes([0.10, 0.065, 0.80, 0.022])

    # Hue is reserved for the molecule's state, so the target cell is marked by weight and a
    # label rather than by colour.
    for c in visited:
        is_target = c == final_cell
        sphere(ax, centres[c], geo["cell_radius"], "#a6a69f" if is_target else "#e0e0dc",
               0.7 if is_target else 0.45)
    tc = centres[final_cell]
    ax.text(tc[0], tc[1], tc[2] + geo["cell_radius"] * 1.12, "target cell (ring 2)",
            ha="center", fontsize=9.5, color=vz.SECONDARY, zorder=8)
    ax.scatter(*xyz[0], s=70, color=vz.INK, marker="X", zorder=5, depthshade=False)
    # Halo behind the star: at capture the molecule is buried in its own local-search trail.
    halo = ax.scatter(*traj[-1, :3], s=0, color=vz.SURFACE, marker="o", zorder=6, depthshade=False)
    cap = ax.scatter(*traj[-1, :3], s=0, color=vz.INK, marker="*", zorder=7, depthshade=False)

    # add_collection3d needs at least one segment, so seed both with a degenerate one.
    seed = [np.stack([xyz[0], xyz[0]])]
    faint = Line3DCollection(seed, colors="#b9b9b3", linewidths=1.0, zorder=3)
    trail = Line3DCollection(seed, linewidths=2.2, zorder=4)
    ax.add_collection3d(faint)
    ax.add_collection3d(trail)
    dot = ax.scatter(*xyz[0], s=95, color=FREE, edgecolors=vz.SURFACE, linewidths=1.4,
                     zorder=7, depthshade=False)

    lo, hi = np.nanmin(xyz, 0), np.nanmax(xyz, 0)
    pad = 3.0
    mid, span = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    for setter, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
        setter(m - span, m + span)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    # State strip: the whole history at a glance, with a cursor.
    bar.set_xlim(0, t[-1]); bar.set_ylim(0, 1); bar.set_yticks([])
    s_full = traj[:, 3]
    bar.imshow(s_full[None, :], aspect="auto", extent=(0, len(traj) * scene["interval"], 0, 1),
               cmap=plt.matplotlib.colors.ListedColormap([FREE, BOUND]), vmin=0, vmax=1,
               interpolation="nearest")
    cursor = bar.axvline(0, color=vz.INK, lw=1.4)
    bar.set_xlabel("time (s)", fontsize=9.5)
    for sp in bar.spines.values():
        sp.set_visible(False)
    bar.tick_params(length=0, labelsize=9)

    pct_bound = float((s_full == 1).mean())
    fig.text(0.5, 0.965, "One molecule finding a receptor", ha="center", fontsize=15, color=vz.INK)
    fig.text(0.5, 0.935,
             f"molecule {scene['molecule']} of {scene['run']}  ·  captured at "
             f"{row['t_end']:.1f} s, which is this run's t_20",
             ha="center", fontsize=10, color=vz.SECONDARY)
    clock = fig.text(0.5, 0.905, "", ha="center", fontsize=12, color=vz.INK, family="monospace")
    fig.text(0.5, 0.012,
             f"bound {pct_bound:.0%} of the time · {int(row['n_trips'])} unbinding events · "
             f"reached a target cell at {row['t_first_target']:.1f} s, captured {row['t_end'] - row['t_first_target']:.1f} s later",
             ha="center", fontsize=9.5, color=vz.SECONDARY)
    ax.legend(handles=[Line2D([], [], color=FREE, lw=3, label="free in the gaps (3D)"),
                       Line2D([], [], color=BOUND, lw=3, label="bound to a membrane (2D)"),
                       Line2D([], [], color=vz.INK, lw=0, marker="X", label="start"),
                       Line2D([], [], color=vz.INK, lw=0, marker="*", label="receptor found")],
              loc="upper left", fontsize=9.5, bbox_to_anchor=(0.02, 0.96))

    hold = int(HOLD_S * args.fps)
    n = len(f) + hold

    def segs(a):
        return np.stack([a[:-1], a[1:]], axis=1) if len(a) > 1 else np.empty((0, 2, 3))

    def update(k):
        i = min(k, len(f) - 1)
        faint.set_segments(segs(xyz[: i + 1]))
        s = segs(xyz[max(0, i - TRAIL): i + 1])
        trail.set_segments(s)
        if len(s):
            c = np.where(state[max(0, i - TRAIL): i][:, None] == 1, *[np.array(plt.matplotlib.colors.to_rgb(x)) for x in (BOUND, FREE)])
            a = np.linspace(0.15, 1.0, len(s))[:, None]
            trail.set_color(np.hstack([c, a]))
        dot._offsets3d = ([xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]])
        dot.set_color(BOUND if state[i] == 1 else FREE)
        cursor.set_xdata([t[i], t[i]])
        clock.set_text(f"t = {t[i]:7.1f} s")
        if k >= len(f) - 1:
            halo.set_sizes([560])
            cap.set_sizes([300])
            clock.set_text(f"t = {row['t_end']:7.1f} s   captured")
        ax.view_init(elev=20, azim=-60 + 28 * k / n)
        return ()

    out = args.out or Path("results/figures") / f"{base.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / args.fps, blit=False)
    anim.save(out, writer="ffmpeg", fps=args.fps, dpi=args.dpi)
    print(f"Saved {out}  ({n} frames, {n / args.fps:.0f} s, {step}x subsampled from {len(traj)})")


if __name__ == "__main__":
    main()
