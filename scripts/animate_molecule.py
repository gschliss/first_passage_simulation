"""Animate one replayed molecule searching the 3D lattice (L103, L104).

    python scripts/replay_molecule.py <run>/runs_a2nm --chunk 0 --molecule 15 --interval 0.05
    python scripts/animate_molecule.py results/trajectories/<name> --out results/figures/<name>.mp4

Takes the `.npy`/`.json` pair that `replay_molecule.py` writes.

The shot opens wide on the packed lattice and then eases in on the molecule while the camera
orbits, so the journey reads at both scales: the cluster is 323 cells at 86% packing and from
outside it is a solid block, but a molecule alone in an empty frame loses the point that it is
squeezing through a crowd.

Cells are drawn as sphere skeletons (a few great circles), not shaded surfaces: matplotlib
does not depth-sort surfaces against lines, so a solid sphere swallows the very trajectory the
animation is about. All the distant ones go into a single Line3DCollection, which keeps ~100
cells cheap to redraw every frame.

Colour is reserved for the molecule's state — orange free in the gaps, blue bound to a
membrane, the report's validated pair — so the target cell is marked by line weight and a
label instead.
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
from scipy.ndimage import uniform_filter1d

from fpsim.config import Geometry
from fpsim.lattice import make_cluster

import vizstyle as vz

BOUND, FREE = vz.SWITCH, vz.REFERENCE
TRAIL = 260          # frames of bright trail behind the molecule
HOLD_S = 2.5         # seconds held on the capture frame at the end
EASE = (0.06, 0.55)  # fraction of the movie over which the zoom happens


def sphere_segments(centre, r, n_lon, n_lat, pts=49):
    """A sphere as great circles: ``n_lon`` meridians plus ``n_lat`` latitude rings."""
    out, a = [], np.linspace(0, 2 * np.pi, pts)
    for lon in np.linspace(0, np.pi, n_lon, endpoint=False):
        c, s = np.cos(lon), np.sin(lon)
        out.append(np.column_stack([r * np.cos(a) * c, r * np.cos(a) * s, r * np.sin(a)]) + centre)
    for lat in np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]:
        rr, z = r * np.cos(lat), r * np.sin(lat)
        out.append(np.column_stack([rr * np.cos(a), rr * np.sin(a), np.full_like(a, z)]) + centre)
    return out


def sphere_surface(ax, centre, r, color, alpha, n=18):
    """Shaded translucent sphere. matplotlib has no true depth sorting, so at any alpha high
    enough to read as a surface it will sometimes paint over the trajectory; keep it low."""
    u, v = np.mgrid[0:2 * np.pi:n * 1j, 0:np.pi:(n // 2 + 1) * 1j]
    return ax.plot_surface(centre[0] + r * np.cos(u) * np.sin(v),
                           centre[1] + r * np.sin(u) * np.sin(v),
                           centre[2] + r * np.cos(v),
                           color=color, alpha=alpha, linewidth=0, shade=True, zorder=0)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectory", type=Path, help="path without extension, as replay_molecule.py printed it")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--frames", type=int, default=900, help="target frame count (the path is subsampled)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--spin", type=float, default=-60.0,
                   help="degrees of azimuth drift over the movie; sign sets the direction")
    p.add_argument("--context", type=float, default=2.2,
                   help="draw neighbouring cells within this many lattice spacings of the path")
    p.add_argument("--narrow", type=float, default=22.0, help="half-width (um) of the final close view")
    p.add_argument("--cells", choices=("skeleton", "surface", "crossfade"), default="crossfade",
                   help="great-circle skeletons, translucent shaded spheres, or spheres while the "
                        "shot is wide fading to skeletons as it pushes in (the default)")
    p.add_argument("--alpha", type=float, default=0.13, help="--cells surface: opacity of a cell")
    p.add_argument("--no-face", dest="face", action="store_false",
                   help="orbit at a fixed rate instead of keeping the molecule on the near side")
    args = p.parse_args()

    base = args.trajectory.parent / args.trajectory.name.removesuffix(".npy").removesuffix(".json")
    traj = np.load(base.parent / f"{base.name}.npy")
    scene = json.loads((base.parent / f"{base.name}.json").read_text())
    row, geo = scene["molecule_row"], dict(scene["geometry"])
    # `spacing` is a derived property, so keep only the real fields. Trajectory columns are
    # (dim + 2), which pins the dimension even for a scene file written before it was stored.
    keep = {f.name for f in dataclasses.fields(Geometry)}
    geo["dim"] = traj.shape[1] - 2
    g = Geometry(**{k: v for k, v in geo.items() if k in keep})
    if g.dim != 3:
        raise SystemExit(f"this animation is 3D only; trajectory is {g.dim}D")
    centres, radius = make_cluster(g).centers, g.cell_radius

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

    # Cells: everything nearby as faint skeletons, the visited ones firmer, the target firmest.
    hub = xyz.mean(0)
    near = [i for i in np.flatnonzero(np.linalg.norm(centres - hub, axis=1) <= args.context * g.spacing)
            if i not in visited]
    # Surfaces read as packed tissue while the shot is wide, but matplotlib does not
    # depth-sort them against lines and the close view sits *inside* the field, where they
    # paint over the trajectory. So the default shows spheres wide and skeletons close.
    surfs, skels = [], []
    if args.cells in ("surface", "crossfade"):
        surfs += [sphere_surface(ax, centres[c], radius, "#b9c4d2", args.alpha) for c in near]
        surfs += [sphere_surface(ax, centres[c], radius, "#aab6c6",
                                 args.alpha * (1.7 if c == final_cell else 1.3)) for c in visited]
    if args.cells in ("skeleton", "crossfade"):
        ctx = [s for c in near for s in sphere_segments(centres[c], radius, 3, 1)]
        vis = [s for c in visited if c != final_cell for s in sphere_segments(centres[c], radius, 6, 3)]
        tgt = sphere_segments(centres[final_cell], radius, 8, 5)
        for segments, colour, lw in ((ctx, "#ebebe7", 0.45), (vis, "#dcdcd8", 0.5),
                                     (tgt, "#a6a69f", 0.7)):
            if segments:
                lc = Line3DCollection(segments, colors=colour, linewidths=lw, zorder=0)
                ax.add_collection3d(lc)
                skels.append(lc)
    tc = centres[final_cell]
    label = ax.text(tc[0], tc[1], tc[2] + radius * 1.14, "target cell (ring 2)",
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
    dot = ax.scatter(*xyz[0], s=110, color=FREE, edgecolors=vz.SURFACE, linewidths=1.4,
                     zorder=7, depthshade=False)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1), zoom=1.4)

    # State strip: the whole history at a glance, with a cursor.
    s_full = traj[:, 3]
    bar.set_xlim(0, t[-1]); bar.set_ylim(0, 1); bar.set_yticks([])
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
             f"reached a target cell at {row['t_first_target']:.1f} s, captured "
             f"{row['t_end'] - row['t_first_target']:.1f} s later",
             ha="center", fontsize=9.5, color=vz.SECONDARY)
    ax.legend(handles=[Line2D([], [], color=FREE, lw=3, label="free in the gaps (3D)"),
                       Line2D([], [], color=BOUND, lw=3, label="bound to a membrane (2D)"),
                       Line2D([], [], color=vz.INK, lw=0, marker="X", label="start"),
                       Line2D([], [], color=vz.INK, lw=0, marker="*", label="receptor found")],
              loc="upper left", fontsize=9.5, bbox_to_anchor=(0.02, 0.96))

    hold = int(HOLD_S * args.fps)
    n = len(f) + hold
    # The camera follows a heavily smoothed position; the raw walk is far too jittery to track.
    look = uniform_filter1d(xyz, max(5, len(f) // 6), axis=0, mode="nearest")
    wide_span = args.context * g.spacing + radius
    wide_at = centres[near].mean(0) if len(near) else hub

    # Which side of its cell the molecule is on. It is bound ~90% of the time, so a fixed
    # orbit spends much of the movie with the cell body between the molecule and the camera.
    # Steering the azimuth to the outward normal keeps it on the near side; `--spin` is then
    # an extra drift on top, so the field still turns.
    on_cell = np.where(np.isfinite(f[:, 4]), f[:, 4], 0).astype(int)
    outward = xyz - centres[on_cell]
    phi = np.unwrap(np.arctan2(outward[:, 1], outward[:, 0]))
    face = np.degrees(uniform_filter1d(phi, max(5, len(f) // 8), mode="nearest"))

    def segs(a):
        return np.stack([a[:-1], a[1:]], axis=1) if len(a) > 1 else np.empty((0, 2, 3))

    def update(k):
        i = min(k, len(f) - 1)
        faint.set_segments(segs(xyz[: i + 1]))
        s = segs(xyz[max(0, i - TRAIL): i + 1])
        trail.set_segments(s)
        if len(s):
            rgb = [np.array(plt.matplotlib.colors.to_rgb(x)) for x in (BOUND, FREE)]
            c = np.where(state[max(0, i - TRAIL): i][:, None] == 1, *rgb)
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

        u = smoothstep((k / n - EASE[0]) / (EASE[1] - EASE[0]))
        span = wide_span + (args.narrow - wide_span) * u
        mid = wide_at * (1 - u) + look[i] * u
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        label.set_alpha(float(u))          # unreadable while the field is wide
        if args.cells == "crossfade":      # spheres while wide, skeletons once close
            fade = smoothstep(u / 0.55)
            for a in surfs:
                a.set_alpha(args.alpha * (1 - fade))
            for lc in skels:
                lc.set_alpha(fade)
        elev = 28 - 12 * u
        azim = (face[i] if args.face else -60.0) + args.spin * k / n
        ax.view_init(elev=elev, azim=azim)
        return ()

    out = args.out or Path("results/figures") / f"{base.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / args.fps, blit=False)
    # Quality-targeted rather than bitrate-targeted: this is line art on white, which CRF
    # compresses far better than a fixed bitrate. yuv420p so it plays in browsers and QuickTime.
    writer = animation.FFMpegWriter(fps=args.fps, codec="libx264",
                                    # yuv420p needs even dimensions and figsize*dpi often is
                                    # not (9x8.6 at dpi 95 gives 855x817), so pad by a pixel.
                                    extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:0xFCFCFB",
                                                "-crf", "24", "-preset", "slow",
                                                "-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"Saved {out}  ({n} frames, {n / args.fps:.0f} s, {step}x subsampled from {len(traj)}; "
          f"{len(near)} context cells, spin {args.spin:g}°)")


if __name__ == "__main__":
    main()
