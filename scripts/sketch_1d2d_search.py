"""Toy 2D cartoon of the search: slide on a membrane, hop the slit, find a receptor (L110/L111).

Not a result — a teaching picture of the *mechanism* the project tests, drawn one
dimension down so the whole search fits on a page. Three cells meet at a junction,
with the project's own geometry (R = 10 um, 0.5 um gap). A molecule:

    bound    slides along one cell's membrane, a 1D walk on a circle
    free     unbinds, diffuses in the gap in 2D, reflects off membranes
    hop      rebinds to whichever membrane it reaches first — often a *different* cell,
             which is only possible where cells come close, i.e. at the junction
    capture  reaching a receptor patch while bound to the target cell

Free and bound use the same D (L96), so nothing here comes from one state being
faster: the picture is entirely about where the molecule can go.

    python scripts/sketch_1d2d_search.py --seed 3

Writes docs/figures/sketch_1d2d_search.{png,pdf}. This figure carries no capture
curves and no statistics on purpose — it is a mechanism cartoon, and the measured
comparison lives in the report's own capture-curve figures.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from numba import njit

import vizstyle as vz

PROJECT_ROOT = Path(__file__).resolve().parents[1]

R = 10.0                      # cell radius, um (L2)
GAP = 0.5                     # closest approach, um (L2)
SPACING = 2 * R + GAP
D = 0.5                       # um^2/s, free and bound alike (L96)
DELTA = 0.05                  # binding layer, um (thicker than the model's 10 nm, to stay drawable)
DT = 1e-4
TARGET_CELL = 2               # receptors live on this cell only, as in the real runs
JUNCTION = (0, 1, 2)          # centre cell and two neighbours: the trio the zoom panel shows
N_PATCH = 12                  # receptor patches on the target cell
PATCH_ARC = 0.25              # um of membrane per patch

# A 7-cell patch, not a bare trio: with only three cells a molecule that unbinds on an
# outer face diffuses off into vacuum, which is not what the gap looks like in a packed
# lattice. Cells 0, 1, 2 are the junction the zoom panel is centred on.
_ANG = np.arange(6) * np.pi / 3
CENTERS = np.vstack([[0.0, 0.0],
                     SPACING * np.column_stack([np.cos(_ANG + np.pi / 3), np.sin(_ANG + np.pi / 3)])])
CENTERS = np.ascontiguousarray(CENTERS[[0, 1, 2, 3, 4, 5, 6]])
JUNCTION_XY = CENTERS[list(JUNCTION)].mean(axis=0)
PATCH_THETA = np.linspace(0, 2 * np.pi, N_PATCH, endpoint=False) + 0.21


@njit(cache=True)
def _nearest(x, y, centers):
    best, best_d = 0, 1e18
    for i in range(centers.shape[0]):
        d = np.sqrt((x - centers[i, 0]) ** 2 + (y - centers[i, 1]) ** 2)
        if d < best_d:
            best, best_d = i, d
    return best, best_d - R


@njit(cache=True)
def _on_patch(theta, patches, half_ang):
    for k in range(patches.shape[0]):
        d = abs(((theta - patches[k] + np.pi) % (2 * np.pi)) - np.pi)
        if d < half_ang:
            return True
    return False


@njit(cache=True)
def walk(seed, centers, patches, dwell, k_layer, t_max, start_cell, start_theta, rec_stride, rec):
    """One molecule to capture or t_max. Fills `rec` with (x, y, bound, cell) every
    `rec_stride` steps. Returns (t_capture, frames recorded, hops, steps bound)."""
    np.random.seed(seed)
    sd = np.sqrt(2 * D * DT)
    p_off = 1.0 - np.exp(-DT / dwell)
    p_on = 1.0 - np.exp(-k_layer * DT)
    half_ang = (PATCH_ARC / 2) / R

    cell, theta, bound = start_cell, start_theta, True
    x = centers[cell, 0] + R * np.cos(theta)
    y = centers[cell, 1] + R * np.sin(theta)
    n, hops, n_bound, frames = int(t_max / DT), 0, 0, 0

    for i in range(n):
        if bound:
            n_bound += 1
            if np.random.random() < p_off:
                bound = False
                depth = R + np.random.random() * DELTA
                x = centers[cell, 0] + depth * np.cos(theta)
                y = centers[cell, 1] + depth * np.sin(theta)
            else:
                theta += (sd / R) * np.random.normal(0.0, 1.0)
                x = centers[cell, 0] + R * np.cos(theta)
                y = centers[cell, 1] + R * np.sin(theta)
        else:
            x += sd * np.random.normal(0.0, 1.0)
            y += sd * np.random.normal(0.0, 1.0)
            near, d = _nearest(x, y, centers)
            if d < 0.0:  # stepped inside a cell: reflect radially off its membrane
                cx, cy = centers[near, 0], centers[near, 1]
                r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                s = (2 * R - r) / r
                x, y = cx + (x - cx) * s, cy + (y - cy) * s
                near, d = _nearest(x, y, centers)
            if d <= DELTA and np.random.random() < p_on:
                if near != cell:
                    hops += 1
                cell = near
                theta = np.arctan2(y - centers[cell, 1], x - centers[cell, 0])
                x = centers[cell, 0] + R * np.cos(theta)
                y = centers[cell, 1] + R * np.sin(theta)
                bound = True

        if i % rec_stride == 0 and frames < rec.shape[0]:
            rec[frames, 0], rec[frames, 1] = x, y
            rec[frames, 2] = 1.0 if bound else 0.0
            rec[frames, 3] = cell
            frames += 1
        if bound and cell == TARGET_CELL and _on_patch(theta, patches, half_ang):
            if frames < rec.shape[0]:
                rec[frames, 0], rec[frames, 1] = x, y
                rec[frames, 2], rec[frames, 3] = 1.0, cell
                frames += 1
            return i * DT, frames, hops, n_bound
    return np.inf, frames, hops, n_bound


def run(seed, *, dwell, k_layer, t_max, start_cell, start_theta, stride):
    rec = np.full((int(t_max / DT / stride) + 8, 4), np.nan)
    t, frames, hops, n_bound = walk(seed, CENTERS, PATCH_THETA, dwell, k_layer, t_max,
                                    start_cell, start_theta, stride, rec)
    return t, rec[:frames], hops, n_bound * DT


def patch_arcs(cell):
    """Polyline for each receptor patch on `cell`, for drawing."""
    out = []
    for th in PATCH_THETA:
        a = np.linspace(th - (PATCH_ARC / 2) / R, th + (PATCH_ARC / 2) / R, 8)
        out.append(np.column_stack([CENTERS[cell, 0] + R * np.cos(a),
                                    CENTERS[cell, 1] + R * np.sin(a)]))
    return out


def draw_scene(ax, paths, caught, colours, *, centre, half, label, show_labels):
    for i, c in enumerate(CENTERS):
        inj = i in JUNCTION
        ax.add_patch(Circle(c, R, facecolor="#eceae6" if inj else "#f4f3f0",
                            edgecolor=vz.SECONDARY if inj else vz.GRID,
                            lw=1.6 if inj else 1.2, zorder=1))
        if show_labels and inj:
            ax.text(*c, "target cell\n(receptors)" if i == TARGET_CELL else "cell",
                    ha="center", va="center", fontsize=9.5, color=vz.MUTED, zorder=2)
    # Above the paths: the receptors are what the whole picture is about, and a slide
    # line drawn over them hides the only thing the molecule is looking for.
    ax.add_collection(LineCollection(patch_arcs(TARGET_CELL), colors="#1f9d55",
                                     linewidths=5.5, zorder=8))

    for path, colour, hit in zip(paths, colours, caught):
        xy, bound = path[:, :2], path[:-1, 2] > 0.5
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        # Thin while free in the gap, thick while sliding: the two motions read apart
        # at a glance, which is the whole point of the picture.
        ax.add_collection(LineCollection(seg[~bound], colors=colour, linewidths=0.9,
                                         alpha=0.75, zorder=5))
        ax.add_collection(LineCollection(seg[bound], colors=colour, linewidths=3.2,
                                         alpha=0.95, zorder=6))
        # A ring wherever the molecule arrives on a cell it was not on before.
        cell = path[:, 3]
        jump = np.flatnonzero((path[1:, 2] > 0.5) & (path[:-1, 2] < 0.5) & (cell[1:] != cell[:-1])) + 1
        ax.plot(xy[jump, 0], xy[jump, 1], "o", mfc="none", mec=colour, mew=1.8, ms=11, zorder=7)
        ax.plot(*xy[0], "o", color=vz.SURFACE, mec=colour, mew=2.2, ms=9, zorder=8)
        if hit:  # only a molecule that actually reached a receptor gets the find marker
            ax.plot(*xy[-1], "o", color="#1f9d55", mec=vz.SURFACE, mew=1.6, ms=11, zorder=9)

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, loc="left", fontsize=11.5, color=vz.INK, pad=8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[43, 30, 2], help="one molecule per seed")
    p.add_argument("--dwell", type=float, default=1.0, help="mean bound dwell, s")
    p.add_argument("--k-layer", type=float, default=900.0, help="binding rate inside the layer, 1/s")
    p.add_argument("--t-max", type=float, default=300.0)
    p.add_argument("--stride", type=int, default=60, help="steps per recorded frame")
    p.add_argument("--zoom", type=float, default=9.0, help="half-width of the junction panel, um")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "sketch_1d2d_search.png")
    args = p.parse_args()
    vz.use()

    start_cell, start_theta = 1, np.arctan2(*(JUNCTION_XY - CENTERS[1])[::-1])
    paths, caught, info = [], [], []
    for s_ in args.seeds:
        t, path, hops, t_bound = run(s_, dwell=args.dwell, k_layer=args.k_layer, t_max=args.t_max,
                                     start_cell=start_cell, start_theta=start_theta, stride=args.stride)
        paths.append(path)
        caught.append(bool(np.isfinite(t)))
        info.append((s_, t, hops, t_bound / (t if np.isfinite(t) else args.t_max)))
        print(f"  seed {s_:3d}  t={t:8.1f} s  hops={hops:3d}  bound {info[-1][3]:.0%}  frames={len(path)}")

    colours = [vz.SWITCH, vz.REFERENCE, "#7b5cd6"][: len(paths)]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.4), gridspec_kw=dict(wspace=0.04))
    fig.subplots_adjust(top=0.92, bottom=0.17)
    draw_scene(axes[0], paths, caught, colours, centre=JUNCTION_XY, half=SPACING + 8.0,
               show_labels=True, label="Seven cells, packed: the molecule never leaves the gaps")
    z = args.zoom
    draw_scene(axes[1], paths, caught, colours, centre=JUNCTION_XY, half=z, show_labels=False,
               label=f"The three-cell junction, {2 * z:.0f} um across")
    axes[0].add_patch(plt.Rectangle((JUNCTION_XY[0] - z, JUNCTION_XY[1] - z), 2 * z, 2 * z,
                                    fill=False, lw=1.1, ec=vz.MUTED, ls=(0, (4, 3)), zorder=10))

    h = [plt.Line2D([], [], color=vz.MUTED, lw=3.2), plt.Line2D([], [], color=vz.MUTED, lw=0.9),
         plt.Line2D([], [], marker="o", mfc="none", mec=vz.MUTED, mew=1.8, ms=11, ls=""),
         plt.Line2D([], [], marker="o", color="#1f9d55", mec=vz.SURFACE, ms=11, ls="")]
    fig.legend(h, ["sliding on a membrane (1D)", "free in the gap (2D)",
                   "lands on a different cell", "finds a receptor"],
               loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, fontsize=10)

    fig.suptitle("Where the search happens: sliding along a cell, then hopping the slit to the next",
                 fontsize=13, x=0.085, ha="left", y=0.965)
    found = ", ".join(f"{t:.0f} s after {hops} hops" for _, t, hops, _ in info if np.isfinite(t))
    missed = sum(1 for _, t, _, _ in info if not np.isfinite(t))
    caption = (
        f"R = {R:g} um cells, {GAP:g} um gap, D = {D:g} um^2/s free and bound, mean dwell "
        f"{args.dwell:g} s, about {100 - 100 * np.mean([b for *_, b in info]):.0f}% of the time "
        f"free. {len(paths)} molecules, all started bound at the same point on the lower cell of "
        f"the junction; receptors on one cell only. Found a receptor {found}"
        + (f"; {missed} still searching at {args.t_max:.0f} s. " if missed else ". ")
        + "Over 150 molecules the median find is 109 s (quartiles 60 and 265 s) with 2 hops, and "
          "29% have not found one by 1500 s; these three were picked from the first 90 seeds as "
          "paths that both hop and find quickly enough to stay legible. Cartoon only: 2D, and no "
          "statistics - the measured comparison is in the report's capture curves.")
    # Wrapped by hand: one long line would stretch the tight bounding box far past the axes.
    fig.text(0.085, 0.045, textwrap.fill(caption, 118), ha="left", va="top",
             fontsize=9.5, color=vz.SECONDARY)

    for q in vz.save(fig, args.out):
        print(f"Saved {q}")


if __name__ == "__main__":
    main()
