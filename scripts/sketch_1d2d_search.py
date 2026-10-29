"""One molecule at a three-cell junction: sliding in 1D, diffusing in 2D (L110-L112).

Not a result — the simplest picture of the mechanism the project tests. Cells sit in
the project's own geometry (R = 10 um, 0.5 um gap). A single molecule is drawn, and the
only thing the colour means is which kind of walk it is doing:

    1D   bound, sliding along one cell's membrane
    2D   free, diffusing in the gap until it meets a membrane again

Where it rebinds to a *different* cell, it has hopped — only possible where cells come
close, which is why the junction is the interesting place to look. Free and bound use
the same D (L96), so nothing here comes from one state being faster.

    python scripts/sketch_1d2d_search.py              # docs/figures/sketch_1d2d_search.{png,pdf}
    python scripts/sketch_1d2d_search.py --animate    # and the mp4 of the same walk

The still and the movie are rendered from one trajectory, so they cannot disagree.
There are no receptors and no capture here: it is a picture of how the molecule moves,
and every measured comparison lives in the report's own capture curves.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
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

C_1D, C_2D = vz.SWITCH, vz.REFERENCE   # the only two colours in the figure

# A 7-cell patch, not a bare trio: with only three cells a molecule that unbinds on an
# outer face diffuses off into vacuum, which is not what the gap looks like in a packed
# lattice. Cells 0, 1, 2 are the junction the view is centred on.
_ANG = np.arange(6) * np.pi / 3 + np.pi / 3
CENTERS = np.ascontiguousarray(np.vstack([[0.0, 0.0],
                                          SPACING * np.column_stack([np.cos(_ANG), np.sin(_ANG)])]))
JUNCTION = (0, 1, 2)
JUNCTION_XY = CENTERS[list(JUNCTION)].mean(axis=0)


@njit(cache=True)
def _nearest(x, y, centers):
    best, best_d = 0, 1e18
    for i in range(centers.shape[0]):
        d = np.sqrt((x - centers[i, 0]) ** 2 + (y - centers[i, 1]) ** 2)
        if d < best_d:
            best, best_d = i, d
    return best, best_d - R


@njit(cache=True)
def walk(seed, centers, dwell, k_layer, t_max, start_cell, start_theta, stride, rec, hop_xy, hop_t):
    """Record (x, y, bound, cell) every `stride` steps for t_max seconds.

    Hops are written out as they happen, with the frame they fall in. Recomputing them
    from `rec` afterwards does not work: a hop can start and finish between two recorded
    frames, so the subsampled record shows a cell change with no free interval and the
    count comes out wrong.
    """
    np.random.seed(seed)
    sd = np.sqrt(2 * D * DT)
    p_off = 1.0 - np.exp(-DT / dwell)
    p_on = 1.0 - np.exp(-k_layer * DT)

    cell, theta, bound = start_cell, start_theta, True
    x = centers[cell, 0] + R * np.cos(theta)
    y = centers[cell, 1] + R * np.sin(theta)
    frames, hops = 0, 0

    for i in range(int(t_max / DT)):
        if bound:
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
                if near != cell and hops < hop_xy.shape[0]:
                    hop_xy[hops, 0], hop_xy[hops, 1] = x, y
                    hop_t[hops] = frames
                    hops += 1
                cell = near
                theta = np.arctan2(y - centers[cell, 1], x - centers[cell, 0])
                x = centers[cell, 0] + R * np.cos(theta)
                y = centers[cell, 1] + R * np.sin(theta)
                bound = True

        if i % stride == 0 and frames < rec.shape[0]:
            rec[frames, 0], rec[frames, 1] = x, y
            rec[frames, 2] = 1.0 if bound else 0.0
            rec[frames, 3] = cell
            frames += 1
    return frames, hops


def run(seed, *, dwell, k_layer, t_max, stride):
    """One walk, started bound on cell 1 facing the junction."""
    rec = np.full((int(t_max / DT / stride) + 8, 4), np.nan)
    hop_xy, hop_t = np.zeros((4096, 2)), np.zeros(4096, dtype=np.int64)
    theta = np.arctan2(*(JUNCTION_XY - CENTERS[1])[::-1])
    frames, hops = walk(seed, CENTERS, dwell, k_layer, t_max, 1, theta, stride,
                        rec, hop_xy, hop_t)
    return rec[:frames], hop_xy[:hops], hop_t[:hops]


def draw_cells(ax, half):
    for i, c in enumerate(CENTERS):
        inj = i in JUNCTION
        ax.add_patch(Circle(c, R, facecolor="#eeece8" if inj else "#f5f4f1",
                            edgecolor=vz.MUTED if inj else vz.GRID, lw=1.4, zorder=1))
    ax.set_xlim(JUNCTION_XY[0] - half, JUNCTION_XY[0] + half)
    ax.set_ylim(JUNCTION_XY[1] - half, JUNCTION_XY[1] + half)
    ax.set_aspect("equal")
    ax.axis("off")


def state_segments(path):
    """Segments split by state, so each is drawn in its own colour and weight."""
    xy = path[:, :2]
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    bound = path[:-1, 2] > 0.5
    return seg[bound], seg[~bound]


def legend(ax, loc="upper left"):
    h = [plt.Line2D([], [], color=C_1D, lw=3.4), plt.Line2D([], [], color=C_2D, lw=1.7),
         plt.Line2D([], [], marker="o", mfc="none", mec=vz.INK, mew=1.6, ms=11, ls="")]
    ax.legend(h, ["1D: sliding on a membrane", "2D: free in the gap", "lands on a different cell"],
              loc=loc, fontsize=10.5, borderaxespad=1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--dwell", type=float, default=1.0, help="mean bound dwell, s")
    p.add_argument("--k-layer", type=float, default=900.0, help="binding rate inside the layer, 1/s")
    p.add_argument("--t-max", type=float, default=90.0, help="seconds of the walk to show")
    p.add_argument("--stride", type=int, default=100, help="steps per recorded frame")
    p.add_argument("--half", type=float, default=21.5, help="half-width of the view, um")
    p.add_argument("--animate", action="store_true", help="also write the mp4 of the same walk")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=18.0, help="length of the movie")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "sketch_1d2d_search.png")
    args = p.parse_args()
    vz.use()

    path, hop_xy, hop_t = run(args.seed, dwell=args.dwell, k_layer=args.k_layer,
                              t_max=args.t_max, stride=args.stride)
    hops = len(hop_xy)
    bound_frac = (path[:, 2] > 0.5).mean()
    print(f"seed {args.seed}: {args.t_max:g} s, {hops} hops, {1 - bound_frac:.0%} of the time free, "
          f"{len(path)} frames")

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    draw_cells(ax, args.half)
    slide, free = state_segments(path)
    ax.add_collection(LineCollection(free, colors=C_2D, linewidths=1.7, alpha=0.95, zorder=4))
    ax.add_collection(LineCollection(slide, colors=C_1D, linewidths=3.4, alpha=0.95, zorder=5))
    ax.plot(hop_xy[:, 0], hop_xy[:, 1], "o", mfc="none", mec=vz.INK, mew=1.6, ms=11, zorder=7)
    ax.plot(*path[0, :2], "o", color=vz.SURFACE, mec=vz.INK, mew=1.8, ms=8, zorder=8)
    legend(ax)

    ax.set_title("One molecule at a three-cell junction", loc="left", fontsize=13,
                 color=vz.INK, pad=10)
    fig.text(0.02, 0.055, textwrap.fill(
        f"{args.t_max:g} s of a single walk. R = {R:g} um cells, {GAP:g} um gap, "
        f"D = {D:g} um^2/s in both states, mean dwell {args.dwell:g} s; "
        f"{1 - bound_frac:.0%} of the time free, {hops} hops between cells. "
        "Cartoon: 2D, and nothing is being measured here.", 92),
        ha="left", va="top", fontsize=10, color=vz.SECONDARY)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.12)

    for q in vz.save(fig, args.out):
        print(f"Saved {q}")
    plt.close(fig)

    if args.animate:
        animate(path, hop_xy, hop_t, args)


def animate(path, hop_xy, hop_t, args):
    """The same walk, played out: the trail grows and changes colour with the state."""
    n = max(1, int(args.fps * args.seconds))
    step = max(1, len(path) // n)
    idx = list(range(1, len(path), step))
    n = len(idx)

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    draw_cells(ax, args.half)
    free_lc = LineCollection([], colors=C_2D, linewidths=1.7, alpha=0.95, zorder=4)
    slide_lc = LineCollection([], colors=C_1D, linewidths=3.4, alpha=0.95, zorder=5)
    ax.add_collection(free_lc)
    ax.add_collection(slide_lc)
    hops_plot, = ax.plot([], [], "o", mfc="none", mec=vz.INK, mew=1.6, ms=11, zorder=7)
    head, = ax.plot([], [], "o", mec=vz.SURFACE, mew=1.6, ms=13, zorder=9)
    # Opposite corner from the legend, which lives at the upper left.
    clock = ax.text(0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right",
                    fontsize=12.5, color=vz.INK)
    legend(ax)
    ax.set_title("One molecule at a three-cell junction", loc="left", fontsize=13,
                 color=vz.INK, pad=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)

    def update(k):
        j = idx[k]
        slide, free = state_segments(path[: j + 1])
        slide_lc.set_segments(list(slide))
        free_lc.set_segments(list(free))
        shown = hop_xy[hop_t <= j]
        hops_plot.set_data(shown[:, 0], shown[:, 1])
        bound = path[j, 2] > 0.5
        head.set_data([path[j, 0]], [path[j, 1]])
        head.set_color(C_1D if bound else C_2D)
        clock.set_text(f"{j * args.stride * DT:.1f} s of {args.t_max:g}\n"
                       f"{'1D: sliding' if bound else '2D: free in the gap'}")
        return ()

    out = args.out.parent / (args.out.stem + ".mp4")
    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / args.fps, blit=False)
    # CRF rather than a bitrate: line art on a flat background compresses far better that
    # way. yuv420p needs even dimensions, which figsize*dpi often is not, hence the pad.
    writer = animation.FFMpegWriter(fps=args.fps, codec="libx264",
                                    extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:0xFCFCFB",
                                                "-crf", "24", "-preset", "slow",
                                                "-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"Saved {out}  ({n} frames, {n / args.fps:.0f} s of video for "
          f"{args.t_max:g} s of walk, {step}x subsampled)")


if __name__ == "__main__":
    main()
