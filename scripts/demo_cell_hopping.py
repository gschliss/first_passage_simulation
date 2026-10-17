"""Single-molecule demo of the agreed 2D model, chosen so cell-to-cell hops are visible.

Model (docs/model_notes.md, L20-L28):
  bound: slides along its cell's circumference, D_BOUND; unbinds at rate k_off and
         lands at a uniform depth 0-DELTA outside the membrane at the same spot.
  free:  diffuses at D_FREE, reflects off cells; can bind only within DELTA of a
         membrane, at rate k_layer = k_on * V_free / V_layer, inserting at the
         nearest membrane point (which may be a different cell).

"Generous" choices for the demo: slowest switching in the sweep (bound dwell
1000 ms, so free trips are longest) and a start position bound at a slit facing
a neighbor. No receptors and no escape boundary here.

Usage:
    python scripts/demo_cell_hopping.py --t-max 20 --seed 3
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RADIUS = 10.0  # µm (L2)
GAP = 0.5  # µm (L2)
SPACING = 2 * RADIUS + GAP
D_FREE = 10.0  # µm²/s (L20)
D_BOUND = 0.5  # µm²/s (L20)
DELTA = 0.010  # µm, binding layer (L28)
DWELL = 1.0  # s, mean bound dwell 1/k_off (slow end of L27)
K_OFF = 1 / DWELL
K_ON = 19 * K_OFF  # 5% free (L23)
# Infinite 2D hex lattice, per cell: free area 49.79 µm², layer area 0.6286 µm².
VFREE_OVER_VLAYER = 79.2
K_LAYER = K_ON * VFREE_OVER_VLAYER

DT_BOUND = 1e-3  # s
DT_LAYER = (DELTA / 4) ** 2 / (2 * D_FREE)  # resolve the 10 nm layer
DT_FREE_MAX = 1e-4  # s


def hex_centers(rings: int):
    cells = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if (abs(q) + abs(r) + abs(q + r)) // 2 <= rings:
                cells.append((SPACING * (q + r / 2), SPACING * math.sqrt(3) / 2 * r))
    return cells


def nearest_cell(x, y, centers):
    best, best_d = 0, float("inf")
    for i, (cx, cy) in enumerate(centers):
        d = math.hypot(x - cx, y - cy)
        if d < best_d:
            best, best_d = i, d
    return best, best_d - RADIUS


def simulate(centers, start_cell, start_theta, t_max, rng):
    """Return a list of samples (t, x, y, state, cell); state 1 = bound, 0 = free."""
    t, cell, theta = 0.0, start_cell, start_theta
    state = 1
    cx, cy = centers[cell]
    x, y = cx + RADIUS * math.cos(theta), cy + RADIUS * math.sin(theta)
    samples = [(t, x, y, state, cell)]
    p_off = 1 - math.exp(-K_OFF * DT_BOUND)
    sig_theta = math.sqrt(2 * D_BOUND * DT_BOUND) / RADIUS
    last_free_sample = 0.0

    while t < t_max:
        if state == 1:
            t += DT_BOUND
            if rng.random() < p_off:
                depth = rng.uniform(0, DELTA)
                x = cx + (RADIUS + depth) * math.cos(theta)
                y = cy + (RADIUS + depth) * math.sin(theta)
                state = 0
            else:
                theta += sig_theta * rng.gauss(0, 1)
                x, y = cx + RADIUS * math.cos(theta), cy + RADIUS * math.sin(theta)
            samples.append((t, x, y, state, cell))
            continue

        near, d = nearest_cell(x, y, centers)
        if d <= DELTA:
            dt = DT_LAYER
            if rng.random() < 1 - math.exp(-K_LAYER * dt):
                cell = near
                cx, cy = centers[cell]
                theta = math.atan2(y - cy, x - cx)
                x, y = cx + RADIUS * math.cos(theta), cy + RADIUS * math.sin(theta)
                state = 1
                t += dt
                samples.append((t, x, y, state, cell))
                continue
        else:
            dt = min(DT_FREE_MAX, max(DT_LAYER, ((d - DELTA) / 4) ** 2 / (2 * D_FREE)))
        sig = math.sqrt(2 * D_FREE * dt)
        x += sig * rng.gauss(0, 1)
        y += sig * rng.gauss(0, 1)
        near, d = nearest_cell(x, y, centers)
        if d < 0:  # stepped into a cell: reflect radially off its surface
            ncx, ncy = centers[near]
            r = math.hypot(x - ncx, y - ncy)
            scale = (2 * RADIUS - r) / r
            x, y = ncx + (x - ncx) * scale, ncy + (y - ncy) * scale
        t += dt
        if t - last_free_sample > 2e-5:
            samples.append((t, x, y, state, cell))
            last_free_sample = t
    return samples


def find_hops(samples):
    """Indices where a molecule binds to a different cell than it last left."""
    hops, prev_cell = [], samples[0][4]
    for i, (_, _, _, state, cell) in enumerate(samples):
        if state == 1 and cell != prev_cell:
            hops.append(i)
        if state == 1:
            prev_cell = cell
    return hops


CELL_COLORS = plt.get_cmap("tab10").colors
FREE_COLOR = "#111111"


def segment_colors(arr):
    """Color each step: black while free, the bound cell's color while bound."""
    state, cell = arr[:-1, 3], arr[:-1, 4].astype(int)
    return [FREE_COLOR if s == 0 else CELL_COLORS[c % 10] for s, c in zip(state, cell)]


def draw_cells(ax, centers, used_cells, label=True):
    for i, (cx, cy) in enumerate(centers):
        color = CELL_COLORS[i % 10] if i in used_cells else "#bbbbbb"
        ax.add_patch(Circle((cx, cy), RADIUS, facecolor=color, alpha=0.12 if i in used_cells else 0.15,
                            edgecolor=color, lw=1.2))
        if label:
            ax.text(cx, cy, f"cell {i}", ha="center", va="center", fontsize=7, color="#444")


def zoom_window(arr, pad=1.2):
    x0, x1 = arr[:, 1].min() - pad, arr[:, 1].max() + pad
    y0, y1 = arr[:, 2].min() - pad, arr[:, 2].max() + pad
    half = max(x1 - x0, y1 - y0) / 2
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    return xc - half, xc + half, yc - half, yc + half


def plot_static(arr, hops, centers, out):
    used = set(arr[:, 4].astype(int))
    pts = arr[:, 1:3]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    colors = segment_colors(arr)
    widths = [0.6 if c == FREE_COLOR else 1.6 for c in colors]
    zx0, zx1, zy0, zy1 = zoom_window(arr)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 0.55])
    ax_ctx, ax_zoom = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])

    # A. Context: the 19-cell patch
    draw_cells(ax_ctx, centers, used)
    ax_ctx.add_collection(LineCollection(segs, colors=colors, linewidths=widths))
    ax_ctx.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, lw=1))
    ext = 3 * SPACING
    ax_ctx.set_xlim(-ext, ext); ax_ctx.set_ylim(-ext, ext); ax_ctx.set_aspect("equal")
    ax_ctx.plot([-ext + 4, -ext + 24], [-ext + 4] * 2, "k", lw=3)
    ax_ctx.text(-ext + 14, -ext + 6, "20 µm", ha="center", fontsize=8)
    ax_ctx.set_title("A. Context: 19-cell patch (box = zoom in B)")
    ax_ctx.axis("off")

    # B. Zoom on the trajectory
    draw_cells(ax_zoom, centers, used, label=False)
    ax_zoom.add_collection(LineCollection(segs, colors=colors, linewidths=widths))
    ax_zoom.plot(*pts[0], "o", color="k", ms=7, mfc="white", zorder=5, label="start (bound)")
    for k, i in enumerate(hops):
        ax_zoom.plot(*pts[i], "*", color="gold", mec="k", ms=14, zorder=6, label="hop: binds a new cell" if k == 0 else None)
    ax_zoom.set_xlim(zx0, zx1); ax_zoom.set_ylim(zy0, zy1); ax_zoom.set_aspect("equal")
    ax_zoom.plot([], [], color=FREE_COLOR, lw=0.8, label="free (D = 10 µm²/s)")
    ax_zoom.plot([], [], color="#888", lw=2, label="bound, colored by cell (D = 0.5 µm²/s)")
    ax_zoom.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_zoom.set_xlabel("µm"); ax_zoom.set_ylabel("µm")
    ax_zoom.set_title(f"B. Zoom: one molecule for {arr[-1, 0]:.0f} s, {len(hops)} hops between cells")

    # C. Timeline: which cell the molecule is on, and free trips
    t = arr[:, 0]
    bound = arr[:, 3] == 1
    cells_sorted = sorted(used)
    row = {c: k for k, c in enumerate(cells_sorted)}
    for c in cells_sorted:
        m = bound & (arr[:, 4] == c)
        ax_t.plot(t[m], np.full(m.sum(), row[c]), "|", color=CELL_COLORS[c % 10], ms=10, mew=1)
    starts = np.flatnonzero(np.diff(arr[:, 3]) == -1)
    for i in starts:
        ax_t.axvline(t[i], color=FREE_COLOR, lw=0.5, alpha=0.5)
    for i in hops:
        ax_t.plot(t[i], row[int(arr[i, 4])], "*", color="gold", mec="k", ms=12)
    ax_t.set_yticks(range(len(cells_sorted)), [f"cell {c}" for c in cells_sorted])
    ax_t.set_ylim(-0.7, len(cells_sorted) - 0.3)
    ax_t.set_xlim(0, t[-1])
    ax_t.set_xlabel("time (s)")
    ax_t.set_title("C. Timeline: bound cell over time; thin lines = free trips (each lasts ~ms, too short to see as width); stars = hops")

    fig.suptitle(f"2D model, one molecule: bound dwell {DWELL*1e3:.0f} ms, 5% free, binding layer {DELTA*1e3:.0f} nm, "
                 f"cell radius {RADIUS} µm, gap {GAP} µm", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gif(arr, hops, centers, out, frame_dt=0.05, trail=2.0, fps=20):
    from matplotlib.animation import FuncAnimation, PillowWriter

    used = set(arr[:, 4].astype(int))
    zx0, zx1, zy0, zy1 = zoom_window(arr)
    t, pts = arr[:, 0], arr[:, 1:3]
    colors = np.array([plt.matplotlib.colors.to_rgba(c) for c in segment_colors(arr)])
    hop_times = t[hops]

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_cells(ax, centers, used, label=False)
    ax.set_xlim(zx0, zx1); ax.set_ylim(zy0, zy1); ax.set_aspect("equal")
    ax.set_xlabel("µm"); ax.set_ylabel("µm")
    lc = LineCollection([], linewidths=1.4)
    ax.add_collection(lc)
    dot, = ax.plot([], [], "o", ms=8, mec="k", zorder=5)
    stars, = ax.plot([], [], "*", color="gold", mec="k", ms=14, zorder=6, ls="none")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_title("Black = free, color = bound to that cell; trail = last 2 s")
    n_frames = int(np.ceil(t[-1] / frame_dt))

    def update(f):
        now = (f + 1) * frame_dt
        i1 = np.searchsorted(t, now)
        i0 = np.searchsorted(t, now - trail)
        seg = np.stack([pts[i0:i1 - 1], pts[i0 + 1:i1]], axis=1) if i1 - i0 > 1 else np.empty((0, 2, 2))
        c = colors[i0:i1 - 1].copy()
        if len(c):
            c[:, 3] = np.linspace(0.1, 1.0, len(c))
        lc.set_segments(seg)
        lc.set_color(c)
        lc.set_linewidth([0.7 if tuple(cc[:3]) == plt.matplotlib.colors.to_rgb(FREE_COLOR) else 1.8 for cc in c])
        j = max(i1 - 1, 0)
        state, cell = arr[j, 3], int(arr[j, 4])
        dot.set_data([pts[j, 0]], [pts[j, 1]])
        dot.set_color(FREE_COLOR if state == 0 else CELL_COLORS[cell % 10])
        dot.set_markeredgecolor("k")
        past = [i for i in hops if t[i] <= now]
        stars.set_data(pts[past, 0], pts[past, 1])
        n_hops = len(past)
        recent = np.any((hop_times > now - frame_dt * 6) & (hop_times <= now))
        txt.set_text(f"t = {now:5.2f} s\n{'FREE' if state == 0 else f'bound to cell {cell}'}\n"
                     f"hops so far: {n_hops}" + ("   ★ HOP!" if recent else ""))
        return lc, dot, stars, txt

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-max", type=float, default=20.0, help="simulated seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true", help="also write an animated GIF")
    args = parser.parse_args()

    centers = hex_centers(2)
    rng = random.Random(args.seed)
    center = min(range(len(centers)), key=lambda i: math.hypot(*centers[i]))
    # Start bound on the center cell, at the slit facing its right-hand neighbor.
    samples = simulate(centers, start_cell=center, start_theta=0.0, t_max=args.t_max, rng=rng)
    arr = np.array(samples)
    hops = find_hops(samples)
    n_trips = int(np.sum(np.diff(arr[:, 3]) == -1))
    dt = np.diff(arr[:, 0])
    free_frac = dt[arr[:-1, 3] == 0].sum() / dt.sum()
    print(f"{args.t_max:.0f} s simulated: {n_trips} free trips, {len(hops)} cell-to-cell hops, "
          f"time free {free_frac:.3f} (target 0.05)")

    out_dir = PROJECT_ROOT / "results" / "sketches"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cell_hopping_seed{args.seed}"
    plot_static(arr, hops, centers, out_dir / f"{stem}.png")
    print(f"Saved {out_dir / f'{stem}.png'}")
    if args.gif:
        plot_gif(arr, hops, centers, out_dir / f"{stem}.gif")
        print(f"Saved {out_dir / f'{stem}.gif'}")


if __name__ == "__main__":
    main()
