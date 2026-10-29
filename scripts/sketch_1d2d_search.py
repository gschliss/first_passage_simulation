"""Toy 2D/1D search cartoon: bulk diffusion plus sliding on a line (L110).

Not the model and not a result — a teaching picture of the *mechanism* the project
tests, drawn one dimension down so the whole search fits on a page. A walker lives
in a 2D box; the bottom edge is a 1D "membrane" carrying one short target patch.

    free     diffuses in the box at D, reflects off all four walls
    bound    binds on contact with the bottom edge, slides along it in 1D at the
             same D, unbinds at rate 1/dwell
    capture  reaching the target patch while in contact with the edge

Both arms use the same D in both states, as in the real runs (L96), so anything
that differs between them is geometry and not speed. The non-switching arm is the
identical walker with binding switched off: it must land on the patch out of the
bulk.

    python scripts/sketch_1d2d_search.py --seed 4

Writes docs/figures/sketch_1d2d_search.{png,pdf}. --walkers 0 draws the two paths
only and skips the capture-curve panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import vizstyle as vz

PROJECT_ROOT = Path(__file__).resolve().parents[1]

L, H = 16.0, 12.0       # box, um
TARGET_X, TARGET_W = 8.0, 0.15
D = 1.0                 # um^2/s, free and bound alike
DELTA = 0.1             # contact layer, um
K_ON = 50.0             # 1/s inside the layer
DT = 0.002


def walk(rng, *, switching, dwell, t_max, record=False):
    """One walker to capture or t_max. Returns (t_capture, path) with path = None unless recorded."""
    sd = np.sqrt(2 * D * DT)
    p_bind = 1 - np.exp(-K_ON * DT) if switching else 0.0
    p_off = 1 - np.exp(-DT / dwell)
    x, y, bound = rng.uniform(0.35 * L, 0.65 * L), H * 0.85, False
    half, n = TARGET_W / 2, int(t_max / DT)
    path = [] if record else None
    for i in range(n):
        if bound:
            x += rng.normal(0, sd)
            if rng.random() < p_off:
                bound, y = False, rng.uniform(0, DELTA)
        else:
            x += rng.normal(0, sd)
            y += rng.normal(0, sd)
            y = abs(y) if y < 0 else (2 * H - y if y > H else y)
            if y < DELTA and rng.random() < p_bind:
                bound, y = True, 0.0
        x = abs(x) if x < 0 else (2 * L - x if x > L else x)
        if record:
            path.append((x, y, bound))
        if (bound or y < DELTA) and abs(x - TARGET_X) < half:
            return i * DT, (np.array(path) if record else None)
    return np.inf, (np.array(path) if record else None)


def draw_path(ax, path, colour, *, label, stride):
    """Thin line for bulk steps, thick for slides; a dot at the start and the find.

    `stride` coarse-grains the path before drawing. Both panels use the same stride, so
    one is not drawn smoother than the other, and the PDF stays small enough to edit.
    """
    path = np.vstack([path[::stride], path[-1]])
    ax.add_patch(plt.Rectangle((0, 0), L, H, facecolor="#f7f7f5", edgecolor=vz.GRID, lw=1.0, zorder=0))
    ax.plot([0, L], [0, 0], color=vz.SECONDARY, lw=2.5, solid_capstyle="butt", zorder=2)
    ax.plot([TARGET_X - TARGET_W / 2, TARGET_X + TARGET_W / 2], [0, 0],
            color="#1f9d55", lw=7, solid_capstyle="butt", zorder=3)
    ax.plot([TARGET_X], [-0.55], marker="^", color="#1f9d55", ms=9, zorder=3)

    xy = path[:, :2]
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    bound = path[:-1, 2] > 0.5
    # Two collections so the slides read as a different motion, not just a different colour.
    ax.add_collection(LineCollection(seg[~bound], colors=colour, linewidths=0.6, alpha=0.45, zorder=4))
    if bound.any():
        ax.add_collection(LineCollection(seg[bound], colors=colour, linewidths=3.0, alpha=0.95, zorder=5))
    ax.plot(*xy[0], "o", color=vz.SURFACE, mec=colour, mew=2, ms=8, zorder=6)
    ax.annotate("start", xy=xy[0], xytext=(9, 4), textcoords="offset points",
                fontsize=9, color=vz.SECONDARY, zorder=6)
    ax.plot(*xy[-1], "o", color="#1f9d55", mec=vz.SURFACE, mew=1.5, ms=9, zorder=7)
    ax.set_xlim(-0.4, L + 0.4)
    ax.set_ylim(-1.5, H + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, loc="left", fontsize=11, color=vz.INK, pad=6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=2, help="seed for the drawn switching path")
    p.add_argument("--seed-ref", type=int, default=2, help="seed for the drawn bulk-only path")
    p.add_argument("--stride", type=int, default=25, help="steps per drawn segment (0.05 s at the default dt)")
    p.add_argument("--dwell", type=float, default=2.0, help="mean bound dwell, s")
    p.add_argument("--t-max", type=float, default=2000.0)
    p.add_argument("--walkers", type=int, default=200, help="per arm for the capture panel; 0 to skip it")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "sketch_1d2d_search.png")
    args = p.parse_args()
    vz.use()

    rng = np.random.default_rng(args.seed)
    t_sw, path_sw = walk(rng, switching=True, dwell=args.dwell, t_max=args.t_max, record=True)
    t_ref, path_ref = walk(np.random.default_rng(args.seed_ref), switching=False,
                           dwell=args.dwell, t_max=args.t_max, record=True)
    print(f"drawn paths: switching {t_sw:.1f} s, non-switching {t_ref:.1f} s")

    ncol = 2 if args.walkers == 0 else 3
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.6),
                             gridspec_kw=dict(wspace=0.14, width_ratios=[1] * ncol))
    draw_path(axes[0], path_ref, vz.REFERENCE, stride=args.stride,
              label=f"Bulk only  —  found at {t_ref:.0f} s")
    draw_path(axes[1], path_sw, vz.SWITCH, stride=args.stride,
              label=f"Bulk + sliding  —  found at {t_sw:.0f} s")
    axes[0].annotate("target patch", xy=(TARGET_X, -0.8), xytext=(0, -8), textcoords="offset points",
                     ha="center", va="top", fontsize=9.5, color="#1f9d55")
    axes[1].annotate("thick = sliding along the line in 1D", xy=(TARGET_X, -0.8), xytext=(0, -8),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=9.5, color=vz.SECONDARY)

    if args.walkers:
        ts = {}
        for role, sw in (("reference", False), ("switch", True)):
            r = np.random.default_rng(args.seed + 1000)
            ts[role] = np.array([walk(r, switching=sw, dwell=args.dwell, t_max=args.t_max)[0]
                                 for _ in range(args.walkers)])
        ax = axes[2]
        for role in ("switch", "reference"):
            t = np.sort(ts[role])
            t = t[np.isfinite(t)]
            y = np.arange(1, len(t) + 1) / args.walkers
            ax.step(np.concatenate([[1e-2], t]), np.concatenate([[0], y]), where="post",
                    color=vz.ROLE[role], lw=2.2,
                    label=f"{'bulk + sliding' if role == 'switch' else 'bulk only'} "
                          f"({len(t) / args.walkers:.0%} found by {args.t_max:.0f} s)")
            frac = len(t) / args.walkers
            print(f"  {role:10} median {np.median(t) if len(t) else np.nan:7.1f} s, found {frac:.0%}")
        ax.axhline(0.20, color=vz.INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        # Start the axis at the first find, not at the step size: three empty decades
        # of log axis make the two arms look closer together than they are.
        first = min(np.min(v[np.isfinite(v)]) for v in ts.values())
        last = max(np.max(v[np.isfinite(v)]) for v in ts.values())
        ax.set_xlim(first * 0.7, last * 1.3)
        vz.log_ticks(ax.xaxis, first * 0.7, last * 1.3)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("fraction of walkers that found the patch")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{args.walkers} walkers per arm", loc="left", fontsize=11, color=vz.INK, pad=6)
        vz.style(ax)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Toy 2D/1D search: the same walker, with and without sliding on the line",
                 fontsize=12.5, x=0.09, ha="left", y=1.04)
    fig.text(0.09, -0.02,
             f"Box {L:g} x {H:g} um, target patch {TARGET_W:g} um ({TARGET_W / L:.0%} of the line), "
             f"D = {D:g} um^2/s free and bound, mean dwell {args.dwell:g} s. Cartoon only: "
             "one dimension below the simulations, and not calibrated to them. Each path is a "
             "near-median example of its arm, not the best one.",
             ha="left", fontsize=9.5, color=vz.SECONDARY)

    for q in vz.save(fig, args.out):
        print(f"Saved {q}")


if __name__ == "__main__":
    main()
