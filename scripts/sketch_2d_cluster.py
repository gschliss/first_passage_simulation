"""Static, to-scale sketch of the 2D hexagonal cell cluster (no simulation).

Used to choose cluster size and source-target separation by eye.

Usage:
    python scripts/sketch_2d_cluster.py --target-dist 4 --seed 0

The source cell sits at the cluster center (L12). Every cell at ring distance
``target-dist`` is a candidate target (L11); one is drawn at random as an example.
The cluster has ``target-dist + 2`` rings unless ``--rings`` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RADIUS = 10.0  # µm (L2)
GAP = 0.5  # µm at closest approach (L2)
SPACING = 2 * RADIUS + GAP


def hex_cluster(rings: int):
    """Axial coords (q, r), hex distance from center, and xy centers in µm."""
    cells = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist <= rings:
                cells.append((q, r, dist))
    qr = np.array([(q, r) for q, r, _ in cells])
    dist = np.array([d for *_, d in cells])
    xy = np.column_stack([SPACING * (qr[:, 0] + qr[:, 1] / 2), SPACING * np.sqrt(3) / 2 * qr[:, 1]])
    return qr, dist, xy


def draw_cells(ax, qr, dist, xy, target_qr, target_dist, label=True):
    for (q, r), d, (x, y) in zip(qr, dist, xy):
        lw = 0.8
        if (q, r) == (0, 0):
            face, edge, tag = "#a6c8f4", "#1f4e9c", "source"
        elif (q, r) == target_qr:
            face, edge, tag, lw = "#f4a6a6", "#b22222", "target", 1.5
        elif d == target_dist:
            face, edge, tag, lw = "#fbe3e3", "#b22222", str(d), 1.2
        else:
            face, edge, tag = "#e6e6e6", "#888888", str(d)
        ax.add_patch(Circle((x, y), RADIUS, facecolor=face, edgecolor=edge, lw=lw))
        if label:
            ax.text(x, y, tag, ha="center", va="center", fontsize=7, color="#333333")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dist", type=int, default=4, help="target ring distance from source, in cells")
    parser.add_argument("--rings", type=int, default=None, help="cluster rings (default: target-dist + 2)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the example target choice")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rings = args.rings if args.rings is not None else args.target_dist + 2
    qr, dist, xy = hex_cluster(rings)
    candidates = np.flatnonzero(dist == args.target_dist)
    pick = np.random.default_rng(args.seed).choice(candidates)
    target_qr, target_xy = tuple(qr[pick]), xy[pick]
    cand_r = np.linalg.norm(xy[candidates], axis=1)

    fig, (ax, zx) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})

    # Overview
    draw_cells(ax, qr, dist, xy, target_qr, args.target_dist)
    ax.annotate("", xy=target_xy, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.2, shrinkA=0, shrinkB=0))
    sep = np.linalg.norm(target_xy)
    y_text = -rings * SPACING * np.sqrt(3) / 2 - RADIUS - 4
    ax.text(0, y_text,
            f"example target: {sep:.1f} µm center-to-center ({sep - 2 * RADIUS:.1f} µm surface-to-surface)\n"
            f"all {candidates.size} ring-{args.target_dist} candidates: {cand_r.min():.1f}–{cand_r.max():.1f} µm center-to-center",
            ha="center", va="top", fontsize=8, color="#b22222")
    ax.add_patch(plt.Rectangle((6, -2.5), 8.5, 11.5, fill=False, color="k", lw=1))
    ax.text(14.8, 9, "zoom", fontsize=7, va="bottom")
    extent = (rings + 1) * SPACING
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.plot([-extent + 5, -extent + 25], [-extent + 5] * 2, color="k", lw=3)
    ax.text(-extent + 15, -extent + 7.5, "20 µm", ha="center", fontsize=8)
    ax.set_title(f"{len(qr)} cells ({rings} rings); numbers = cells from source; outlined = candidate targets")
    ax.axis("off")

    # Zoom on one gap and one triangular pore
    draw_cells(zx, qr, dist, xy, target_qr, args.target_dist, label=False)
    zx.annotate("", xy=(RADIUS, 0), xytext=(RADIUS + GAP, 0),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1, shrinkA=0, shrinkB=0))
    zx.text(RADIUS + GAP / 2, -0.6, f"gap {GAP} µm", ha="center", va="top", fontsize=9)
    pore = np.array([SPACING / 2, SPACING / (2 * np.sqrt(3))])
    pore_r = SPACING / np.sqrt(3) - RADIUS
    zx.add_patch(Circle(pore, pore_r, fill=False, ls="--", color="#b8860b", lw=1.2))
    zx.text(pore[0], pore[1] + pore_r + 0.3, f"pore radius {pore_r:.2f} µm", ha="center", fontsize=9, color="#b8860b")
    zx.set_xlim(6, 14.5)
    zx.set_ylim(-2.5, 9)
    zx.set_aspect("equal")
    zx.set_title("Zoom: slit between two cells and the pore between three")
    zx.set_xlabel("µm")
    zx.set_ylabel("µm")

    fig.suptitle(f"2D hexagonal cluster to scale: cell radius {RADIUS} µm, gap {GAP} µm, spacing {SPACING} µm")
    fig.tight_layout()
    out = args.out or PROJECT_ROOT / "results" / "sketches" / f"cluster_2d_target{args.target_dist}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
