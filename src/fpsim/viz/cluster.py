"""Check figure for a 2D cluster: roles of cells, escape line, receptors, and true-scale detail."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from scipy.spatial import cKDTree

from fpsim.config import Config
from fpsim.lattice import Cluster
from fpsim.receptors import ReceptorSet, coverage

SOURCE = ("#a6c8f4", "#1f4e9c")
TARGET = ("#f4a6a6", "#b22222")
OTHER = ("#e6e6e6", "#888888")
ESCAPE = "#d9480f"


def escape_mask(cluster: Cluster, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """True where the nearest lattice site is a ghost, i.e. a free molecule there is lost (L15)."""
    _, idx = cKDTree(cluster.centers).query(np.column_stack([X.ravel(), Y.ravel()]))
    return cluster.is_ghost[idx].reshape(X.shape)


def draw_cells(ax, cluster: Cluster, target_ring: int, labels=True):
    for i in range(cluster.n_sites):
        x, y = cluster.centers[i]
        if cluster.is_ghost[i]:
            ax.add_patch(Circle((x, y), cluster.cell_radius, fill=False, ls=":", ec="#bbbbbb", lw=0.7))
            continue
        face, edge = SOURCE if i == cluster.source else TARGET if cluster.ring[i] == target_ring else OTHER
        ax.add_patch(Circle((x, y), cluster.cell_radius, fc=face, ec=edge, lw=0.8))
        if labels:
            ax.text(x, y, "source" if i == cluster.source else str(cluster.ring[i]),
                    ha="center", va="center", fontsize=6.5, color="#333")


def plot_cluster_check(cfg: Config, cluster: Cluster, receptors: ReceptorSet, out):
    g, R = cfg.geometry, cluster.cell_radius
    a, delta = receptors.capture_radius, cfg.membrane.binding_layer
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # A. Whole cluster
    extent = (cluster.rings + 2) * cluster.spacing
    X, Y = np.meshgrid(np.linspace(-extent, extent, 1200), np.linspace(-extent, extent, 1200))
    lost = escape_mask(cluster, X, Y)
    ax_a.contourf(X, Y, lost.astype(float), levels=[0.5, 1.5], colors=["#fde0c5"])
    ax_a.contour(X, Y, lost.astype(float), levels=[0.5], colors=ESCAPE, linewidths=1.2)
    draw_cells(ax_a, cluster, g.target_ring)
    pts = receptors.positions.reshape(-1, 2)
    ax_a.plot(pts[:, 0], pts[:, 1], "k.", ms=3)
    ax_a.set_xlim(-extent, extent); ax_a.set_ylim(-extent, extent); ax_a.set_aspect("equal"); ax_a.axis("off")
    ax_a.plot([-extent + 5, -extent + 25], [-extent + 5] * 2, "k", lw=3)
    ax_a.text(-extent + 15, -extent + 8, "20 µm", ha="center", fontsize=8)
    ax_a.set_title(f"A. {cluster.n_cells} cells ({cluster.rings} rings); numbers = ring.\n"
                   f"Red = ring-{g.target_ring} targets ({len(receptors.cells)} cells); dotted = ghost ring; "
                   f"orange = escape region", fontsize=9.5)

    # B. One target cell and its receptors
    k = int(np.argmin(np.linalg.norm(cluster.centers[receptors.cells] - [extent, 0], axis=1)))
    cell = receptors.cells[k]
    cx, cy = cluster.centers[cell]
    draw_cells(ax_b, cluster, g.target_ring, labels=False)
    for j, (px, py) in enumerate(receptors.positions[k]):
        ax_b.plot(px, py, "o", ms=7, mfc="k", mec="white", zorder=5)
        ang = np.degrees(np.arctan2(py - cy, px - cx)) % 360
        ax_b.annotate(f"receptor {j}\n{ang:.0f}°", (px, py), xytext=(cx + 1.25 * (px - cx), cy + 1.25 * (py - cy)),
                      ha="center", va="center", fontsize=8)
    ax_b.set_xlim(cx - 1.6 * R, cx + 1.6 * R); ax_b.set_ylim(cy - 1.6 * R, cy + 1.6 * R); ax_b.set_aspect("equal")
    spacing_um = 2 * np.pi * R / receptors.n_per_cell
    ax_b.set_title(f"B. One target cell: {receptors.n_per_cell} receptors, {spacing_um:.1f} µm apart,\n"
                   f"pattern randomly rotated per cell (markers not to scale)", fontsize=9.5)
    ax_b.set_xlabel("µm")

    # C. True scale around one receptor (rotated so the membrane runs along the bottom)
    w = 6 * a
    ax_c.add_patch(Wedge((0, -R), R, 80, 100, fc=TARGET[0], ec=TARGET[1], lw=1.2))
    ax_c.add_patch(Wedge((0, -R), R + delta, 80, 100, width=delta, fc="#c7e9c0", ec="none", alpha=0.9))
    ax_c.add_patch(Circle((0, 0), a, fill=False, ec="k", lw=1.5, ls="--"))
    ax_c.plot(0, 0, "ko", ms=5)
    ax_c.set_xlim(-w, w); ax_c.set_ylim(-0.6 * w, 1.4 * w); ax_c.set_aspect("equal")
    ticks = np.arange(-0.05, 0.051, 0.01)
    ax_c.set_xticks(ticks, [f"{t*1e3:.0f}" for t in ticks])
    yt = np.arange(-0.03, 0.081, 0.01)
    ax_c.set_yticks(yt, [f"{t*1e3:.0f}" for t in yt])
    ax_c.set_xlabel("nm along membrane"); ax_c.set_ylabel("nm from membrane")
    ax_c.plot([], [], "s", color="#c7e9c0", ms=10, label=f"binding layer, {delta*1e3:.0f} nm (L28)")
    ax_c.plot([], [], "k--", label=f"capture zone, {a*1e3:.0f} nm from receptor center (L19)")
    ax_c.plot([], [], "s", color=TARGET[0], ms=10, label="cell interior")
    ax_c.legend(loc="upper center", fontsize=8)
    ax_c.set_title("C. True scale around one receptor:\nthe capture zone lies within the binding layer's reach", fontsize=9.5)

    cov = coverage(2, receptors.n_per_cell, a, R)
    fig.suptitle(f"{cfg.name}: cell radius {R} µm, gap {g.gap} µm, spacing {cluster.spacing} µm, "
                 f"receptors cover {cov*100:.3f}% of each target membrane", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
