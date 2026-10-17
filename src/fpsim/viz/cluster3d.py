"""Check figure for a 3D (HCP) cluster: roles of cells, a cross-section, receptors, true-scale detail."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from scipy.spatial import cKDTree

from fpsim.config import Config
from fpsim.lattice import Cluster
from fpsim.receptors import ReceptorSet, coverage
from fpsim.viz.cluster import ESCAPE, OTHER, SOURCE, TARGET


def plot_cluster_check_3d(cfg: Config, cluster: Cluster, receptors: ReceptorSet, out):
    g, R = cfg.geometry, cluster.cell_radius
    a, delta = receptors.capture_radius, cfg.membrane.binding_layer
    real = ~cluster.is_ghost
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1])

    # A. Cells in 3D, by role
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    c = cluster.centers
    roles = [(real & (cluster.ring != g.target_ring) & (np.arange(cluster.n_sites) != cluster.source), OTHER[1], 8, 0.15, "other cells"),
             (cluster.is_ghost, "#dddddd", 3, 0.25, "ghost sites (escape rule)"),
             (cluster.ring == g.target_ring, TARGET[1], 30, 0.9, f"ring-{g.target_ring} targets ({len(receptors.cells)})"),
             (np.arange(cluster.n_sites) == cluster.source, SOURCE[1], 120, 1.0, "source")]
    for mask, color, size, alpha, label in roles:
        ax.scatter(c[mask, 0], c[mask, 1], c[mask, 2], s=size, c=color, alpha=alpha, label=label, depthshade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"A. {cluster.n_cells} cells ({cluster.rings} neighbor-step rings, HCP)\n"
                 "dots = cell centers (not to scale)", fontsize=10)

    # B. Cross-section through the source layer (z = 0)
    ax = fig.add_subplot(gs[0, 1])
    extent = (cluster.rings + 1.5) * cluster.spacing
    X, Y = np.meshgrid(np.linspace(-extent, extent, 900), np.linspace(-extent, extent, 900))
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    _, idx = cKDTree(c).query(pts)
    lost = cluster.is_ghost[idx].reshape(X.shape)
    ax.contourf(X, Y, lost.astype(float), levels=[0.5, 1.5], colors=["#fde0c5"])
    ax.contour(X, Y, lost.astype(float), levels=[0.5], colors=ESCAPE, linewidths=1)
    for i in np.flatnonzero(real & (np.abs(c[:, 2]) < R)):
        r_cut = np.sqrt(R**2 - c[i, 2] ** 2)
        face, edge = SOURCE if i == cluster.source else TARGET if cluster.ring[i] == g.target_ring else OTHER
        ax.add_patch(Circle(c[i, :2], r_cut, fc=face, ec=edge, lw=0.6))
        if c[i, 2] == 0:
            ax.text(*c[i, :2], "src" if i == cluster.source else str(cluster.ring[i]), ha="center", va="center", fontsize=6)
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent); ax.set_aspect("equal")
    ax.set_title("B. Slice through the source layer (z = 0); numbers = ring. Layers are\n"
                 f"{cluster.spacing * np.sqrt(2 / 3):.1f} µm apart, so only this layer's cells are cut. Orange = escape region.",
                 fontsize=10)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

    # C. One target cell's receptors
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    d = receptors.directions[0]
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=1.5, c=d[:, 2], cmap="viridis", depthshade=True)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    nn = cKDTree(d).query(d, k=2)[0][:, 1] * R
    ax.set_title(f"C. One target cell: {receptors.n_per_cell} receptors (Fibonacci sphere,\n"
                 f"randomly rotated per cell); spacing {nn.mean():.2f} µm ({nn.min():.2f}–{nn.max():.2f})", fontsize=10)

    # D. True scale around one receptor (cross-section through it)
    ax = fig.add_subplot(gs[1, 1:])
    w = 6 * a
    ax.add_patch(Wedge((0, -R), R, 80, 100, fc=TARGET[0], ec=TARGET[1], lw=1.2))
    ax.add_patch(Wedge((0, -R), R + delta, 80, 100, width=delta, fc="#c7e9c0", ec="none", alpha=0.9))
    ax.add_patch(Circle((0, 0), a, fill=False, ec="k", lw=1.5, ls="--"))
    ax.plot(0, 0, "ko", ms=5)
    ax.set_xlim(-w, w); ax.set_ylim(-0.6 * w, 1.4 * w); ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.05, 0.051, 0.01), [f"{t * 1e3:.0f}" for t in np.arange(-0.05, 0.051, 0.01)])
    ax.set_yticks(np.arange(-0.03, 0.081, 0.01), [f"{t * 1e3:.0f}" for t in np.arange(-0.03, 0.081, 0.01)])
    ax.set_xlabel("nm along membrane"); ax.set_ylabel("nm from membrane")
    ax.plot([], [], "s", color="#c7e9c0", ms=10, label=f"binding layer, {delta * 1e3:.0f} nm (L28)")
    ax.plot([], [], "k--", label=f"capture zone (a sphere), {a * 1e3:.0f} nm from receptor center (L19)")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_title("D. True scale, cross-section through one receptor", fontsize=10)

    cov = coverage(3, receptors.n_per_cell, a, R)
    fig.suptitle(f"{cfg.name}: sphere radius {R} µm, gap {g.gap} µm, spacing {cluster.spacing} µm; "
                 f"receptors cover {cov * 100:.3f}% of each target membrane", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
