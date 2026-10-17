"""3D hexagonal close packing of spheres (L1, L6).

Hexagonal layers stacked A-B-A-B: A layers have a site at the origin; B layers are
shifted by (s/2, s/(2√3)) and sit h = s·√(2/3) above/below. Every sphere touches 12
neighbors at distance s. Rings are counted in neighbor steps (L26).
"""

from __future__ import annotations

import numpy as np

from fpsim.lattice.base import Cluster, build_cluster


def hcp_points(extent: int, spacing: float) -> np.ndarray:
    """All HCP sites within ``extent`` spacings of the origin (a superset of ``extent`` neighbor steps)."""
    s = spacing
    h = s * np.sqrt(2 / 3)
    n_xy = extent + 2
    n_z = int(np.ceil(extent * s / h)) + 1
    q, r = np.meshgrid(np.arange(-n_xy, n_xy + 1), np.arange(-n_xy, n_xy + 1))
    base = np.column_stack([s * (q.ravel() + r.ravel() / 2), s * np.sqrt(3) / 2 * r.ravel()])
    shift = np.array([s / 2, s / (2 * np.sqrt(3))])
    layers = []
    for k in range(-n_z, n_z + 1):
        xy = base + (shift if k % 2 else 0.0)
        layers.append(np.column_stack([xy, np.full(len(xy), k * h)]))
    pts = np.concatenate(layers)
    return pts[np.linalg.norm(pts, axis=1) <= extent * s * (1 + 1e-9)]


def hcp_cluster(rings: int, cell_radius: float, spacing: float) -> Cluster:
    # Neighbor-step distance k reaches at most k spacings, so a ball of rings + 2 spacings suffices.
    return build_cluster(hcp_points(rings + 2, spacing), rings, cell_radius, spacing)
