"""2D hexagonal (triangular) lattice of disks (L1)."""

from __future__ import annotations

import numpy as np

from fpsim.lattice.base import Cluster, build_cluster


def hex_points(extent: int, spacing: float) -> np.ndarray:
    """All lattice points within ``extent`` neighbor steps of the origin."""
    q, r = np.meshgrid(np.arange(-extent, extent + 1), np.arange(-extent, extent + 1))
    q, r = q.ravel(), r.ravel()
    keep = (np.abs(q) + np.abs(r) + np.abs(q + r)) // 2 <= extent
    q, r = q[keep], r[keep]
    return np.column_stack([spacing * (q + r / 2), spacing * np.sqrt(3) / 2 * r])


def hex_cluster(rings: int, cell_radius: float, spacing: float) -> Cluster:
    return build_cluster(hex_points(rings + 2, spacing), rings, cell_radius, spacing)
