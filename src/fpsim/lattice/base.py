"""A finite cluster of cells on a close-packed lattice, plus its ghost ring.

Rings are counted in neighbor steps from the source cell (L26), which works the same
on any lattice: build the touching-neighbor graph and breadth-first search from the
origin. The ring just outside the cluster is kept as **ghost sites**: empty lattice
positions used by the escape rule (a free molecule is lost once it is closer to a
ghost site than to any real cell, L15).

Sites are ordered real cells first (by ring), then ghosts, so ``centers[:n_cells]``
are the real cells.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Cluster:
    dim: int
    cell_radius: float
    spacing: float
    rings: int
    centers: np.ndarray  # (n_sites, dim), real cells then ghosts
    ring: np.ndarray  # (n_sites,) neighbor-step distance from the source
    neighbors: np.ndarray  # (n_sites, max_neighbors) site indices, -1 padded
    source: int  # index of the source cell (L12: at the cluster center)

    @property
    def n_sites(self) -> int:
        return len(self.centers)

    @property
    def n_cells(self) -> int:
        return int(np.sum(self.ring <= self.rings))

    @property
    def is_ghost(self) -> np.ndarray:
        return self.ring > self.rings

    def cells_in_ring(self, k: int) -> np.ndarray:
        if not 0 <= k <= self.rings:
            raise ValueError(f"ring {k} is outside the cluster (0..{self.rings})")
        return np.flatnonzero(self.ring == k)


def build_cluster(points: np.ndarray, rings: int, cell_radius: float, spacing: float) -> Cluster:
    """Keep lattice ``points`` within ``rings + 1`` neighbor steps of the point at the origin.

    ``points`` must include the origin and extend at least ``rings + 2`` steps out.
    """
    points = np.asarray(points, dtype=float)
    origin = int(np.argmin(np.linalg.norm(points, axis=1)))
    if np.linalg.norm(points[origin]) > 1e-9 * spacing:
        raise ValueError("lattice points must include the origin")

    tree = cKDTree(points)
    tol = 1e-6 * spacing
    nb_lists = tree.query_ball_point(points, spacing + tol)

    hops = np.full(len(points), -1)
    hops[origin] = 0
    queue = deque([origin])
    while queue:
        u = queue.popleft()
        if hops[u] > rings:  # ghosts are the last ring we need
            continue
        for v in nb_lists[u]:
            if hops[v] < 0:
                hops[v] = hops[u] + 1
                queue.append(v)

    keep = np.flatnonzero((hops >= 0) & (hops <= rings + 1))
    order = keep[np.lexsort((np.arange(len(keep)), hops[keep]))]
    new_index = np.full(len(points), -1)
    new_index[order] = np.arange(len(order))

    max_nb = max(len(nb_lists[i]) - 1 for i in order)
    neighbors = np.full((len(order), max_nb), -1, dtype=np.int64)
    for i, old in enumerate(order):
        nbs = [new_index[v] for v in nb_lists[old] if v != old and new_index[v] >= 0]
        neighbors[i, : len(nbs)] = sorted(nbs)

    # The farthest ghosts must be a complete ring, or the escape line would have holes.
    edge_hops = hops[order].max()
    if edge_hops != rings + 1:
        raise ValueError("lattice points do not extend far enough for the ghost ring")

    return Cluster(
        dim=points.shape[1],
        cell_radius=cell_radius,
        spacing=spacing,
        rings=rings,
        centers=points[order],
        ring=hops[order],
        neighbors=neighbors,
        source=int(new_index[origin]),
    )
