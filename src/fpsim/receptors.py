"""Receptor positions on the target cells (L17, L18, L36).

Each target cell gets ``n_per_cell`` evenly spaced receptors: equal angles in 2D, a
Fibonacci sphere in 3D. Each cell's pattern gets its own random rotation, drawn once
per job from the job's seed (L36). Receptors are points on the membrane; capture
happens within ``capture_radius`` of a receptor center (L19).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from fpsim.lattice import Cluster


@dataclass(frozen=True)
class ReceptorSet:
    cells: np.ndarray  # (n_targets,) site index of each target cell
    directions: np.ndarray  # (n_targets, n_per_cell, dim) unit vectors from cell center
    positions: np.ndarray  # (n_targets, n_per_cell, dim) absolute positions, µm
    block_of_cell: np.ndarray  # (n_sites,) row in the arrays above, or -1 if no receptors
    capture_radius: float
    base: np.ndarray  # (n_per_cell, dim) unrotated pattern, shared by every target cell
    rotations: np.ndarray  # (n_targets, dim, dim): directions[k] = base @ rotations[k].T

    @property
    def n_per_cell(self) -> int:
        return self.directions.shape[1]


def fibonacci_sphere(n: int) -> np.ndarray:
    """n nearly evenly spaced unit vectors (golden-angle spiral)."""
    i = np.arange(n)
    z = 1 - (2 * i + 1) / n
    rho = np.sqrt(1 - z**2)
    phi = i * math.pi * (3 - math.sqrt(5))
    return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])


def make_receptors(cluster: Cluster, target_ring: int, n_per_cell: int, capture_radius: float,
                   rng: np.random.Generator) -> ReceptorSet:
    cells = cluster.cells_in_ring(target_ring)
    if cluster.dim == 2:
        angles0 = 2 * math.pi * np.arange(n_per_cell) / n_per_cell
        base = np.column_stack([np.cos(angles0), np.sin(angles0)])
        offsets = rng.uniform(0, 2 * math.pi / max(n_per_cell, 1), size=len(cells))  # n_per_cell = 0: no receptors
        c, s_ = np.cos(offsets), np.sin(offsets)
        rotations = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], axis=-2)
    else:
        base = fibonacci_sphere(n_per_cell)
        rotations = Rotation.random(len(cells), random_state=rng).as_matrix()
    directions = np.einsum("kij,nj->kni", rotations, base)

    positions = cluster.centers[cells][:, None, :] + cluster.cell_radius * directions
    block_of_cell = np.full(cluster.n_sites, -1, dtype=np.int64)
    block_of_cell[cells] = np.arange(len(cells))
    return ReceptorSet(cells=cells, directions=directions, positions=positions,
                       block_of_cell=block_of_cell, capture_radius=capture_radius,
                       base=base, rotations=rotations)


@dataclass(frozen=True)
class ReceptorGrid:
    """Uniform 3D grid over [-1, 1]³ holding the unrotated unit-vector pattern (CSR layout).

    For a query direction, the 27 grid boxes around it contain every receptor within
    chord distance ``box`` (unit-sphere units), so "nearest found, capped at ``box``" is a
    guaranteed lower bound on the true nearest distance.
    """

    box: float
    n: int  # boxes per axis
    start: np.ndarray  # (n**3 + 1,) offsets into ``index``
    index: np.ndarray  # receptor indices sorted by box


def make_grid(base: np.ndarray, box: float = 0.05) -> ReceptorGrid:
    n = int(math.ceil(2.0 / box))
    ijk = np.clip(((base + 1.0) / box).astype(np.int64), 0, n - 1)
    flat = (ijk[:, 0] * n + ijk[:, 1]) * n + ijk[:, 2]
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n**3)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ReceptorGrid(box=box, n=n, start=start, index=order.astype(np.int64))


def coverage(dim: int, n_per_cell: int, capture_radius: float, cell_radius: float) -> float:
    """Fraction of the membrane within capture range of a receptor (L18)."""
    if dim == 2:
        return n_per_cell * 2 * capture_radius / (2 * math.pi * cell_radius)
    return n_per_cell * math.pi * capture_radius**2 / (4 * math.pi * cell_radius**2)
