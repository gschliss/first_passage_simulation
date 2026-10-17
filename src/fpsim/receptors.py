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
        base = 2 * math.pi * np.arange(n_per_cell) / n_per_cell
        offsets = rng.uniform(0, 2 * math.pi / n_per_cell, size=len(cells))
        angles = base[None, :] + offsets[:, None]
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    else:
        base = fibonacci_sphere(n_per_cell)
        rotations = Rotation.random(len(cells), random_state=rng)
        directions = np.stack([rot.apply(base) for rot in rotations])

    positions = cluster.centers[cells][:, None, :] + cluster.cell_radius * directions
    block_of_cell = np.full(cluster.n_sites, -1, dtype=np.int64)
    block_of_cell[cells] = np.arange(len(cells))
    return ReceptorSet(cells=cells, directions=directions, positions=positions,
                       block_of_cell=block_of_cell, capture_radius=capture_radius)


def coverage(dim: int, n_per_cell: int, capture_radius: float, cell_radius: float) -> float:
    """Fraction of the membrane within capture range of a receptor (L18)."""
    if dim == 2:
        return n_per_cell * 2 * capture_radius / (2 * math.pi * cell_radius)
    return n_per_cell * math.pi * capture_radius**2 / (4 * math.pi * cell_radius**2)
