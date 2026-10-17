"""Baseline geometry: absorbing inner ball inside a reflecting outer ball.

This works in any dimension (d=1 is an interval, d=2 a disk, d=3 a sphere)
and has a closed-form MFPT (see ``fpsim.analysis.theory``), so it is our
validation case before moving to bio-inspired geometries.
"""

from __future__ import annotations

import numpy as np

from fpsim.geometry.base import Geometry


class SphericalShell(Geometry):
    """Target of radius ``a`` at the origin; reflecting wall at radius ``R``.

    Agents start uniformly on the sphere of radius ``r0`` (a < r0 <= R).
    """

    def __init__(self, dim: int, R: float, a: float, r0: float):
        super().__init__(dim)
        if not 0 < a < r0 <= R:
            raise ValueError(f"need 0 < a < r0 <= R, got a={a}, r0={r0}, R={R}")
        self.R, self.a, self.r0 = R, a, r0

    def sample_start(self, n: int, rng: np.random.Generator) -> np.ndarray:
        directions = rng.standard_normal((n, self.dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return self.r0 * directions

    def apply_boundaries(self, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        # Radial specular reflection off the outer wall: r -> 2R - r.
        r = np.linalg.norm(x_new, axis=1)
        outside = r > self.R
        if np.any(outside):
            scale = (2 * self.R - r[outside]) / r[outside]
            x_new[outside] *= scale[:, None]
        return x_new

    def is_absorbed(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x, axis=1) <= self.a
