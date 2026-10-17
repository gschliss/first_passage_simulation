"""Brownian walkers, stored as a vectorized ensemble.

Each agent is one row of ``positions``. Keeping the ensemble as arrays
(rather than one Python object per agent) keeps large-N runs fast; per-agent
state such as diffusion mode (1D/2D/3D) can be added as extra arrays later.
"""

from __future__ import annotations

import numpy as np


class BrownianWalkers:
    """Overdamped Brownian motion: dx = sqrt(2 D dt) * xi, xi ~ N(0, I)."""

    def __init__(self, positions: np.ndarray, D: float):
        self.positions = np.asarray(positions, dtype=float)
        self.D = D

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    def propose_step(self, dt: float, rng: np.random.Generator, mask=None) -> np.ndarray:
        """Return proposed new positions for the agents selected by ``mask``."""
        x = self.positions if mask is None else self.positions[mask]
        return x + np.sqrt(2 * self.D * dt) * rng.standard_normal(x.shape)
