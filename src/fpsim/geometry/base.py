"""Abstract interface every geometry must implement.

A geometry owns three things:
  1. where agents start          -> ``sample_start``
  2. what happens at walls       -> ``apply_boundaries`` (e.g. reflection)
  3. where the target is         -> ``is_absorbed``

Positions are always arrays of shape ``(n_agents, dim)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim

    @abstractmethod
    def sample_start(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return initial positions, shape (n, dim)."""

    @abstractmethod
    def apply_boundaries(self, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """Map proposed positions back into the domain (reflecting walls etc.)."""

    @abstractmethod
    def is_absorbed(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask, shape (n,), True where an agent has reached the target."""
