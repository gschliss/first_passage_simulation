"""Free space vs binding-layer space, per cell of the infinite close-packed lattice (L29).

``k_layer = k_on · V_free / V_layer`` keeps the free fraction exact (L28): free molecules
are uniform in free space at equilibrium, so the average binding rate per free molecule
is k_layer × (fraction of free space inside the layer) = k_on.
"""

from __future__ import annotations

import math


def site_volume(dim: int, spacing: float) -> float:
    """Area (2D hexagonal) or volume (3D close packing, HCP or FCC) per lattice site."""
    if dim == 2:
        return math.sqrt(3) / 2 * spacing**2
    if dim == 3:
        return spacing**3 / math.sqrt(2)
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def ball_volume(dim: int, radius: float) -> float:
    return math.pi * radius**2 if dim == 2 else 4 / 3 * math.pi * radius**3


def layer_factor(dim: int, cell_radius: float, spacing: float, layer: float) -> float:
    """V_free / V_layer per cell. Requires the layers of facing cells not to overlap."""
    gap = spacing - 2 * cell_radius
    if not 0 < layer < gap / 2:
        raise ValueError("binding layer must be thinner than half the gap")
    v_free = site_volume(dim, spacing) - ball_volume(dim, cell_radius)
    v_layer = ball_volume(dim, cell_radius + layer) - ball_volume(dim, cell_radius)
    return v_free / v_layer
