"""Numba geometry queries for the 2D disk lattice."""

from __future__ import annotations

import math

from numba import njit


@njit(cache=True)
def nearest_cell(x, y, guess, centers, neighbors, n_cells):
    """Nearest real cell to (x, y) and the distance to its center.

    Greedy walk over the neighbor graph from ``guess`` (usually the previous answer,
    so this is O(1) per step). Ghost sites (index >= n_cells) are skipped: they are
    empty space, not cells.
    """
    best = guess
    dx, dy = x - centers[best, 0], y - centers[best, 1]
    best_d2 = dx * dx + dy * dy
    improved = True
    while improved:
        improved = False
        cur = best
        for k in range(neighbors.shape[1]):
            j = neighbors[cur, k]
            if j < 0 or j >= n_cells:
                continue
            dx, dy = x - centers[j, 0], y - centers[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best, best_d2 = j, d2
                improved = True
    return best, math.sqrt(best_d2)


@njit(cache=True)
def reflect_out_of_cell(x, y, cx, cy, dist, radius):
    """Radial reflection off a disk boundary: r -> 2R - r (L21)."""
    scale = (2.0 * radius - dist) / dist
    return cx + (x - cx) * scale, cy + (y - cy) * scale


@njit(cache=True)
def point_on_membrane(cx, cy, theta, radius, height):
    """Point at angle ``theta`` and ``height`` above the membrane of a cell."""
    r = radius + height
    return cx + r * math.cos(theta), cy + r * math.sin(theta)
