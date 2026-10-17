"""Numba geometry queries for the 3D sphere lattice (HCP)."""

from __future__ import annotations

import math

from numba import njit


@njit(cache=True)
def nearest_cell3(x, y, z, guess, centers, neighbors, n_cells):
    """Nearest real cell to (x, y, z) and the distance to its center.

    Greedy walk over the 12-neighbor contact graph from ``guess``. In HCP each Voronoi
    face is shared with exactly one touching neighbor, so the contact graph is the
    Delaunay graph and the walk ends at the true nearest site. Ghosts are skipped.
    """
    best = guess
    dx, dy, dz = x - centers[best, 0], y - centers[best, 1], z - centers[best, 2]
    best_d2 = dx * dx + dy * dy + dz * dz
    improved = True
    while improved:
        improved = False
        cur = best
        for k in range(neighbors.shape[1]):
            j = neighbors[cur, k]
            if j < 0 or j >= n_cells:
                continue
            dx, dy, dz = x - centers[j, 0], y - centers[j, 1], z - centers[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best, best_d2 = j, d2
                improved = True
    return best, math.sqrt(best_d2)


@njit(cache=True)
def escape_test3(x, y, z, near, near_dist, centers, neighbors, n_cells, spacing):
    """(escaped, distance to the escape line); see geom2d.escape_test (L15)."""
    best = 1e300
    d2c = near_dist * near_dist
    for k in range(neighbors.shape[1]):
        g = neighbors[near, k]
        if g < n_cells:
            continue
        dx, dy, dz = x - centers[g, 0], y - centers[g, 1], z - centers[g, 2]
        signed = (dx * dx + dy * dy + dz * dz - d2c) / (2.0 * spacing)
        if signed < best:
            best = signed
    return best < 0.0, best


@njit(cache=True)
def to_local(rot, vx, vy, vz):
    """Rotate a world-frame vector into a target cell's unrotated receptor frame (R^T v)."""
    return (rot[0, 0] * vx + rot[1, 0] * vy + rot[2, 0] * vz,
            rot[0, 1] * vx + rot[1, 1] * vy + rot[2, 1] * vz,
            rot[0, 2] * vx + rot[1, 2] * vy + rot[2, 2] * vz)


@njit(cache=True)
def nearest_receptor(lx, ly, lz, base, grid_start, grid_index, n, box):
    """Nearest receptor to local point (lx, ly, lz), in unit-sphere units.

    Scans the 27 grid boxes around the point. Returns (index, distance) with the distance
    capped at ``box``; index -1 means no receptor within ``box``. The capped distance is
    always a lower bound on the true nearest distance (docs: receptors.ReceptorGrid).
    """
    i0 = min(max(int((lx + 1.0) / box), 0), n - 1)
    j0 = min(max(int((ly + 1.0) / box), 0), n - 1)
    k0 = min(max(int((lz + 1.0) / box), 0), n - 1)
    best_j, best_d2 = -1, box * box
    for i in range(max(i0 - 1, 0), min(i0 + 2, n)):
        for j in range(max(j0 - 1, 0), min(j0 + 2, n)):
            for k in range(max(k0 - 1, 0), min(k0 + 2, n)):
                b = (i * n + j) * n + k
                for q in range(grid_start[b], grid_start[b + 1]):
                    r = grid_index[q]
                    dx, dy, dz = lx - base[r, 0], ly - base[r, 1], lz - base[r, 2]
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < best_d2:
                        best_j, best_d2 = r, d2
    return best_j, math.sqrt(best_d2)


@njit(cache=True)
def sphere_step(ux, uy, uz, sigma, g1, g2, g3):
    """Diffuse a unit vector on the sphere: tangent-plane Gaussian step, then renormalize.

    ``sigma`` is the angular standard deviation per axis (sqrt(2 D dt) / R); g1..g3 are
    standard normal draws. Error per step is O(sigma²).
    """
    dot = g1 * ux + g2 * uy + g3 * uz
    tx, ty, tz = g1 - dot * ux, g2 - dot * uy, g3 - dot * uz
    vx, vy, vz = ux + sigma * tx, uy + sigma * ty, uz + sigma * tz
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    return vx / norm, vy / norm, vz / norm
