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


@njit(cache=True)
def wrap_angle(a):
    """Map an angle to (-pi, pi]."""
    two_pi = 2.0 * math.pi
    return a - two_pi * math.ceil((a - math.pi) / two_pi)


@njit(cache=True)
def receptor_arc_gap(theta, phis, half_width):
    """Smallest angular gap from ``theta`` to the edge of any receptor's capture window.

    The window around receptor angle phi is [phi - half_width, phi + half_width];
    the gap is 0 when ``theta`` is inside one.
    """
    best = 1e300
    for j in range(phis.shape[0]):
        g = abs(wrap_angle(theta - phis[j])) - half_width
        if g < best:
            best = g
    return max(best, 0.0)


@njit(cache=True)
def bound_capture(theta_old, theta_new, phis, half_width, sigma, u):
    """Receptor index hit by a bound molecule moving theta_old -> theta_new, or -1.

    1D along the circumference (angles in radians), so detection is essentially exact:
      - ends inside a window (or started inside one);
      - jumped clear across a window during the step;
      - otherwise both ends lie on the same side: the path still touched the window
        with the Brownian-bridge probability exp(-2 g_old g_new / sigma^2), tested
        against the uniform draw ``u``. sigma is the step's angular standard deviation.
    """
    step = theta_new - theta_old
    for j in range(phis.shape[0]):
        d_old = wrap_angle(theta_old - phis[j])
        d_new = d_old + step
        if abs(d_old) <= half_width or abs(d_new) <= half_width:
            return j
        if (d_old > 0.0) != (d_new > 0.0) and abs(step) < math.pi:
            return j
        g_old, g_new = abs(d_old) - half_width, abs(d_new) - half_width
        if sigma > 0.0 and u < math.exp(-2.0 * g_old * g_new / (sigma * sigma)):
            return j
    return -1


@njit(cache=True)
def free_capture(x, y, cx, cy, phis, radius, capture_radius):
    """Receptor index within ``capture_radius`` of a free molecule at (x, y), or -1."""
    r2 = capture_radius * capture_radius
    for j in range(phis.shape[0]):
        dx = x - (cx + radius * math.cos(phis[j]))
        dy = y - (cy + radius * math.sin(phis[j]))
        if dx * dx + dy * dy <= r2:
            return j
    return -1


@njit(cache=True)
def escape_test(x, y, near, near_dist, centers, neighbors, n_cells, spacing):
    """(escaped, distance to the escape line) for a free molecule whose nearest cell is ``near``.

    Escape (L15): the nearest lattice site is a ghost (index >= n_cells). Only cells in
    the outer ring have ghost neighbors; elsewhere the escape line is far away, so the
    distance returned is a large number. The distance to the bisector between the
    nearest cell c and a ghost g is (|p-g|² - |p-c|²) / (2|g-c|).
    """
    best = 1e300
    d2c = near_dist * near_dist
    for k in range(neighbors.shape[1]):
        g = neighbors[near, k]
        if g < n_cells:  # -1 (none) or a real cell
            continue
        dx, dy = x - centers[g, 0], y - centers[g, 1]
        signed = (dx * dx + dy * dy - d2c) / (2.0 * spacing)
        if signed < best:
            best = signed
    return best < 0.0, best


@njit(cache=True)
def receptor_point_gap(x, y, cx, cy, phis, radius, capture_radius):
    """Distance from (x, y) to the nearest receptor's capture zone (0 if inside one)."""
    best = 1e300
    for j in range(phis.shape[0]):
        dx = x - (cx + radius * math.cos(phis[j]))
        dy = y - (cy + radius * math.sin(phis[j]))
        d = math.sqrt(dx * dx + dy * dy)
        if d < best:
            best = d
    return max(best - capture_radius, 0.0)
