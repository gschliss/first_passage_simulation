"""Isolated-sphere checks of 3D capture against theory (L69).

Two small Numba loops that reuse the 3D kernel's geometry, receptor lookup and step
rules (kernel/geom3d.py, run3d.py), on a single cell with no lattice:

  free_isolated   free-only walkers start at r0 outside a sphere of radius R carrying
                  N receptors, reflect off it, and are lost at R_out; outcome =
                  captured or lost. Compared with Berg–Purcell (below).
  bound_isolated  walkers bound to the sphere that never unbind, diffusing on its
                  surface until they reach a receptor. Compared with the
                  narrow-escape cell approximation (below).

Theory:
  Berg–Purcell: N small sinks on a reflecting sphere act like a uniformly partially
  absorbing surface with trapping rate κ = D·N·s/(π R²), where s is a sink's
  "size" defined by its current in a half space, I = 4·D·s·c. A capture zone that is
  a hemisphere of radius a (points within a of a receptor on the membrane) carries
  I = 2π·D·a·c, so s = π·a/2. The splitting probability u(r) to be captured before
  reaching R_out then solves Laplace's equation with u(R_out) = 0 and the Robin
  condition −D u'(R) = κ (1 − u(R)).
  Narrow escape on the sphere: each of N evenly spread receptors owns a patch of area
  4πR²/N, approximated by a flat disk of radius L = √(4R²/N); a walker starting uniformly
  in an annulus between the capture radius a and a reflecting edge at L has mean
  first-passage time L²/(2D) · (ln(L/a) − 3/4) for L ≫ a.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fpsim.kernel.geom3d import nearest_receptor, sphere_step


def bp_capture_probability(R, R_out, r0, N, a, D=1.0):
    """Berg–Purcell splitting probability (independent of D)."""
    s = math.pi * a / 2
    kappa = D * N * s / (math.pi * R**2)
    A = kappa / (D / R**2 + kappa * (1 / R - 1 / R_out))
    return A * (1 / r0 - 1 / R_out)


def perfect_capture_probability(R, R_out, r0):
    """Splitting probability for a fully absorbing sphere (upper bound)."""
    return (1 / r0 - 1 / R_out) / (1 / R - 1 / R_out)


def narrow_escape_time(R, N, a, D):
    L = math.sqrt(4 * R**2 / N)
    return L**2 / (2 * D) * (math.log(L / a) - 0.75)


@njit(parallel=True, cache=True)
def free_isolated(seeds, R, R_out, r0, d_free, layer, capture_radius, base, grid_start, grid_index, grid_n, grid_box,
                  fixed, dt_fixed, dist_frac, dt_cap, rho=0.0, rec_frac=-1.0):
    """rho > 0: start uniformly within lateral distance rho (µm) of receptor 0, at radius r0;
    rho = 0: start uniformly over the sphere of radius r0."""
    n = seeds.shape[0]
    captured = np.zeros(n, dtype=np.bool_)
    t_end = np.zeros(n)
    n_steps = np.zeros(n, dtype=np.int64)
    a_unit = capture_radius / R
    dt_layer = (dist_frac * layer) ** 2 / (2.0 * d_free)
    rf = dist_frac if rec_frac <= 0.0 else rec_frac  # receptor rule fraction (L71); default = dist_frac
    dt_rec_min = (rf * capture_radius) ** 2 / (2.0 * d_free)
    for i in prange(n):
        np.random.seed(seeds[i])
        if rho > 0.0:
            bx, by, bz = base[0, 0], base[0, 1], base[0, 2]
            # tangent basis at receptor 0
            ex, ey, ez = (1.0, 0.0, 0.0) if abs(bx) < 0.9 else (0.0, 1.0, 0.0)
            dot = ex * bx + ey * by + ez * bz
            ex, ey, ez = ex - dot * bx, ey - dot * by, ez - dot * bz
            ne = math.sqrt(ex * ex + ey * ey + ez * ez)
            ex, ey, ez = ex / ne, ey / ne, ez / ne
            fx, fy, fz = by * ez - bz * ey, bz * ex - bx * ez, bx * ey - by * ex
            rr = rho * math.sqrt(np.random.random()) / R
            ph = 2.0 * math.pi * np.random.random()
            vx = bx + rr * (math.cos(ph) * ex + math.sin(ph) * fx)
            vy = by + rr * (math.cos(ph) * ey + math.sin(ph) * fy)
            vz = bz + rr * (math.cos(ph) * ez + math.sin(ph) * fz)
            nv = math.sqrt(vx * vx + vy * vy + vz * vz)
            x, y, z = r0 * vx / nv, r0 * vy / nv, r0 * vz / nv
        else:
            g1, g2, g3 = np.random.standard_normal(), np.random.standard_normal(), np.random.standard_normal()
            nrm = math.sqrt(g1 * g1 + g2 * g2 + g3 * g3)
            x, y, z = r0 * g1 / nrm, r0 * g2 / nrm, r0 * g3 / nrm
        t = 0.0
        while True:
            n_steps[i] += 1
            dist = math.sqrt(x * x + y * y + z * z)
            height = dist - R
            if fixed:
                dt = dt_fixed
            else:
                # Same rules as run3d's free state: layer floor, distance to the nearest
                # feature (here the layer edge or the loss sphere), slow-down near receptors.
                if height <= layer:
                    dt = dt_layer
                else:
                    feature = min(height - layer, R_out - dist)
                    dt = max(min(dt_cap, (dist_frac * feature) ** 2 / (2.0 * d_free)), dt_layer)
                if height < grid_box * R:
                    _, d = nearest_receptor(x / dist, y / dist, z / dist, base, grid_start, grid_index, grid_n, grid_box)
                    rgap = max(d * R - capture_radius, 0.0)
                    dt = min(dt, max((rf * rgap) ** 2 / (2.0 * d_free), dt_rec_min))
            s = math.sqrt(2.0 * d_free * dt)
            x += s * np.random.standard_normal()
            y += s * np.random.standard_normal()
            z += s * np.random.standard_normal()
            t += dt
            dist = math.sqrt(x * x + y * y + z * z)
            if dist < R:  # radial reflection, as in run3d
                scale = (2.0 * R - dist) / dist
                x, y, z = x * scale, y * scale, z * scale
                dist = 2.0 * R - dist
            if dist - R <= capture_radius:
                _, d = nearest_receptor(x / R, y / R, z / R, base, grid_start, grid_index, grid_n, grid_box)
                if d <= a_unit:
                    captured[i] = True
                    break
            if dist >= R_out:
                break
        t_end[i] = t
    return captured, t_end, n_steps


@njit(parallel=True, cache=True)
def bound_isolated(seeds, R, d_bound, capture_radius, base, grid_start, grid_index, grid_n, grid_box,
                   fixed, dt_fixed, dist_frac, dt_cap, t_max):
    n = seeds.shape[0]
    t_end = np.zeros(n)
    captured = np.zeros(n, dtype=np.bool_)
    n_steps = np.zeros(n, dtype=np.int64)
    a_unit = capture_radius / R
    dt_rec_min = (dist_frac * capture_radius) ** 2 / (2.0 * d_bound)
    for i in prange(n):
        np.random.seed(seeds[i])
        g1, g2, g3 = np.random.standard_normal(), np.random.standard_normal(), np.random.standard_normal()
        nrm = math.sqrt(g1 * g1 + g2 * g2 + g3 * g3)
        ux, uy, uz = g1 / nrm, g2 / nrm, g3 / nrm
        t = 0.0
        while t < t_max:
            n_steps[i] += 1
            _, d = nearest_receptor(ux, uy, uz, base, grid_start, grid_index, grid_n, grid_box)
            if d <= a_unit:
                captured[i] = True
                break
            if fixed:
                dt = dt_fixed
            else:  # same rule as run3d's bound state
                gap = (d - a_unit) * R
                dt = min(dt_cap, max((dist_frac * gap) ** 2 / (2.0 * d_bound), dt_rec_min))
            sigma = math.sqrt(2.0 * d_bound * dt) / R
            ux, uy, uz = sphere_step(ux, uy, uz, sigma, np.random.standard_normal(),
                                     np.random.standard_normal(), np.random.standard_normal())
            t += dt
        t_end[i] = t
    return captured, t_end, n_steps
