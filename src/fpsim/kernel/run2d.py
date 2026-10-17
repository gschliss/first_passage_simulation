"""2D kernel: one loop per molecule, parallel over molecules.

Implements docs/architecture.md §3–4 for the free and bound states and the binding
layer. Receptors (capture) and the escape line are added in build step 3.

Event timing is exact, independent of the step size:
  - unbinding: on binding, draw the bound dwell T ~ Exp(k_off); the bound period ends
    exactly at T (the last step is shortened to land on it);
  - binding: on unbinding, draw the exposure needed E ~ Exp(1); while in the layer,
    exposure accumulates at k_layer per second, and the molecule binds at the exact
    moment exposure reaches E.
Step size therefore only affects *where* molecules are (spatial accuracy), not when
events happen.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fpsim.kernel.common import BOUND, FREE, MODE_FIXED, TIMEOUT
from fpsim.kernel.geom2d import nearest_cell, point_on_membrane, reflect_out_of_cell


@njit(parallel=True, cache=True)
def run_molecules_2d(
    seeds,  # (n,) per-molecule RNG seeds
    centers, neighbors, n_cells, source,  # lattice
    radius, layer,  # µm: cell radius, binding layer δ
    d_free, d_bound,  # µm²/s
    free_fraction,  # initial split (L16)
    k_off, k_layer,  # s⁻¹
    mode, dt_fixed, dist_frac, rate_frac, dt_cap,  # stepping (L30, L34)
    t_max,
    n_traj, traj_interval,
):
    n = seeds.shape[0]
    outcome = np.full(n, TIMEOUT, dtype=np.int8)
    t_end = np.zeros(n)
    start_state = np.zeros(n, dtype=np.int8)
    final_state = np.zeros(n, dtype=np.int8)
    final_cell = np.zeros(n, dtype=np.int64)
    start_xy = np.zeros((n, 2))
    end_xy = np.zeros((n, 2))
    time_free = np.zeros(n)
    n_trips = np.zeros(n, dtype=np.int64)
    n_cell_changes = np.zeros(n, dtype=np.int64)
    n_steps = np.zeros(n, dtype=np.int64)

    n_frames = int(t_max / traj_interval) + 1
    traj = np.full((max(n_traj, 1), n_frames, 4), np.nan)  # x, y, state, cell

    # Smallest step: resolve the layer and keep k_layer·dt small (L30).
    dt_layer = min((dist_frac * layer) ** 2 / (2.0 * d_free), rate_frac / k_layer)
    dt_bound = min(dt_cap, rate_frac / k_off)

    for i in prange(n):
        np.random.seed(seeds[i])

        # Start on the source cell surface, uniform angle, 5% free just off it (L16, L25).
        cell = source
        cx, cy = centers[cell, 0], centers[cell, 1]
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        t = 0.0
        t_unbind = 0.0
        exposure, exposure_needed = 0.0, 0.0
        if np.random.random() < free_fraction:
            state = FREE
            x, y = point_on_membrane(cx, cy, theta, radius, np.random.uniform(0.0, layer))
            exposure_needed = np.random.exponential(1.0)
        else:
            state = BOUND
            x, y = point_on_membrane(cx, cy, theta, radius, 0.0)
            t_unbind = np.random.exponential(1.0 / k_off)
        start_xy[i, 0], start_xy[i, 1] = x, y
        start_state[i] = state
        near = cell
        last_cell = cell
        frame = 0

        while t < t_max:
            n_steps[i] += 1
            if state == BOUND:
                dt = min(dt_fixed if mode == MODE_FIXED else dt_bound, t_unbind - t)
                theta += math.sqrt(2.0 * d_bound * dt) / radius * np.random.standard_normal()
                t += dt
                if t >= t_unbind:
                    # Unbind: land at a uniform depth in the layer, same spot (L25, L28).
                    x, y = point_on_membrane(cx, cy, theta, radius, np.random.uniform(0.0, layer))
                    state = FREE
                    near = cell
                    n_trips[i] += 1
                    exposure, exposure_needed = 0.0, np.random.exponential(1.0)
                else:
                    x, y = point_on_membrane(cx, cy, theta, radius, 0.0)
            else:
                near, dist = nearest_cell(x, y, near, centers, neighbors, n_cells)
                height = dist - radius
                in_layer = height <= layer
                if mode == MODE_FIXED:
                    dt = dt_fixed
                elif in_layer:
                    dt = dt_layer
                else:
                    gap_to_layer = height - layer
                    dt = min(dt_cap, (dist_frac * gap_to_layer) ** 2 / (2.0 * d_free))
                    dt = max(dt, dt_layer)
                if in_layer and exposure + k_layer * dt >= exposure_needed:
                    # Binds partway through this step, at the exact moment exposure reaches E.
                    dt = (exposure_needed - exposure) / k_layer
                    time_free[i] += dt
                    t += dt
                    # Bind at the nearest point of the nearest membrane (L22).
                    cell = near
                    cx, cy = centers[cell, 0], centers[cell, 1]
                    theta = math.atan2(y - cy, x - cx)
                    x, y = point_on_membrane(cx, cy, theta, radius, 0.0)
                    state = BOUND
                    t_unbind = t + np.random.exponential(1.0 / k_off)
                    if cell != last_cell:
                        n_cell_changes[i] += 1
                    last_cell = cell
                else:
                    if in_layer:
                        exposure += k_layer * dt
                    time_free[i] += dt
                    t += dt
                    s = math.sqrt(2.0 * d_free * dt)
                    x += s * np.random.standard_normal()
                    y += s * np.random.standard_normal()
                    near, dist = nearest_cell(x, y, near, centers, neighbors, n_cells)
                    if dist < radius:
                        x, y = reflect_out_of_cell(x, y, centers[near, 0], centers[near, 1], dist, radius)

            if i < n_traj:
                while frame < n_frames and frame * traj_interval <= t:
                    traj[i, frame, 0] = x
                    traj[i, frame, 1] = y
                    traj[i, frame, 2] = state
                    traj[i, frame, 3] = cell if state == BOUND else near
                    frame += 1

        t_end[i] = t
        final_state[i] = state
        final_cell[i] = cell if state == BOUND else near
        end_xy[i, 0], end_xy[i, 1] = x, y

    return (outcome, t_end, start_state, final_state, final_cell, start_xy, end_xy,
            time_free, n_trips, n_cell_changes, n_steps, traj[:n_traj])
