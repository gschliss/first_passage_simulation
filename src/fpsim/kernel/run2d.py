"""2D kernel: one loop per molecule, parallel over molecules.

Implements docs/architecture.md §3–4: free and bound states, the binding layer,
receptor capture (L17–L19) and escape (L15).

Event timing is exact, independent of the step size (L37):
  - unbinding: on binding, draw the bound dwell T ~ Exp(k_off); the bound period ends
    exactly at T (the last step is shortened to land on it);
  - binding: on unbinding, draw the exposure needed E ~ Exp(1); while in the layer,
    exposure accumulates at k_layer per second, and the molecule binds at the exact
    moment exposure reaches E.
Step size therefore only affects *where* molecules are (spatial accuracy), not when
events happen.

Capture:
  - bound: along the 1D circumference, with exact detection of steps that land in,
    jump over, or (Brownian bridge) brush a receptor's capture window;
  - free: within ``capture_radius`` of a receptor center. These zones lie inside the
    binding layer, where steps are already ~1 nm.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fpsim.kernel.common import BOUND, CAPTURED, ESCAPED, FREE, MODE_FIXED, TIMEOUT
from fpsim.kernel.geom2d import (
    bound_capture,
    escape_test,
    free_capture,
    nearest_cell,
    point_on_membrane,
    receptor_arc_gap,
    receptor_point_gap,
    reflect_out_of_cell,
)


@njit(parallel=True, cache=True)
def run_molecules_2d(
    seeds,  # (n,) per-molecule RNG seeds
    centers, neighbors, n_cells, source, ring, outer_ring, spacing,  # lattice
    receptor_phis, block_of_cell, capture_radius,  # receptors: (n_targets, n_per) angles; (n_sites,) row or -1
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
    receptor_id = np.full(n, -1, dtype=np.int64)
    start_xy = np.zeros((n, 2))
    end_xy = np.zeros((n, 2))
    time_free = np.zeros(n)
    n_trips = np.zeros(n, dtype=np.int64)
    n_cell_changes = np.zeros(n, dtype=np.int64)
    n_steps = np.zeros(n, dtype=np.int64)
    t_first_target = np.full(n, -1.0)  # first time in a target cell's layer or bound to it (L49)
    n_target_visits = np.zeros(n, dtype=np.int64)  # binding episodes on target cells (L49)

    n_frames = int(t_max / traj_interval) + 2
    traj = np.full((max(n_traj, 1), n_frames, 4), np.nan)  # x, y, state, cell

    # Smallest steps: resolve the layer and keep k_layer·dt small (L30); resolve receptors.
    # k_layer = k_off = 0 for free-only reference points (L47): no rate limit then.
    dt_layer = min((dist_frac * layer) ** 2 / (2.0 * d_free), rate_frac / k_layer if k_layer > 0.0 else 1e300)
    dt_bound = min(dt_cap, rate_frac / k_off if k_off > 0.0 else 1e300)
    half_width = capture_radius / radius  # receptor capture window, radians
    dt_receptor_min = (dist_frac * capture_radius) ** 2 / (2.0 * d_bound)
    dt_receptor_free_min = (dist_frac * capture_radius) ** 2 / (2.0 * d_free)

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
        result = TIMEOUT

        while t < t_max:
            n_steps[i] += 1
            if state == BOUND:
                block = block_of_cell[cell]
                if mode == MODE_FIXED:
                    dt = dt_fixed
                else:
                    dt = dt_bound
                    if block >= 0:  # slow down near receptors (L30)
                        gap = receptor_arc_gap(theta, receptor_phis[block], half_width) * radius
                        dt = min(dt, max((dist_frac * gap) ** 2 / (2.0 * d_bound), dt_receptor_min))
                dt = min(dt, t_unbind - t)
                sigma = math.sqrt(2.0 * d_bound * dt) / radius
                theta_new = theta + sigma * np.random.standard_normal()
                t += dt
                if block >= 0:
                    j = bound_capture(theta, theta_new, receptor_phis[block], half_width, sigma, np.random.random())
                    if j >= 0:
                        theta = receptor_phis[block, j]
                        x, y = point_on_membrane(cx, cy, theta, radius, 0.0)
                        receptor_id[i] = j
                        result = CAPTURED
                        break
                theta = theta_new
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
                    feature = height - layer
                    if ring[near] == outer_ring:
                        _, d_escape = escape_test(x, y, near, dist, centers, neighbors, n_cells, spacing)
                        feature = min(feature, d_escape)
                    dt = min(dt_cap, (dist_frac * feature) ** 2 / (2.0 * d_free))
                    dt = max(dt, dt_layer)
                block = block_of_cell[near]
                if mode != MODE_FIXED and block >= 0:  # slow down near receptors (L30, L48)
                    rgap = receptor_point_gap(x, y, centers[near, 0], centers[near, 1], receptor_phis[block],
                                              radius, capture_radius)
                    dt = min(dt, max((dist_frac * rgap) ** 2 / (2.0 * d_free), dt_receptor_free_min))
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
                    if block_of_cell[cell] >= 0:
                        if t_first_target[i] < 0.0:
                            t_first_target[i] = t
                        if block_of_cell[last_cell] < 0:
                            n_target_visits[i] += 1
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
                        dist = 2.0 * radius - dist
                    block = block_of_cell[near]
                    if block >= 0 and t_first_target[i] < 0.0 and dist - radius <= layer:
                        t_first_target[i] = t
                    if block >= 0 and dist - radius <= capture_radius:
                        j = free_capture(x, y, centers[near, 0], centers[near, 1], receptor_phis[block],
                                         radius, capture_radius)
                        if j >= 0:
                            receptor_id[i] = j
                            result = CAPTURED
                            break
                    if ring[near] == outer_ring:
                        escaped, _ = escape_test(x, y, near, dist, centers, neighbors, n_cells, spacing)
                        if escaped:
                            result = ESCAPED
                            break

            if i < n_traj:
                while frame < n_frames - 1 and frame * traj_interval <= t:
                    traj[i, frame, 0] = x
                    traj[i, frame, 1] = y
                    traj[i, frame, 2] = state
                    traj[i, frame, 3] = cell if state == BOUND else near
                    frame += 1

        if result == CAPTURED and state == FREE:
            cell = near
        outcome[i] = result
        t_end[i] = t  # for timeouts this may overshoot t_max by under one step
        final_state[i] = state
        final_cell[i] = cell if state == BOUND or result == CAPTURED else near
        end_xy[i, 0], end_xy[i, 1] = x, y
        if i < n_traj and frame < n_frames:  # final position, so animations show the end
            traj[i, frame, 0] = x
            traj[i, frame, 1] = y
            traj[i, frame, 2] = state
            traj[i, frame, 3] = final_cell[i]

    return (outcome, t_end, start_state, final_state, final_cell, receptor_id, start_xy, end_xy,
            time_free, n_trips, n_cell_changes, n_steps, traj[:n_traj], t_first_target, n_target_visits)
