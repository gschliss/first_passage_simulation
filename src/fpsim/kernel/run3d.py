"""3D kernel: one loop per molecule, parallel over molecules.

Same rules as run2d (see its docstring) on the HCP sphere lattice:
  - free: 3D diffusion in the gaps, reflecting off spheres; binds only within the
    layer, at the nearest membrane point; escape when the nearest site is a ghost;
  - bound: 2D diffusion on the sphere surface (tangent step + renormalize);
  - capture within ``capture_radius`` of a receptor center, from either state, on the
    ``n_finds``-th find (L86): after each earlier find the molecule carries on, and the
    next find counts only once it has been more than ``rearm`` from the receptor it last
    found (any receptor, including that one, can then be found again).
Event timing is exact (L37). Receptors are looked up in each target cell's unrotated
frame using one shared grid; near a receptor, bound steps shrink so each move is
≤ dist_frac × the remaining gap (down to dist_frac × capture radius).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fpsim.kernel.common import BOUND, CAPTURED, ESCAPED, FREE, MODE_FIXED, TIMEOUT
from fpsim.kernel.geom3d import escape_test3, nearest_cell3, nearest_receptor, sphere_step, to_local


@njit(cache=True)
def receptor_world(rot, base, j, cx, cy, cz, radius):
    """World position of receptor j on a cell with rotation ``rot`` (inverse of to_local: R v)."""
    bx, by, bz = base[j, 0], base[j, 1], base[j, 2]
    return (cx + radius * (rot[0, 0] * bx + rot[0, 1] * by + rot[0, 2] * bz),
            cy + radius * (rot[1, 0] * bx + rot[1, 1] * by + rot[1, 2] * bz),
            cz + radius * (rot[2, 0] * bx + rot[2, 1] * by + rot[2, 2] * bz))


@njit(parallel=True, cache=True)
def run_molecules_3d(
    seeds,
    centers, neighbors, n_cells, source, ring, outer_ring, spacing,  # lattice
    rotations, block_of_cell, base, grid_start, grid_index, grid_n, grid_box, capture_radius,  # receptors
    radius, layer,
    d_free, d_bound,
    free_fraction,
    k_off, k_layer,
    mode, dt_fixed, dist_frac, rate_frac, dt_cap, rec_frac,
    t_max,
    n_traj, traj_interval,
    n_finds, rearm,
):
    n = seeds.shape[0]
    outcome = np.full(n, TIMEOUT, dtype=np.int8)
    t_end = np.zeros(n)
    start_state = np.zeros(n, dtype=np.int8)
    final_state = np.zeros(n, dtype=np.int8)
    final_cell = np.zeros(n, dtype=np.int64)
    receptor_id = np.full(n, -1, dtype=np.int64)
    start_xyz = np.zeros((n, 3))
    end_xyz = np.zeros((n, 3))
    time_free = np.zeros(n)
    n_trips = np.zeros(n, dtype=np.int64)
    n_cell_changes = np.zeros(n, dtype=np.int64)
    n_steps = np.zeros(n, dtype=np.int64)
    t_first_target = np.full(n, -1.0)  # first time in a target cell's layer or bound to it (L49)
    n_target_visits = np.zeros(n, dtype=np.int64)  # binding episodes on target cells (L49)
    t_first_find = np.full(n, -1.0)  # time of the first receptor find (L86)
    finds = np.zeros(n, dtype=np.int64)
    rearm2 = rearm * rearm

    n_frames = int(t_max / traj_interval) + 2
    traj = np.full((max(n_traj, 1), n_frames, 5), np.nan)  # x, y, z, state, cell

    # k_layer = k_off = 0 for free-only reference points (L47): no rate limit then.
    dt_layer = min((dist_frac * layer) ** 2 / (2.0 * d_free), rate_frac / k_layer if k_layer > 0.0 else 1e300)
    dt_bound = min(dt_cap, rate_frac / k_off if k_off > 0.0 else 1e300)
    a_unit = capture_radius / radius  # capture radius in unit-sphere units
    dt_receptor_min = (dist_frac * capture_radius) ** 2 / (2.0 * d_bound)
    dt_receptor_free_min = (rec_frac * capture_radius) ** 2 / (2.0 * d_free)  # L71

    for i in prange(n):
        np.random.seed(seeds[i])

        # Start uniform on the source sphere; 5% (the point's f) free just off it (L16, L25).
        cell = source
        cx, cy, cz = centers[cell, 0], centers[cell, 1], centers[cell, 2]
        g1, g2, g3 = np.random.standard_normal(), np.random.standard_normal(), np.random.standard_normal()
        norm = math.sqrt(g1 * g1 + g2 * g2 + g3 * g3)
        ux, uy, uz = g1 / norm, g2 / norm, g3 / norm
        t = 0.0
        t_unbind = 0.0
        exposure, exposure_needed = 0.0, 0.0
        if np.random.random() < free_fraction:
            state = FREE
            h = radius + np.random.uniform(0.0, layer)
            exposure_needed = np.random.exponential(1.0)
        else:
            state = BOUND
            h = radius
            t_unbind = np.random.exponential(1.0 / k_off)
        x, y, z = cx + h * ux, cy + h * uy, cz + h * uz
        start_xyz[i, 0], start_xyz[i, 1], start_xyz[i, 2] = x, y, z
        start_state[i] = state
        near = cell
        last_cell = cell
        frame = 0
        result = TIMEOUT
        armed = True
        rx, ry, rz = 0.0, 0.0, 0.0  # world position of the receptor found last

        while t < t_max:
            n_steps[i] += 1
            if not armed and (x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2 > rearm2:
                armed = True
            if state == BOUND:
                block = block_of_cell[cell]
                if block >= 0:
                    lx, ly, lz = to_local(rotations[block], ux, uy, uz)
                    j, d = nearest_receptor(lx, ly, lz, base, grid_start, grid_index, grid_n, grid_box)
                    if armed and d <= a_unit:
                        finds[i] += 1
                        if t_first_find[i] < 0.0:
                            t_first_find[i] = t
                        receptor_id[i] = j
                        if finds[i] >= n_finds:
                            result = CAPTURED
                            break
                        armed = False
                        rx, ry, rz = receptor_world(rotations[block], base, j, cx, cy, cz, radius)
                if mode == MODE_FIXED:
                    dt = dt_fixed
                else:
                    dt = dt_bound
                    if block >= 0:  # slow down near receptors (L30); d is a lower bound
                        gap = (d - a_unit) * radius
                        dt = min(dt, max((dist_frac * gap) ** 2 / (2.0 * d_bound), dt_receptor_min))
                dt = min(dt, t_unbind - t)
                sigma = math.sqrt(2.0 * d_bound * dt) / radius
                ux, uy, uz = sphere_step(ux, uy, uz, sigma, np.random.standard_normal(),
                                         np.random.standard_normal(), np.random.standard_normal())
                t += dt
                if t >= t_unbind:
                    h = radius + np.random.uniform(0.0, layer)  # unbind (L25, L28)
                    state = FREE
                    near = cell
                    n_trips[i] += 1
                    exposure, exposure_needed = 0.0, np.random.exponential(1.0)
                else:
                    h = radius
                x, y, z = cx + h * ux, cy + h * uy, cz + h * uz
            else:
                near, dist = nearest_cell3(x, y, z, near, centers, neighbors, n_cells)
                height = dist - radius
                in_layer = height <= layer
                if mode == MODE_FIXED:
                    dt = dt_fixed
                elif in_layer:
                    dt = dt_layer
                else:
                    feature = height - layer
                    if ring[near] == outer_ring:
                        _, d_escape = escape_test3(x, y, z, near, dist, centers, neighbors, n_cells, spacing)
                        feature = min(feature, d_escape)
                    dt = min(dt_cap, (dist_frac * feature) ** 2 / (2.0 * d_free))
                    dt = max(dt, dt_layer)
                block = block_of_cell[near]
                if mode != MODE_FIXED and block >= 0 and height < grid_box * radius:  # near receptors (L30, L48)
                    lx, ly, lz = to_local(rotations[block], (x - centers[near, 0]) / radius,
                                          (y - centers[near, 1]) / radius, (z - centers[near, 2]) / radius)
                    _, d = nearest_receptor(lx, ly, lz, base, grid_start, grid_index, grid_n, grid_box)
                    rgap = max(d * radius - capture_radius, 0.0)  # d is a lower bound, so this is too
                    dt = min(dt, max((rec_frac * rgap) ** 2 / (2.0 * d_free), dt_receptor_free_min))
                if in_layer and exposure + k_layer * dt >= exposure_needed:
                    dt = (exposure_needed - exposure) / k_layer
                    time_free[i] += dt
                    t += dt
                    # Bind at the nearest point of the nearest membrane (L22).
                    cell = near
                    cx, cy, cz = centers[cell, 0], centers[cell, 1], centers[cell, 2]
                    ux, uy, uz = (x - cx) / dist, (y - cy) / dist, (z - cz) / dist
                    x, y, z = cx + radius * ux, cy + radius * uy, cz + radius * uz
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
                    z += s * np.random.standard_normal()
                    near, dist = nearest_cell3(x, y, z, near, centers, neighbors, n_cells)
                    if dist < radius:  # radial reflection off the sphere (L21)
                        scale = (2.0 * radius - dist) / dist
                        x = centers[near, 0] + (x - centers[near, 0]) * scale
                        y = centers[near, 1] + (y - centers[near, 1]) * scale
                        z = centers[near, 2] + (z - centers[near, 2]) * scale
                        dist = 2.0 * radius - dist
                    block = block_of_cell[near]
                    if block >= 0 and t_first_target[i] < 0.0 and dist - radius <= layer:
                        t_first_target[i] = t
                    if block >= 0 and dist - radius <= capture_radius:
                        lx, ly, lz = to_local(rotations[block], (x - centers[near, 0]) / radius,
                                              (y - centers[near, 1]) / radius, (z - centers[near, 2]) / radius)
                        j, d = nearest_receptor(lx, ly, lz, base, grid_start, grid_index, grid_n, grid_box)
                        if armed and d <= a_unit:
                            finds[i] += 1
                            if t_first_find[i] < 0.0:
                                t_first_find[i] = t
                            receptor_id[i] = j
                            if finds[i] >= n_finds:
                                result = CAPTURED
                                break
                            armed = False
                            rx, ry, rz = receptor_world(rotations[block], base, j, centers[near, 0], centers[near, 1],
                                                        centers[near, 2], radius)
                    if ring[near] == outer_ring:
                        escaped, _ = escape_test3(x, y, z, near, dist, centers, neighbors, n_cells, spacing)
                        if escaped:
                            result = ESCAPED
                            break

            if i < n_traj:
                while frame < n_frames - 1 and frame * traj_interval <= t:
                    traj[i, frame, 0] = x
                    traj[i, frame, 1] = y
                    traj[i, frame, 2] = z
                    traj[i, frame, 3] = state
                    traj[i, frame, 4] = cell if state == BOUND else near
                    frame += 1

        if result == CAPTURED and state == FREE:
            cell = near
        outcome[i] = result
        t_end[i] = t
        final_state[i] = state
        final_cell[i] = cell if state == BOUND or result == CAPTURED else near
        end_xyz[i, 0], end_xyz[i, 1], end_xyz[i, 2] = x, y, z
        if i < n_traj and frame < n_frames:
            traj[i, frame, 0] = x
            traj[i, frame, 1] = y
            traj[i, frame, 2] = z
            traj[i, frame, 3] = state
            traj[i, frame, 4] = final_cell[i]

    return (outcome, t_end, start_state, final_state, final_cell, receptor_id, start_xyz, end_xyz,
            time_free, n_trips, n_cell_changes, n_steps, traj[:n_traj], t_first_target, n_target_visits,
            t_first_find, finds)
