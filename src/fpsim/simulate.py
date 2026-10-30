"""Python entry point to the kernels: config in, per-molecule results out."""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpsim.config import Config
from fpsim.kernel.common import MODE_ADAPTIVE, MODE_FIXED, OUTCOME_NAMES, molecule_seeds
from fpsim.kernel.run2d import run_molecules_2d
from fpsim.kernel.run3d import run_molecules_3d
from fpsim.lattice import layer_factor, make_cluster
from fpsim.receptors import make_grid, make_receptors


@dataclass
class Result:
    molecules: pd.DataFrame  # one row per molecule (docs/architecture.md §6)
    trajectories: np.ndarray  # (n_traj, n_frames, dim + 2): position, state, cell
    meta: dict = field(default_factory=dict)


def strictest_dt(cfg: Config, k_layer: float, k_off: float, d_free: float | None = None) -> float:
    """Default fixed-mode step: the smallest adaptive limit at this sweep point (L34)."""
    s, d = cfg.stepping, cfg.diffusion
    rate = lambda k: s.max_rate_dt / k if k > 0 else math.inf  # noqa: E731
    d_free = d_free if d_free is not None else d.d_free
    dt_layer = min((s.distance_fraction * cfg.membrane.binding_layer) ** 2 / (2 * d_free), rate(k_layer))
    a = cfg.receptors.capture_radius
    dt_receptor = min((s.receptor_fraction * a) ** 2 / (2 * d_free), (s.distance_fraction * a) ** 2 / (2 * d.d_bound))
    return min(dt_layer, dt_receptor, rate(k_off), s.dt_cap)


def simulate(cfg: Config, sweep_index: int = 0, n_molecules: int | None = None, t_max: float | None = None,
             seed: int | None = None, n_trajectories: int | None = None, first_id: int = 0,
             layout_index: int = 0, layout_seed: int | None = None) -> Result:
    """Simulate ``n_molecules`` for one sweep point.

    ``first_id`` offsets molecule ids (and so their RNG seeds) when a sweep point is split
    into chunks. ``layout_index`` picks the receptor layout: one layout per job (L36),
    drawn from (seed, sweep_index, layout_index) and independent of ``first_id``.
    ``layout_seed`` instead draws the layout from (seed, layout_seed) alone, so different
    sweep points can share layouts; used to pair conditions on the same receptor layouts (L57).
    """
    g, run, st = cfg.geometry, cfg.run, cfg.stepping
    n = n_molecules or run.n_molecules
    t_max = t_max or run.t_max
    seed = run.seed if seed is None else seed
    n_traj = run.n_trajectories if n_trajectories is None else n_trajectories

    cluster = make_cluster(g)
    factor = layer_factor(g.dim, g.cell_radius, g.spacing, cfg.membrane.binding_layer)
    sp = cfg.sweep_point(sweep_index, factor)
    mode = MODE_FIXED if st.mode == "fixed" else MODE_ADAPTIVE
    d_free = sp.d_free if sp.d_free is not None else cfg.diffusion.d_free
    dt_fixed = st.dt_fixed if st.dt_fixed is not None else strictest_dt(cfg, sp.k_layer, sp.k_off, d_free)
    seeds = molecule_seeds(seed, sweep_index, n, first_id)
    receptors = make_receptors(cluster, g.target_ring, cfg.receptors.n_per_cell, cfg.receptors.capture_radius,
                               np.random.default_rng([seed, sweep_index, layout_index] if layout_seed is None
                                                     else [seed, 1_000_003, layout_seed]),
                               on_cells=cfg.receptors.on_cells)
    lattice = (seeds, cluster.centers, cluster.neighbors, cluster.n_cells, cluster.source,
               cluster.ring, cluster.rings, cluster.spacing)
    physics = (g.cell_radius, cfg.membrane.binding_layer, d_free, cfg.diffusion.d_bound,
               sp.free_fraction, sp.k_off, sp.k_layer,
               mode, dt_fixed, st.distance_fraction, st.max_rate_dt, st.dt_cap, st.receptor_fraction,
               t_max, n_traj, run.trajectory_interval)

    t0 = time.time()
    if g.dim == 2:
        phis = np.arctan2(receptors.directions[..., 1], receptors.directions[..., 0])
        out = run_molecules_2d(*lattice, phis, receptors.block_of_cell, receptors.capture_radius, *physics)
    else:
        grid = make_grid(receptors.base)
        out = run_molecules_3d(*lattice, np.ascontiguousarray(receptors.rotations), receptors.block_of_cell,
                               receptors.base, grid.start, grid.index, grid.n, grid.box,
                               receptors.capture_radius, *physics,
                               cfg.receptors.finds_to_capture, cfg.receptors.rearm_distance)
        t_first_find, n_finds = out[15:]
        out = out[:15]
    (outcome, t_end, start_state, final_state, final_cell, receptor_id, start_pos, end_pos,
     time_free, n_trips, n_cell_changes, n_steps, traj, t_first_target, n_target_visits) = out
    wall = time.time() - t0

    molecules = pd.DataFrame({
        "molecule_id": np.arange(first_id, first_id + n),
        "seed": seeds,
        "outcome": pd.Categorical([OUTCOME_NAMES[o] for o in outcome], categories=list(OUTCOME_NAMES.values())),
        "t_end": t_end,
        "start_state": np.where(start_state == 1, "bound", "free"),
        "final_state": np.where(final_state == 1, "bound", "free"),
        "final_cell": final_cell,
        "final_ring": cluster.ring[final_cell],
        "receptor_id": receptor_id,  # index on the capturing cell, -1 if not captured
        **{f"{ax}0": start_pos[:, k] for k, ax in enumerate("xyz"[: g.dim])},
        **{f"{ax}1": end_pos[:, k] for k, ax in enumerate("xyz"[: g.dim])},
        "time_free": time_free,
        "n_trips": n_trips,  # unbinding events (excludes a free start)
        "n_cell_changes": n_cell_changes,
        "n_steps": n_steps,
        "t_first_target": np.where(t_first_target >= 0, t_first_target, np.nan),  # NaN = never reached (L49)
        "n_target_visits": n_target_visits,
        **({"t_first_find": np.where(t_first_find >= 0, t_first_find, np.nan),  # L86
            "n_finds": n_finds} if g.dim == 3 else {}),
    })
    meta = {
        "config": cfg.to_dict(),
        "sweep_index": sweep_index,
        "sweep_point": dict(dataclasses.asdict(sp), label=sp.label),
        "layer_factor": factor,
        "stepping_mode": st.mode,
        "dt_fixed": dt_fixed if mode == MODE_FIXED else None,
        "n_molecules": n,
        "t_max": t_max,
        "seed": seed,
        "layout_index": layout_index,
        "layout_seed": layout_seed,
        "receptor_cells": receptors.cells.tolist(),
        "receptor_rotations": receptors.rotations.tolist(),
        **({"receptor_angles": phis.tolist()} if g.dim == 2 else {}),
        "wall_time_s": wall,
    }
    return Result(molecules=molecules, trajectories=traj, meta=meta)
