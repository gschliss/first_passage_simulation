"""Core time-stepping loop: move agents until they hit the target."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fpsim.agents import BrownianWalkers
from fpsim.geometry import Geometry


@dataclass
class SimulationResult:
    fpt: np.ndarray  # first-passage time per agent; NaN if not absorbed by t_max
    dt: float
    t_max: float
    meta: dict = field(default_factory=dict)

    @property
    def absorbed_fraction(self) -> float:
        return float(np.mean(~np.isnan(self.fpt)))


def run_simulation(
    geometry: Geometry,
    n_agents: int,
    D: float,
    dt: float,
    t_max: float,
    seed=None,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    walkers = BrownianWalkers(geometry.sample_start(n_agents, rng), D)

    fpt = np.full(n_agents, np.nan)
    alive = ~geometry.is_absorbed(walkers.positions)
    fpt[~alive] = 0.0

    n_steps = int(np.ceil(t_max / dt))
    for step in range(1, n_steps + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        x_old = walkers.positions[idx]
        x_new = geometry.apply_boundaries(x_old, walkers.propose_step(dt, rng, alive))
        walkers.positions[idx] = x_new

        hit = geometry.is_absorbed(x_new)
        fpt[idx[hit]] = step * dt
        alive[idx[hit]] = False

    return SimulationResult(
        fpt=fpt,
        dt=dt,
        t_max=t_max,
        meta={"dim": geometry.dim, "D": D, "n_agents": n_agents, "seed": seed},
    )
