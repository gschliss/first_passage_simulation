"""Measure the long-range effective diffusion coefficient D_eff (L50).

Runs a sweep point with **no receptors** on an enlarged cluster (so molecules don't
reach the escape line during the run), records every molecule's position at regular
intervals, and fits the time-averaged mean squared displacement (TAMSD):

    D_eff = TAMSD(lag) / (2 · dim · lag)

over lags long enough to average over many free/bound switches and cell-to-cell hops.
Reporting D_eff at a short and a long lag shows whether the long-time regime was reached.
In 3D the in-layer (x, y) and across-layer (z) components are also reported, since HCP
need not be isotropic.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fpsim.config import Config
from fpsim.simulate import simulate


def calibration_config(cfg: Config, cluster_rings: int = 12, interval: float = 2.0) -> Config:
    """Copy of ``cfg`` with no receptors and a cluster of ``cluster_rings`` rings."""
    g = cfg.geometry
    return dataclasses.replace(
        cfg,
        name=f"{cfg.name}_deff",
        geometry=dataclasses.replace(g, rings_beyond_target=max(cluster_rings - g.target_ring, 1)),
        receptors=dataclasses.replace(cfg.receptors, n_per_cell=0),
        run=dataclasses.replace(cfg.run, trajectory_interval=interval),
    )


def tamsd(traj: np.ndarray, dim: int, lags: np.ndarray) -> np.ndarray:
    """Per-molecule time-averaged MSD, shape (n_molecules, len(lags), dim) per axis."""
    pos = traj[:, :-1, :dim]  # last slot holds the final position, not a regular frame
    out = np.full((pos.shape[0], len(lags), dim), np.nan)
    for k, lag in enumerate(lags):
        d = pos[:, lag:, :] - pos[:, :-lag, :]
        out[:, k, :] = np.nanmean(d**2, axis=1)
    return out


@dataclasses.dataclass
class DeffResult:
    sweep_index: int
    label: str
    d_eff: float  # µm²/s at the long lag
    d_eff_se: float
    d_eff_short: float  # at the short lag; ≈ d_eff once the long-time regime is reached
    d_eff_axes: tuple  # per-axis D at the long lag (x, y[, z])
    n: int
    t_max: float
    escaped: int


def measure_deff(cfg: Config, sweep_index: int, n_molecules: int = 100, t_max: float = 2000.0,
                 short_lag: float = 20.0, long_lag: float = 200.0, cluster_rings: int = 12) -> DeffResult:
    cal = calibration_config(cfg, cluster_rings)
    r = simulate(cal, sweep_index, n_molecules=n_molecules, t_max=t_max, n_trajectories=n_molecules)
    dim = cal.geometry.dim
    dt = cal.run.trajectory_interval
    lags = np.array([int(round(short_lag / dt)), int(round(long_lag / dt))])
    msd = tamsd(r.trajectories, dim, lags)  # (n, 2, dim)
    per_mol = msd.sum(axis=2) / (2 * dim * lags * dt)  # (n, 2) D per molecule
    d_long = per_mol[:, 1]
    axes = tuple(float(np.nanmean(msd[:, 1, a]) / (2 * lags[1] * dt)) for a in range(dim))
    return DeffResult(
        sweep_index=sweep_index, label=r.meta["sweep_point"]["label"],
        d_eff=float(np.nanmean(d_long)), d_eff_se=float(np.nanstd(d_long, ddof=1) / np.sqrt(np.isfinite(d_long).sum())),
        d_eff_short=float(np.nanmean(per_mol[:, 0])), d_eff_axes=axes, n=n_molecules, t_max=t_max,
        escaped=int((r.molecules.outcome == "escaped").sum()),
    )
