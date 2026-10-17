"""Fast checks of the 2D kernel (build step 2: free/bound states and binding layer).

Slower, statistical validation lives in scripts/check_kernel.py.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fpsim.config import Config, Run, Stepping
from fpsim.lattice import make_cluster
from fpsim.simulate import simulate

CFG = Config()


@pytest.fixture(scope="module")
def short_runs():
    return {i: simulate(CFG, i, n_molecules=200, t_max=2.0, n_trajectories=2) for i in range(3)}


def test_same_seed_same_result():
    a = simulate(CFG, 0, n_molecules=20, t_max=0.5).molecules
    b = simulate(CFG, 0, n_molecules=20, t_max=0.5).molecules
    pd.testing.assert_frame_equal(a, b)


def test_results_independent_of_how_molecules_are_split_into_jobs():
    whole = simulate(CFG, 1, n_molecules=10, t_max=0.5).molecules
    second_half = simulate(CFG, 1, n_molecules=5, t_max=0.5, first_id=5).molecules
    pd.testing.assert_frame_equal(whole.iloc[5:].reset_index(drop=True), second_half)


def test_invariants(short_runs):
    cluster = make_cluster(CFG.geometry)
    R = CFG.geometry.cell_radius
    for r in short_runs.values():
        m = r.molecules
        # Changing cell needs a free trip (L21): one per unbinding, plus the initial free period.
        free_periods = m.n_trips + (m.start_state == "free")
        assert (m.n_cell_changes <= free_periods).all()
        assert (m.time_free <= m.t_end + 1e-12).all()
        assert (m.outcome == "timeout").all()  # no receptors/escape yet
        end = m[["x1", "y1"]].to_numpy()
        d = np.linalg.norm(end - cluster.centers[m.final_cell.to_numpy()], axis=1)
        bound = (m.final_state == "bound").to_numpy()
        np.testing.assert_allclose(d[bound], R, atol=1e-9)  # bound molecules sit on the membrane
        assert (d[~bound] >= R - 1e-9).all()  # free molecules are never inside a cell
        start = m[["x0", "y0"]].to_numpy()
        np.testing.assert_allclose(np.linalg.norm(start - cluster.centers[cluster.source], axis=1), R, atol=CFG.membrane.binding_layer + 1e-9)


def test_unbinding_rate_is_exact(short_runs):
    for i, r in short_runs.items():
        m = r.molecules
        T = m.t_end.sum()
        bound_time = T - m.time_free.sum()
        expected = bound_time / r.meta["sweep_point"]["bound_dwell"]
        assert m.n_trips.sum() == pytest.approx(expected, rel=4 / np.sqrt(expected))  # Poisson, 4 sigma


def test_free_fraction_near_five_percent():
    r = simulate(CFG, 0, n_molecules=300, t_max=10.0)
    f = r.molecules.time_free / r.molecules.t_end
    se = f.std(ddof=1) / np.sqrt(len(f))
    assert abs(f.mean() - 0.05) < 0.002 + 4 * se  # small start-up deficit allowed (L16)


def test_fixed_mode_same_schema():
    cfg = dataclasses.replace(CFG, stepping=Stepping(mode="fixed"))
    r = simulate(cfg, 0, n_molecules=4, t_max=0.01)
    a = simulate(CFG, 0, n_molecules=4, t_max=0.01)
    assert list(r.molecules.columns) == list(a.molecules.columns)
    assert r.meta["dt_fixed"] > 0 and a.meta["dt_fixed"] is None
    assert (r.molecules.n_steps > a.molecules.n_steps).all()


def test_trajectories_recorded(short_runs):
    traj = short_runs[2].trajectories
    n_frames = int(2.0 / CFG.run.trajectory_interval) + 2  # +1 slot for the final position
    assert traj.shape == (2, n_frames, 4)
    assert np.isfinite(traj[:, :-1]).all()  # all timeouts here, so every regular frame is filled
    assert set(np.unique(traj[:, :, 2])) <= {0.0, 1.0}
