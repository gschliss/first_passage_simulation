"""Features for the dimension-switching question (L47-L50): free-only reference points,
per-point D_free, 2 nm targets, arrival-time outputs, receptor-free runs."""

import numpy as np
import pytest

from fpsim import make_cluster
from fpsim.config import from_dict
from fpsim.simulate import simulate


def small(dim=2, capture_radius=0.002, n_per_cell=None, sweeps=None):
    """Scaled-down geometry (2 µm cells) so runs take seconds. Test-only."""
    return from_dict({
        "name": "small_dim_test",
        "geometry": {"dim": dim, "cell_radius": 2.0, "gap": 0.5, "target_ring": 1, "rings_beyond_target": 1},
        "receptors": {"n_per_cell": n_per_cell if n_per_cell is not None else (1 if dim == 2 else 50),
                      "capture_radius": capture_radius},
        "sweeps": sweeps or {"switch": {"free_fraction": 0.05, "bound_dwell": 1.0},
                             "reference": {"free_fraction": 1.0, "d_free": 0.5}},
    })


@pytest.mark.parametrize("dim", [2, 3])
def test_free_only_points_never_bind(dim):
    cfg = small(dim)
    i = cfg.find_point("reference", free_fraction=1.0)
    r = simulate(cfg, i, n_molecules=50, t_max=5.0)
    m = r.molecules
    assert r.meta["sweep_point"]["k_on"] == 0.0
    assert (m.start_state == "free").all() and (m.n_trips == 0).all() and (m.n_target_visits == 0).all()
    assert (m.final_state == "free").all()
    np.testing.assert_allclose(m.time_free, m.t_end)


def test_d_free_override_changes_spreading():
    cfg = small(sweeps={"slow": {"free_fraction": 1.0, "d_free": 0.5}, "fast": {"free_fraction": 1.0, "d_free": 5.0}},
                n_per_cell=1)
    msd = []
    for i in range(2):
        m = simulate(cfg, i, n_molecules=200, t_max=0.5).molecules
        msd.append((((m.x1 - m.x0) ** 2 + (m.y1 - m.y0) ** 2) / m.t_end).mean())
    assert msd[1] / msd[0] == pytest.approx(10, rel=0.35)  # spreading scales with D at short times


@pytest.mark.parametrize("dim", [2, 3])
def test_arrival_time_and_two_nm_capture(dim):
    cfg = small(dim)
    r = simulate(cfg, 0, n_molecules=300, t_max=60.0)
    m = r.molecules
    cap = m[m.outcome == "captured"]
    assert len(cap) > 20
    assert cap.t_first_target.notna().all() and (cap.t_first_target <= cap.t_end + 1e-12).all()
    assert m[m.t_first_target.isna()].outcome.isin(["escaped", "timeout"]).all()
    bound_caps = cap[cap.final_state == "bound"]
    assert (bound_caps.n_target_visits >= 1).all()  # bound capture needs a binding episode on a target
    # Captured within 2 nm of a receptor center.
    cluster = make_cluster(cfg.geometry)
    rc, rr = np.array(r.meta["receptor_cells"]), np.array(r.meta["receptor_rotations"])
    if dim == 2:
        base = np.array([[1.0, 0.0]])
    else:
        from fpsim.receptors import fibonacci_sphere
        base = fibonacci_sphere(cfg.receptors.n_per_cell)
    cols = ["x1", "y1"] + (["z1"] if dim == 3 else [])
    for _, row in cap.iterrows():
        k = int(np.flatnonzero(rc == row.final_cell)[0])
        rec = cluster.centers[row.final_cell] + cfg.geometry.cell_radius * rr[k] @ base[row.receptor_id]
        assert np.linalg.norm(row[cols].to_numpy(float) - rec) <= 0.002 + 1e-9


def test_receptor_free_runs():
    cfg = small(n_per_cell=0)
    m = simulate(cfg, 0, n_molecules=50, t_max=5.0).molecules
    assert not (m.outcome == "captured").any()
