"""3D (HCP) lattice, receptor lookup and kernel."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fpsim import load_config, make_cluster
from fpsim.config import from_dict
from fpsim.kernel.geom3d import escape_test3, nearest_cell3, nearest_receptor, sphere_step
from fpsim.lattice import hcp_cluster
from fpsim.receptors import fibonacci_sphere, make_grid
from fpsim.simulate import simulate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
R, S = 10.0, 20.5


@pytest.fixture(scope="module")
def cluster():
    return hcp_cluster(5, R, S)


def test_hcp_ring_sizes_and_cluster(cluster):
    sizes = [int(np.sum(cluster.ring == k)) for k in range(7)]
    assert sizes == [1, 12, 44, 96, 170, 264, 380]  # L26
    assert cluster.n_cells == 587


def test_hcp_twelve_touching_neighbors(cluster):
    for i in range(cluster.n_cells):
        nbs = cluster.neighbors[i][cluster.neighbors[i] >= 0]
        assert len(nbs) == 12
        np.testing.assert_allclose(np.linalg.norm(cluster.centers[nbs] - cluster.centers[i], axis=1), S)


def test_hcp_is_abab_stacked(cluster):
    z = np.unique(np.round(cluster.centers[:, 2], 6))
    h = S * np.sqrt(2 / 3)
    np.testing.assert_allclose(np.diff(z), h, rtol=1e-6)  # heights were rounded to 1e-6
    xy_a = cluster.centers[np.isclose(cluster.centers[:, 2], 0)][:, :2]
    xy_aa = cluster.centers[np.isclose(cluster.centers[:, 2], 2 * h)][:, :2]
    xy_b = cluster.centers[np.isclose(cluster.centers[:, 2], h)][:, :2]
    # A layers repeat every second layer (HCP); B layers are offset (L6).
    assert cKDTree(xy_a).query(xy_aa)[0].max() < 1e-6
    assert cKDTree(xy_a).query(xy_b)[0].min() > 1.0


def test_nearest_cell_walk_matches_kdtree(cluster):
    tree = cKDTree(cluster.centers[: cluster.n_cells])
    rng = np.random.default_rng(0)
    pts = rng.uniform(-90, 90, size=(5000, 3))
    for p in pts:
        _, idx = tree.query(p)
        near, _ = nearest_cell3(p[0], p[1], p[2], cluster.source, cluster.centers, cluster.neighbors, cluster.n_cells)
        assert near == idx


def test_escape_test_matches_nearest_site(cluster):
    tree = cKDTree(cluster.centers)
    rng = np.random.default_rng(1)
    checked = 0
    for p in rng.uniform(-140, 140, size=(30000, 3)):
        near, dist = nearest_cell3(p[0], p[1], p[2], cluster.source, cluster.centers, cluster.neighbors, cluster.n_cells)
        if dist < R or cluster.ring[near] != cluster.rings or dist > 1.5 * S:
            continue
        escaped, _ = escape_test3(p[0], p[1], p[2], near, dist, cluster.centers, cluster.neighbors, cluster.n_cells, S)
        assert escaped == bool(cluster.is_ghost[tree.query(p)[1]])
        checked += 1
    assert checked > 500


def test_receptor_grid_finds_nearest_and_lower_bounds():
    base = fibonacci_sphere(4000)
    grid = make_grid(base)
    tree = cKDTree(base)
    rng = np.random.default_rng(2)
    q = rng.standard_normal((3000, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for v in q:
        true_d, true_j = tree.query(v)
        j, d = nearest_receptor(v[0], v[1], v[2], base, grid.start, grid.index, grid.n, grid.box)
        assert d <= true_d + 1e-12  # never overestimates the distance to the nearest receptor
        if true_d < grid.box:
            assert j == true_j and d == pytest.approx(true_d)


def test_sphere_step_diffusion():
    # On a sphere, <u(t)·u(0)> = exp(-2 D t / R²).
    rng = np.random.default_rng(3)
    d, dt, steps, n = 0.5, 1e-3, 400, 400
    sigma = np.sqrt(2 * d * dt) / R
    dots = []
    for _ in range(n):
        u = np.array([0.0, 0.0, 1.0])
        for _ in range(steps):
            u = np.array(sphere_step(*u, sigma, *rng.standard_normal(3)))
        assert np.linalg.norm(u) == pytest.approx(1.0)
        dots.append(u[2])
    expected = np.exp(-2 * d * dt * steps / R**2)
    assert np.mean(dots) == pytest.approx(expected, abs=4 * np.std(dots) / np.sqrt(n) + 1e-6)


SMALL3 = from_dict({
    "name": "small3_test",
    "geometry": {"dim": 3, "cell_radius": 2.0, "gap": 0.5, "target_ring": 1, "rings_beyond_target": 1},
    "receptors": {"n_per_cell": 50},
    "sweeps": {"a": {"free_fraction": 0.05, "bound_dwell": [0.005, 1.0]}},
})


@pytest.fixture(scope="module")
def small_run():
    return simulate(SMALL3, 1, n_molecules=300, t_max=30.0)


def test_3d_outcomes_and_invariants(small_run):
    m = small_run.molecules
    counts = m.outcome.value_counts()
    assert counts["captured"] > 10 and counts["escaped"] > 5
    cluster = make_cluster(SMALL3.geometry)
    r = SMALL3.geometry.cell_radius
    end = m[["x1", "y1", "z1"]].to_numpy()
    d = np.linalg.norm(end - cluster.centers[m.final_cell.to_numpy()], axis=1)
    bound = (m.final_state == "bound").to_numpy()
    np.testing.assert_allclose(d[bound], r, atol=1e-9)
    assert (d[~bound] >= r - 1e-9).all()
    assert (m.n_cell_changes <= m.n_trips + (m.start_state == "free")).all()


def test_3d_captured_at_receptor(small_run):
    m = small_run.molecules[small_run.molecules.outcome == "captured"]
    cluster = make_cluster(SMALL3.geometry)
    r = SMALL3.geometry.cell_radius
    cells = np.array(small_run.meta["receptor_cells"])
    rots = np.array(small_run.meta["receptor_rotations"])
    base = fibonacci_sphere(50)
    for _, row in m.iterrows():
        k = int(np.flatnonzero(cells == row.final_cell)[0])
        rec = cluster.centers[row.final_cell] + r * rots[k] @ base[row.receptor_id]
        assert np.linalg.norm([row.x1 - rec[0], row.y1 - rec[1], row.z1 - rec[2]]) <= 0.010 + 1e-9


def test_3d_escaped_nearest_ghost(small_run):
    m = small_run.molecules[small_run.molecules.outcome == "escaped"]
    cluster = make_cluster(SMALL3.geometry)
    _, idx = cKDTree(cluster.centers).query(m[["x1", "y1", "z1"]].to_numpy())
    assert cluster.is_ghost[idx].all()


def test_3d_unbinding_rate_exact_and_free_fraction():
    cfg = load_config(CONFIGS / "lattice3d.yaml")
    i = cfg.find_point("dwell", bound_dwell=0.005)
    m = simulate(cfg, i, n_molecules=100, t_max=5.0).molecules
    bound_time = m.t_end.sum() - m.time_free.sum()
    expected = bound_time / 0.005
    assert m.n_trips.sum() == pytest.approx(expected, rel=4 / np.sqrt(expected))
    f = m.time_free / m.t_end
    assert abs(f.mean() - 0.05) < 0.003 + 4 * f.std(ddof=1) / np.sqrt(len(f))
