"""Receptor capture (L19) and escape (L15) in the 2D kernel."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fpsim import load_config, make_cluster
from fpsim.config import from_dict
from fpsim.kernel.geom2d import bound_capture, escape_test, free_capture, nearest_cell
from fpsim.simulate import simulate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MINI = load_config(CONFIGS / "mini2d.yaml")
R, A = 10.0, 0.010
HW = A / R  # capture half-width in radians
PHIS = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])


def test_bound_capture_inside_window():
    assert bound_capture(0.5 * HW, 0.6 * HW, PHIS, HW, 1e-6, 1.0) == 0
    assert bound_capture(0.1, 2 * np.pi / 3 + 0.5 * HW, PHIS, HW, 1e-6, 1.0) == 1


def test_bound_capture_jump_across_window():
    assert bound_capture(-3 * HW, 3 * HW, PHIS, HW, 1e-9, 1.0) == 0


def test_bound_no_capture_far_away():
    assert bound_capture(1.0, 1.0 + 1e-4, PHIS, HW, 1e-5, 0.0) == -1


def test_bound_bridge_probability():
    # Both ends one sigma outside the window, same side: P(touch) = exp(-2).
    sigma = 1e-4
    th = HW + sigma
    rng = np.random.default_rng(0)
    hits = np.mean([bound_capture(th, th, PHIS, HW, sigma, u) >= 0 for u in rng.random(20000)])
    assert hits == pytest.approx(np.exp(-2), abs=0.01)


def test_free_capture_distance():
    on = np.array([R + 0.5 * A, 0.0])
    assert free_capture(on[0], on[1], 0.0, 0.0, PHIS, R, A) == 0
    assert free_capture(R + 2 * A, 0.0, 0.0, 0.0, PHIS, R, A) == -1


def test_escape_test_matches_nearest_site():
    cluster = make_cluster(MINI.geometry)
    tree = cKDTree(cluster.centers)
    rng = np.random.default_rng(1)
    extent = (cluster.rings + 1.5) * cluster.spacing
    pts = rng.uniform(-extent, extent, size=(20000, 2))
    checked = 0
    for x, y in pts:
        _, idx = tree.query([x, y])
        near, dist = nearest_cell(x, y, cluster.source, cluster.centers, cluster.neighbors, cluster.n_cells)
        if dist < cluster.cell_radius or cluster.ring[near] != cluster.rings:
            continue
        escaped, _ = escape_test(x, y, near, dist, cluster.centers, cluster.neighbors, cluster.n_cells, cluster.spacing)
        assert escaped == bool(cluster.is_ghost[idx])
        checked += 1
    assert checked > 1000


# Scaled-down geometry (2 µm cells, 1 receptor per cell) so captures and escapes both
# happen within seconds. Test-only; not a model configuration.
SMALL = from_dict({
    "name": "small_test",
    "geometry": {"cell_radius": 2.0, "gap": 0.5, "target_ring": 1, "rings_beyond_target": 1},
    "receptors": {"n_per_cell": 1},
})


@pytest.fixture(scope="module")
def mini_run():
    return simulate(SMALL, 2, n_molecules=400, t_max=50.0)


def test_outcomes_occur(mini_run):
    counts = mini_run.molecules.outcome.value_counts()
    assert counts["captured"] > 10 and counts["escaped"] > 10


def test_captured_molecules_end_at_a_receptor(mini_run):
    m = mini_run.molecules[mini_run.molecules.outcome == "captured"]
    cluster = make_cluster(SMALL.geometry)
    r = SMALL.geometry.cell_radius
    assert (cluster.ring[m.final_cell] == SMALL.geometry.target_ring).all()
    assert (m.receptor_id >= 0).all()
    cells = np.array(mini_run.meta["receptor_cells"])
    angles = np.array(mini_run.meta["receptor_angles"])
    for _, row in m.iterrows():
        block = int(np.flatnonzero(cells == row.final_cell)[0])
        phi = angles[block, row.receptor_id]
        rec = cluster.centers[row.final_cell] + r * np.array([np.cos(phi), np.sin(phi)])
        assert np.hypot(row.x1 - rec[0], row.y1 - rec[1]) <= A + 1e-9


def test_escaped_molecules_end_nearest_a_ghost(mini_run):
    m = mini_run.molecules[mini_run.molecules.outcome == "escaped"]
    cluster = make_cluster(SMALL.geometry)
    _, idx = cKDTree(cluster.centers).query(m[["x1", "y1"]].to_numpy())
    assert cluster.is_ghost[idx].all()
    assert (m.final_state == "free").all()


def test_receptor_layout_independent_of_first_id():
    a = simulate(MINI, 0, n_molecules=2, t_max=0.01)
    b = simulate(MINI, 0, n_molecules=2, t_max=0.01, first_id=100)
    c = simulate(MINI, 0, n_molecules=2, t_max=0.01, layout_index=1)
    assert a.meta["receptor_angles"] == b.meta["receptor_angles"]
    assert a.meta["receptor_angles"] != c.meta["receptor_angles"]
