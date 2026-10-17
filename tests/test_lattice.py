import numpy as np
import pytest

from fpsim.lattice import hex_cluster, layer_factor

R, S = 10.0, 20.5


@pytest.fixture(scope="module")
def cluster():
    return hex_cluster(rings=5, cell_radius=R, spacing=S)


def test_ring_sizes_are_6k(cluster):
    sizes = [int(np.sum(cluster.ring == k)) for k in range(cluster.rings + 2)]
    assert sizes == [1] + [6 * k for k in range(1, cluster.rings + 2)]
    assert cluster.n_cells == 91  # L17: 5 rings in 2D


def test_ghost_ring_is_the_ring_just_outside(cluster):
    assert np.all(cluster.ring[cluster.is_ghost] == cluster.rings + 1)
    assert cluster.is_ghost.sum() == 6 * (cluster.rings + 1)
    assert not cluster.is_ghost[: cluster.n_cells].any()  # real cells come first


def test_source_at_origin(cluster):
    assert cluster.ring[cluster.source] == 0
    np.testing.assert_allclose(cluster.centers[cluster.source], [0, 0], atol=1e-12)


def test_neighbors_touch_at_one_spacing(cluster):
    for i in range(cluster.n_cells):
        nbs = cluster.neighbors[i][cluster.neighbors[i] >= 0]
        d = np.linalg.norm(cluster.centers[nbs] - cluster.centers[i], axis=1)
        np.testing.assert_allclose(d, S)
        assert len(nbs) == 6  # every real cell has a full neighbor shell (real or ghost)


def test_ring_is_neighbor_step_distance(cluster):
    for i in range(cluster.n_sites):
        nbs = cluster.neighbors[i][cluster.neighbors[i] >= 0]
        assert np.all(np.abs(cluster.ring[nbs] - cluster.ring[i]) <= 1)


def test_ring3_targets_and_distance_range(cluster):
    targets = cluster.cells_in_ring(3)
    assert len(targets) == 18
    d = np.linalg.norm(cluster.centers[targets], axis=1)
    assert d.min() == pytest.approx(3 * S * np.sqrt(3) / 2, rel=0.02)  # edge midpoints of the ring
    assert d.max() == pytest.approx(3 * S)  # corners


def test_cells_do_not_overlap(cluster):
    c = cluster.centers[: cluster.n_cells]
    d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(len(c)) * 1e9
    assert d.min() == pytest.approx(S)  # closest approach leaves exactly the 0.5 µm gap


@pytest.mark.parametrize("dim, expected", [(2, 79.2), (3, 151.3)])
def test_layer_factor_matches_logged_values(dim, expected):
    assert layer_factor(dim, R, S, 0.010) == pytest.approx(expected, abs=0.05)  # L29


def test_layer_factor_rejects_overlapping_layers():
    with pytest.raises(ValueError):
        layer_factor(2, R, S, 0.3)
