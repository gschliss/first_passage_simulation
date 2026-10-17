import numpy as np
import pytest

from fpsim.lattice import hex_cluster
from fpsim.receptors import coverage, fibonacci_sphere, make_receptors

R, S = 10.0, 20.5


@pytest.fixture(scope="module")
def cluster():
    return hex_cluster(rings=5, cell_radius=R, spacing=S)


def make(cluster, seed):
    return make_receptors(cluster, target_ring=3, n_per_cell=3, capture_radius=0.01,
                          rng=np.random.default_rng(seed))


def test_receptors_only_on_target_ring(cluster):
    rs = make(cluster, 0)
    assert np.all(cluster.ring[rs.cells] == 3)
    assert rs.positions.shape == (18, 3, 2)
    has = rs.block_of_cell >= 0
    assert has.sum() == 18 and np.all(cluster.ring[has] == 3)


def test_receptors_sit_on_membrane_evenly_spaced(cluster):
    rs = make(cluster, 0)
    for k, cell in enumerate(rs.cells):
        r = np.linalg.norm(rs.positions[k] - cluster.centers[cell], axis=1)
        np.testing.assert_allclose(r, R)
        ang = np.sort(np.arctan2(rs.directions[k][:, 1], rs.directions[k][:, 0]) % (2 * np.pi))
        gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
        np.testing.assert_allclose(gaps, 2 * np.pi / 3)


def test_rotation_random_per_cell_and_reproducible_per_seed(cluster):
    a, b, c = make(cluster, 0), make(cluster, 0), make(cluster, 1)
    np.testing.assert_array_equal(a.positions, b.positions)  # one layout per job seed (L36)
    assert not np.allclose(a.positions, c.positions)
    first_angles = np.arctan2(a.directions[:, 0, 1], a.directions[:, 0, 0]) % (2 * np.pi / 3)
    assert np.std(first_angles) > 0.1  # cells are not all aligned (L18)


@pytest.mark.parametrize("dim, n, expected", [(2, 3, 0.000955), (3, 4000, 0.001)])
def test_coverage_about_one_tenth_percent(dim, n, expected):
    assert coverage(dim, n, 0.01, R) == pytest.approx(expected, rel=0.01)  # L18


def test_fibonacci_sphere_is_unit_and_even():
    p = fibonacci_sphere(4000)
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 1)
    np.testing.assert_allclose(p.mean(axis=0), 0, atol=1e-3)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(p).query(p, k=2)
    nn = d[:, 1] * R
    assert nn.min() > 0.4 and nn.max() < 0.7  # µm; ~0.56 µm typical spacing (L18)
