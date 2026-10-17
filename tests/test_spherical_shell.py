import numpy as np
import pytest

from fpsim.analysis import fpt_summary, mfpt_spherical_shell
from fpsim.geometry import SphericalShell
from fpsim.simulation import run_simulation


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_start_positions_on_r0(dim):
    geo = SphericalShell(dim=dim, R=1.0, a=0.2, r0=0.6)
    x = geo.sample_start(100, np.random.default_rng(0))
    assert x.shape == (100, dim)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 0.6)


def test_reflection_keeps_agents_inside():
    geo = SphericalShell(dim=3, R=1.0, a=0.2, r0=0.6)
    x_new = np.array([[1.1, 0.0, 0.0], [0.0, 0.5, 0.0]])
    out = geo.apply_boundaries(None, x_new.copy())
    np.testing.assert_allclose(out[0], [0.9, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 0.5, 0.0])


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mfpt_matches_theory(dim):
    R, a, r0, D = 1.0, 0.2, 0.6, 1.0
    geo = SphericalShell(dim=dim, R=R, a=a, r0=r0)
    result = run_simulation(geo, n_agents=2000, D=D, dt=1e-4, t_max=30.0, seed=1)
    summary = fpt_summary(result.fpt)
    theory = mfpt_spherical_shell(dim, D, R, a, r0)

    assert summary["absorbed_fraction"] > 0.999
    # Tolerance covers sampling error (~2-3%) plus time-discretization bias.
    assert summary["mean"] == pytest.approx(theory, rel=0.1)
