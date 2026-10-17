from pathlib import Path

import pytest

from fpsim.config import Config, Geometry, Membrane, Stepping, from_dict, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_lattice2d_config_loads_with_agreed_values():
    cfg = load_config(CONFIGS / "lattice2d.yaml")
    g = cfg.geometry
    assert (g.dim, g.cell_radius, g.gap, g.target_ring) == (2, 10.0, 0.5, 3)
    assert g.spacing == 20.5
    assert g.cluster_rings == 5
    assert cfg.receptors.n_per_cell == 3
    assert cfg.membrane.binding_layer == 0.010
    assert cfg.stepping.mode == "adaptive"
    assert cfg.run.bound_dwell == (0.005, 0.0707, 1.0)


def test_sweep_point_rates_keep_five_percent_free():
    cfg = Config()
    p = cfg.sweep_point(0, layer_factor=79.2)
    assert p.k_off == pytest.approx(200.0)
    assert p.k_on == pytest.approx(19 * 200.0)
    assert p.k_off / (p.k_on + p.k_off) == pytest.approx(0.05)
    assert p.k_layer == pytest.approx(79.2 * p.k_on)


@pytest.mark.parametrize("bad", [
    {"geometry": {"dim": 4}},
    {"membrane": {"binding_layer": 0.3}},  # thicker than half the 0.5 µm gap
    {"stepping": {"mode": "sometimes"}},
    {"run": {"bound_dwell": [0.01, -1]}},
])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        from_dict(bad)


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict({"geometry": {"radius": 10}})
    with pytest.raises(ValueError, match="unknown config sections"):
        from_dict({"geometery": {}})


def test_fixed_mode_toggle():
    cfg = Config(stepping=Stepping(mode="fixed", dt_fixed=1e-7))
    assert cfg.stepping.mode == "fixed"
    assert cfg.to_dict()["stepping"]["dt_fixed"] == 1e-7
