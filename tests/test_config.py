from pathlib import Path

import pytest

from fpsim.config import Config, Stepping, from_dict, load_config

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
    assert cfg.run.t_max == 20000.0  # L41
    pts = cfg.sweep_points(79.2)
    assert len(pts) == cfg.n_sweep_points == 14
    assert [p.bound_dwell for p in pts[:4]] == [0.005, 0.0707, 0.285, 1.0]  # L27, L39
    assert [p.free_fraction for p in pts[4:9]] == [0.01, 0.05, 0.10, 0.20, 0.40]  # L38
    assert [p.bound_dwell for p in pts[9:]] == [0.005, 0.030, 0.135, 0.5, 1.0]  # L45
    assert all(p.free_fraction == 0.10 for p in pts[9:])


def test_sweep_point_rates_keep_five_percent_free():
    cfg = Config()
    p = cfg.sweep_point(cfg.find_point("dwell", bound_dwell=0.005), layer_factor=79.2)
    assert p.k_off == pytest.approx(200.0)
    assert p.k_on == pytest.approx(19 * 200.0)
    assert p.k_off / (p.k_on + p.k_off) == pytest.approx(0.05)
    assert p.k_layer == pytest.approx(79.2 * p.k_on)


@pytest.mark.parametrize("bad", [
    {"geometry": {"dim": 4}},
    {"membrane": {"binding_layer": 0.3}},  # thicker than half the 0.5 µm gap
    {"stepping": {"mode": "sometimes"}},
    {"sweeps": {"a": {"free_fraction": 0.05, "bound_dwell": [0.01, -1]}}},
    {"sweeps": {"a": {"free_fraction": 1.5, "bound_dwell": 0.1}}},
    {"sweeps": {"a": {"bound_dwell": 0.1}}},  # needs two quantities
    {"sweeps": {"a": {"free_fraction": [0.1, 0.2], "bound_dwell": [0.1, 0.2]}}},  # two lists
    {"membrane": {"free_fraction": 0.05}},  # moved to sweeps
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


def test_free_fraction_sweep_holds_free_trip_length():
    pts = [p for p in Config().sweep_points(79.2) if p.sweep == "free_fraction"]
    for p in pts:
        assert p.free_dwell == pytest.approx(0.015)  # L38
        assert p.k_off / (p.k_on + p.k_off) == pytest.approx(p.free_fraction)
    assert [round(p.bound_dwell, 4) for p in pts] == [1.485, 0.285, 0.135, 0.06, 0.0225]


def test_shared_point_identical_in_both_sweeps():
    cfg = Config()
    a = cfg.sweep_point(cfg.find_point("dwell", bound_dwell=0.285), 79.2)
    b = cfg.sweep_point(cfg.find_point("free_fraction", free_fraction=0.05), 79.2)
    assert (a.k_on, a.k_off) == pytest.approx((b.k_on, b.k_off))  # L39


def test_ten_percent_sweep_shares_a_point_with_free_fraction_sweep():
    cfg = Config()
    a = cfg.sweep_point(cfg.find_point("dwell_10pct", bound_dwell=0.135), 79.2)
    b = cfg.sweep_point(cfg.find_point("free_fraction", free_fraction=0.10), 79.2)
    assert (a.k_on, a.k_off) == pytest.approx((b.k_on, b.k_off))  # L45
    assert a.free_dwell == pytest.approx(0.015)
