"""Typed run configuration, loaded from YAML.

Every parameter of a run lives here, with the decision-log entry (docs/model_notes.md)
it implements. Derived quantities (rates per sweep point, cluster size) are computed
here and nowhere else, so the saved config fully describes a run.

Units: µm, s.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Geometry:
    dim: int = 2  # 2 or 3 (L1, L5)
    cell_radius: float = 10.0  # µm (L2)
    gap: float = 0.5  # µm at closest approach (L2)
    target_ring: int = 3  # every cell in this ring is a target (L17)
    rings_beyond_target: int = 2  # cluster extends this many rings past the target ring (L12)

    @property
    def spacing(self) -> float:
        return 2 * self.cell_radius + self.gap

    @property
    def cluster_rings(self) -> int:
        return self.target_ring + self.rings_beyond_target


@dataclass(frozen=True)
class Receptors:
    n_per_cell: int = 3  # 3 in 2D, 4000 in 3D: ~0.1% coverage (L18)
    capture_radius: float = 0.010  # µm, from the receptor center (L19)


@dataclass(frozen=True)
class Diffusion:
    d_free: float = 10.0  # µm²/s (L20)
    d_bound: float = 0.5  # µm²/s (L20)


@dataclass(frozen=True)
class Membrane:
    free_fraction: float = 0.05  # equilibrium fraction free (L23)
    binding_layer: float = 0.010  # µm, δ: binding only within this of a membrane (L28)


@dataclass(frozen=True)
class Stepping:
    mode: str = "adaptive"  # "adaptive" or "fixed" (L34)
    distance_fraction: float = 0.125  # typical move ≤ this × distance to nearest feature (L30, L37)
    max_rate_dt: float = 0.05  # k·dt ≤ this for every active rate (L30)
    dt_cap: float = 1e-2  # s, upper bound on any step (L30)
    dt_fixed: float | None = None  # s; fixed mode only. None = strictest limit for the sweep point (L34)


@dataclass(frozen=True)
class Run:
    bound_dwell: tuple = (0.005, 0.0707, 1.0)  # s, mean 1/k_off per sweep point (L27)
    n_molecules: int = 1000
    t_max: float = 2000.0  # s; placeholder until the pilot run measures search times (build step 3)
    seed: int = 0
    n_trajectories: int = 0  # molecules whose paths are recorded (for animations)
    trajectory_interval: float = 0.05  # s between recorded positions


@dataclass(frozen=True)
class SweepPoint:
    """Rates for one bound-dwell value. ``k_layer`` needs the lattice's layer factor."""

    bound_dwell: float
    k_off: float
    k_on: float
    k_layer: float


@dataclass(frozen=True)
class Config:
    name: str = "unnamed"
    geometry: Geometry = field(default_factory=Geometry)
    receptors: Receptors = field(default_factory=Receptors)
    diffusion: Diffusion = field(default_factory=Diffusion)
    membrane: Membrane = field(default_factory=Membrane)
    stepping: Stepping = field(default_factory=Stepping)
    run: Run = field(default_factory=Run)

    def __post_init__(self):
        validate(self)

    def sweep_point(self, index: int, layer_factor: float) -> SweepPoint:
        """Rates for sweep point ``index``: k_off = 1/dwell, k_on from the free fraction (L23),
        k_layer = k_on · V_free/V_layer (L28, L29)."""
        dwell = self.run.bound_dwell[index]
        k_off = 1.0 / dwell
        f = self.membrane.free_fraction
        k_on = k_off * (1 - f) / f
        return SweepPoint(bound_dwell=dwell, k_off=k_off, k_on=k_on, k_layer=k_on * layer_factor)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


SECTIONS = {
    "geometry": Geometry,
    "receptors": Receptors,
    "diffusion": Diffusion,
    "membrane": Membrane,
    "stepping": Stepping,
    "run": Run,
}


def from_dict(d: dict) -> Config:
    unknown = set(d) - set(SECTIONS) - {"name"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    kwargs = {"name": d.get("name", "unnamed")}
    for key, cls in SECTIONS.items():
        section = dict(d.get(key) or {})
        allowed = {f.name for f in dataclasses.fields(cls)}
        bad = set(section) - allowed
        if bad:
            raise ValueError(f"unknown keys in '{key}': {sorted(bad)}")
        if key == "run" and "bound_dwell" in section:
            section["bound_dwell"] = tuple(section["bound_dwell"])
        kwargs[key] = cls(**section)
    return Config(**kwargs)


def load_config(path) -> Config:
    with open(Path(path)) as f:
        return from_dict(yaml.safe_load(f) or {})


def validate(cfg: Config) -> None:
    g, r, m, s, run = cfg.geometry, cfg.receptors, cfg.membrane, cfg.stepping, cfg.run
    checks = [
        (g.dim in (2, 3), f"geometry.dim must be 2 or 3, got {g.dim}"),
        (g.cell_radius > 0 and g.gap > 0, "cell_radius and gap must be positive"),
        (g.target_ring >= 1 and g.rings_beyond_target >= 1, "target_ring and rings_beyond_target must be >= 1"),
        (r.n_per_cell >= 1, "receptors.n_per_cell must be >= 1"),
        (0 < r.capture_radius < g.gap / 2, "capture_radius must be in (0, gap/2)"),
        (cfg.diffusion.d_free > 0 and cfg.diffusion.d_bound > 0, "diffusion coefficients must be positive"),
        (0 < m.free_fraction < 1, "free_fraction must be in (0, 1)"),
        # Layers of facing cells must not overlap, or "nearest membrane" becomes ambiguous.
        (0 < m.binding_layer < g.gap / 2, "binding_layer must be in (0, gap/2)"),
        (s.mode in ("adaptive", "fixed"), f"stepping.mode must be 'adaptive' or 'fixed', got {s.mode!r}"),
        (0 < s.distance_fraction <= 1 and 0 < s.max_rate_dt < 1, "accuracy settings out of range"),
        (s.dt_fixed is None or s.dt_fixed > 0, "dt_fixed must be positive"),
        (len(run.bound_dwell) >= 1 and all(t > 0 for t in run.bound_dwell), "bound_dwell must be positive values"),
        (run.n_molecules >= 1 and run.t_max > 0, "n_molecules and t_max must be positive"),
        (math.isfinite(run.trajectory_interval) and run.trajectory_interval > 0, "trajectory_interval must be positive"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))
