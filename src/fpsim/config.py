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
    n_molecules: int = 1000
    t_max: float = 20000.0  # s (L41; pilot: longest search 4.8 ks)
    seed: int = 0
    n_trajectories: int = 0  # molecules whose paths are recorded (for animations)
    trajectory_interval: float = 0.05  # s between recorded positions


RATE_KEYS = ("free_fraction", "bound_dwell", "free_dwell")
SWEEP_KEYS = RATE_KEYS + ("d_free",)


@dataclass(frozen=True)
class Sweep:
    """One named sweep: exactly two of free_fraction, bound_dwell (1/k_off) and
    free_dwell (1/k_on) are given, and at most one of them is a list (L23, L27, L38).

    ``free_fraction: 1.0`` alone defines **free-only** points that never bind (L47);
    ``d_free`` (number or list) overrides diffusion.d_free for this sweep's points.
    """

    name: str
    free_fraction: float | tuple | None = None
    bound_dwell: float | tuple | None = None  # s
    free_dwell: float | tuple | None = None  # s
    d_free: float | tuple | None = None  # µm²/s, overrides diffusion.d_free

    @property
    def free_only(self) -> bool:
        return self.free_fraction == 1.0 and self.bound_dwell is None and self.free_dwell is None

    def points(self) -> list[dict]:
        given = {k: getattr(self, k) for k in SWEEP_KEYS if getattr(self, k) is not None}
        lists = [k for k, v in given.items() if isinstance(v, tuple)]
        n = len(given[lists[0]]) if lists else 1
        out = []
        for j in range(n):
            vals = {k: (v[j] if isinstance(v, tuple) else v) for k, v in given.items()}
            f, tb, tf = vals.get("free_fraction"), vals.get("bound_dwell"), vals.get("free_dwell")
            if self.free_only:
                tb, tf = 0.0, math.inf
            # f = tf / (tf + tb): the third quantity follows from the other two.
            elif f is None:
                f = tf / (tf + tb)
            elif tb is None:
                tb = tf * (1 - f) / f
            elif tf is None:
                tf = tb * f / (1 - f)
            out.append(dict(free_fraction=f, bound_dwell=tb, free_dwell=tf, d_free=vals.get("d_free")))
        return out


DEFAULT_SWEEPS = (
    Sweep("dwell", free_fraction=0.05, bound_dwell=(0.005, 0.0707, 0.285, 1.0)),  # L27, L39
    Sweep("free_fraction", free_dwell=0.015, free_fraction=(0.01, 0.05, 0.10, 0.20, 0.40)),  # L38
    Sweep("dwell_10pct", free_fraction=0.10, bound_dwell=(0.005, 0.030, 0.135, 0.5, 1.0)),  # L45
)


@dataclass(frozen=True)
class SweepPoint:
    """Rates for one sweep point. ``k_layer`` needs the lattice's layer factor."""

    sweep: str
    free_fraction: float
    bound_dwell: float  # s, 1/k_off (0 for free-only points)
    free_dwell: float  # s, 1/k_on (inf for free-only points)
    k_off: float
    k_on: float
    k_layer: float
    d_free: float | None = None  # µm²/s override; None = diffusion.d_free

    @property
    def free_only(self) -> bool:
        return self.k_on == 0.0

    @property
    def label(self) -> str:
        d = "" if self.d_free is None else f", D_free {self.d_free:.3g} µm²/s"
        if self.free_only:
            return f"{self.sweep}: free only (never binds){d}"
        return (f"{self.sweep}: {self.free_fraction:.0%} free, bound {self.bound_dwell * 1e3:.4g} ms, "
                f"free {self.free_dwell * 1e3:.3g} ms{d}")


@dataclass(frozen=True)
class Config:
    name: str = "unnamed"
    geometry: Geometry = field(default_factory=Geometry)
    receptors: Receptors = field(default_factory=Receptors)
    diffusion: Diffusion = field(default_factory=Diffusion)
    membrane: Membrane = field(default_factory=Membrane)
    stepping: Stepping = field(default_factory=Stepping)
    run: Run = field(default_factory=Run)
    sweeps: tuple = DEFAULT_SWEEPS

    def __post_init__(self):
        validate(self)

    def sweep_points(self, layer_factor: float) -> list[SweepPoint]:
        """All sweep points, sweeps in config order. k_off = 1/bound_dwell, k_on = 1/free_dwell,
        k_layer = k_on · V_free/V_layer (L28, L29)."""
        out = []
        for sw in self.sweeps:
            for p in sw.points():
                if sw.free_only:
                    k_on = k_off = 0.0
                else:
                    k_on, k_off = 1.0 / p["free_dwell"], 1.0 / p["bound_dwell"]
                out.append(SweepPoint(sweep=sw.name, k_off=k_off, k_on=k_on, k_layer=k_on * layer_factor, **p))
        return out

    def sweep_point(self, index: int, layer_factor: float) -> SweepPoint:
        return self.sweep_points(layer_factor)[index]

    @property
    def n_sweep_points(self) -> int:
        return sum(len(sw.points()) for sw in self.sweeps)

    def find_point(self, sweep: str, **values) -> int:
        """Index of the point in ``sweep`` matching ``values`` (e.g. bound_dwell=1.0)."""
        for i, p in enumerate(self.sweep_points(1.0)):
            if p.sweep == sweep and all(math.isclose(getattr(p, k), v, rel_tol=1e-6) for k, v in values.items()):
                return i
        raise KeyError(f"no point {values} in sweep {sweep!r}")

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
    unknown = set(d) - set(SECTIONS) - {"name", "sweeps"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    kwargs = {"name": d.get("name", "unnamed")}
    for key, cls in SECTIONS.items():
        section = dict(d.get(key) or {})
        allowed = {f.name for f in dataclasses.fields(cls)}
        bad = set(section) - allowed
        if bad:
            raise ValueError(f"unknown keys in '{key}': {sorted(bad)}")
        kwargs[key] = cls(**section)
    if "sweeps" in d:
        sweeps = []
        for name, spec in (d["sweeps"] or {}).items():
            spec = dict(spec or {})
            bad = set(spec) - set(SWEEP_KEYS)
            if bad:
                raise ValueError(f"unknown keys in sweep '{name}': {sorted(bad)}")
            sweeps.append(Sweep(name, **{k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()}))
        kwargs["sweeps"] = tuple(sweeps)
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
        (r.n_per_cell >= 0, "receptors.n_per_cell must be >= 0 (0 = no receptors, e.g. for D_eff calibration)"),
        (0 < r.capture_radius < g.gap / 2, "capture_radius must be in (0, gap/2)"),
        (cfg.diffusion.d_free > 0 and cfg.diffusion.d_bound > 0, "diffusion coefficients must be positive"),
        # Layers of facing cells must not overlap, or "nearest membrane" becomes ambiguous.
        (0 < m.binding_layer < g.gap / 2, "binding_layer must be in (0, gap/2)"),
        (s.mode in ("adaptive", "fixed"), f"stepping.mode must be 'adaptive' or 'fixed', got {s.mode!r}"),
        (0 < s.distance_fraction <= 1 and 0 < s.max_rate_dt < 1, "accuracy settings out of range"),
        (s.dt_fixed is None or s.dt_fixed > 0, "dt_fixed must be positive"),
        (run.n_molecules >= 1 and run.t_max > 0, "n_molecules and t_max must be positive"),
        (math.isfinite(run.trajectory_interval) and run.trajectory_interval > 0, "trajectory_interval must be positive"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    errors += sweep_errors(cfg.sweeps)
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))


def sweep_errors(sweeps) -> list[str]:
    errors = []
    if not sweeps:
        return ["at least one sweep is required"]
    names = [sw.name for sw in sweeps]
    if len(set(names)) != len(names):
        errors.append("sweep names must be unique")
    for sw in sweeps:
        given = {k: getattr(sw, k) for k in RATE_KEYS if getattr(sw, k) is not None}
        if sw.d_free is not None:
            vals = sw.d_free if isinstance(sw.d_free, tuple) else (sw.d_free,)
            if not vals or not all(isinstance(x, (int, float)) and x > 0 for x in vals):
                errors.append(f"sweep '{sw.name}': d_free must be positive numbers")
        n_lists = sum(isinstance(v, tuple) for v in list(given.values()) + [sw.d_free])
        if n_lists > 1:
            errors.append(f"sweep '{sw.name}': at most one quantity may be a list")
            continue
        if sw.free_only:
            continue
        if len(given) != 2:
            errors.append(f"sweep '{sw.name}': give exactly two of {RATE_KEYS} "
                          f"(or free_fraction: 1.0 alone for free-only points), got {sorted(given)}")
            continue
        for k, v in given.items():
            vals = v if isinstance(v, tuple) else (v,)
            if not vals or any(not isinstance(x, (int, float)) for x in vals):
                errors.append(f"sweep '{sw.name}': {k} must be numbers")
            elif k == "free_fraction" and not all(0 < x < 1 for x in vals):
                errors.append(f"sweep '{sw.name}': free_fraction must be in (0, 1)")
            elif k != "free_fraction" and not all(x > 0 for x in vals):
                errors.append(f"sweep '{sw.name}': {k} must be positive")
    return errors
