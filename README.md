# first_passage_simulation

Agent-based simulations of how **diffusion dimension** (1D / 2D / 3D search)
affects **first-passage time** to a target in bio-inspired geometries.

## Layout

```
├── src/fpsim/            # the Python package
│   ├── geometry/         # domains: start region, walls, target
│   │   ├── base.py       #   Geometry interface every domain implements
│   │   └── sphere.py     #   SphericalShell: validation geometry, any dimension
│   ├── agents/           # walker models (vectorized ensembles)
│   ├── simulation.py     # time-stepping loop -> first-passage times
│   ├── analysis/         # FPT statistics + analytical benchmarks
│   └── io.py             # configs in, results out
├── configs/              # YAML experiment definitions (one file per experiment)
├── scripts/              # command-line entry points
├── notebooks/            # exploration and figures
├── tests/                # pytest checks, incl. simulation vs. theory
├── docs/                 # model notes, derivations, decisions
├── data/                 # external inputs (gitignored)
└── results/              # simulation outputs (gitignored, timestamped run dirs)
```

## Setup

```bash
conda env create -f environment.yml
conda activate fpsim
```

or with a plain virtualenv:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

## Usage

```bash
python scripts/run_simulation.py configs/baseline_sphere.yaml
pytest
```

Each run writes `results/<timestamp>_<name>/` with a copy of the config,
per-dimension `dimN.npz` files of first-passage times, and `summary.json`.

## Adding a geometry

Subclass `fpsim.geometry.Geometry`, implement `sample_start`,
`apply_boundaries` and `is_absorbed`, register it in
`fpsim/geometry/__init__.py::GEOMETRIES`, and point a config at it.

See [docs/model_notes.md](docs/model_notes.md) for the model and its assumptions.
