# first_passage_simulation

Agent-based simulations of how **diffusion dimension** affects **first-passage time**
for ligands searching a packed lattice of cells. Molecules switch between free
diffusion in the gaps between cells and diffusion on cell membranes, and are
captured by receptors on a ring of target cells. A 2D version (disks; free 2D
diffusion ↔ 1D membrane diffusion) and a 3D version (HCP spheres; free 3D ↔
2D membrane) are compared.

- **Model and every decision:** [docs/model_notes.md](docs/model_notes.md)
- **Code plan and build order:** [docs/architecture.md](docs/architecture.md)

## Layout

```
├── src/fpsim/
│   ├── config.py         # typed YAML config + derived rates per sweep point
│   ├── lattice/          # cell clusters: rings, neighbors, ghost ring, layer factor
│   ├── receptors.py      # receptor layouts on target cells
│   ├── kernel/           # Numba per-molecule simulation loops
│   ├── simulate.py       # config + sweep point -> per-molecule results table
│   ├── jobs.py           # split sweeps into chunks, run, merge
│   ├── io.py             # Parquet + JSON results
│   └── viz/              # figures and animations
├── configs/              # one YAML per experiment
├── scripts/              # command-line entry points and demos
├── slurm/                # SLURM array job template and instructions
├── tests/                # pytest
├── docs/                 # model notes and architecture
├── data/                 # external inputs (gitignored)
└── results/              # outputs and figures (gitignored)
```

Build status: steps 1–5 of 6 done (config with named sweeps; 2D hexagonal and
3D HCP lattices, receptors, layer factor; 2D and 3D kernels with free/bound
switching, binding layer, receptor capture and escape; pilots; job runner, SLURM
script and merge). Next: analysis and figures across sweeps (step 6).

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
python scripts/plot_cluster.py configs/lattice2d.yaml                  # cluster check figure (2D or 3D config)
python scripts/check_kernel.py --config configs/lattice2d.yaml         # kernel validation figure (~15-20 min)
python scripts/check_capture.py --dim 2                                # adaptive vs fixed on capture/escape
python scripts/pilot.py --config configs/lattice2d.yaml --n 200 --t-max 5000   # pilot run + figure
pytest
```

A sweep, locally (small) or on SLURM (see [slurm/README.md](slurm/README.md)):

```bash
python scripts/run.py configs/mini2d.yaml --all --chunks-per-point 2 --n-molecules 20 --t-max 200
python scripts/merge.py results/runs/mini2d --chunks-per-point 2
python scripts/plot_capture_curves.py results/runs/mini2d --sweep free_fraction   # capture curves across a sweep
```

Demos used to make model decisions:
`scripts/demo_binding_models.py` (binding-layer comparison) and
`scripts/demo_cell_hopping.py` (single molecule hopping between cells, with GIF).
