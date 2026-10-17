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
│   └── viz/              # figures and animations
├── configs/              # one YAML per experiment
├── scripts/              # command-line entry points and demos
├── tests/                # pytest
├── docs/                 # model notes and architecture
├── data/                 # external inputs (gitignored)
└── results/              # outputs and figures (gitignored)
```

Build status: steps 1–2 of 6 done (config, 2D lattice, receptors, layer factor;
2D kernel with free/bound switching and the binding layer). Receptors and escape
come next (step 3).

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
python scripts/plot_cluster.py configs/lattice2d.yaml   # cluster check figure
python scripts/check_kernel_2d.py                       # kernel validation figure (~15-20 min)
pytest
```

Demos used to make model decisions:
`scripts/demo_binding_models.py` (binding-layer comparison) and
`scripts/demo_cell_hopping.py` (single molecule hopping between cells, with GIF).
