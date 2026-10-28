# Project notes for Claude

Start here: read `docs/handoff.md` (current state, how to run things, gotchas), then the decision log
in `docs/model_notes.md` (L1 onward). Add every new decision or result there as a new row.

## Code

- Package code lives in `src/fpsim/`; scripts import from it. Install with `pip install -e ".[dev]"` into `.venv`.
- The simulation kernels are Numba per-molecule loops, parallel over molecules: `src/fpsim/kernel/run2d.py` and `run3d.py`.
  `simulate.py` turns a config plus a sweep point into a per-molecule table.
- Lattices (hexagonal 2D, HCP 3D) are in `src/fpsim/lattice/`; receptor layouts in `receptors.py`;
  D_eff calibration in `deff.py`; chunked runs, merging and config hashes in `jobs.py`.
- Experiments are defined by YAML in `configs/` and overridden by script flags (mainly
  `scripts/dimension_experiment.py`). Outputs go to `results/` (gitignored, not on GitHub).
- New config fields must be left out of `Config.to_dict()` at their defaults, so earlier runs keep their
  config hashes.
- Validate new physics against an analytical case where one exists (`src/fpsim/validation.py`,
  `scripts/check_bp_dense.py`, `scripts/validate_3d_capture.py`). Check that a change with default settings
  reproduces a stored chunk bit-for-bit.
- Run tests with `NUMBA_NUM_THREADS=8 .venv/bin/python -m pytest -q` from the project root.

## Working with the user

- Decide modeling choices together: propose options and trade-offs and wait for agreement before coding
  physics or design changes. Make every assumption visible.
- Conceptual "why / where would X matter" questions: answer and stop; don't start simulations unless asked.
- Keep runs small (about 200 molecules per condition) and use 8 threads (`NUMBA_NUM_THREADS=8`), leaving 2 cores free.
- Visuals first. Figures always include the cumulative capture curves, and use the stacked "where the
  time goes" bars. Show figures as soon as they're ready.
- Report escape fractions alongside mean times (means count captured molecules only), and say plainly
  when something failed or is noise.
- Commit and push after each logged result; commit messages end with a `Co-Authored-By` line for Claude.
