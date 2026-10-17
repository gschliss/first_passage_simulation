# Project notes for Claude

- Package code lives in `src/fpsim/`; scripts and notebooks import from it.
- Positions are always `(n_agents, dim)` arrays; keep agent updates vectorized.
- New geometries subclass `fpsim.geometry.Geometry` and register in `GEOMETRIES`.
- Experiments are defined by YAML in `configs/`; outputs go to `results/` (gitignored).
- Validate new physics against an analytical case where one exists (see `analysis/theory.py`).
- Record modeling decisions and open questions in `docs/model_notes.md`.
- Run tests with `pytest` from the project root.
