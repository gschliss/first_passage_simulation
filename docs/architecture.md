# Code architecture (draft for review)

Implements the model in [model_notes.md](model_notes.md) (decisions L1–L34).
Nothing here is coded yet. Plan reviewed and approved 2026-10-17, including the
three choices below (Parquet output, one receptor layout per job, removing v0.1).

## 1. Overview

```
config.yaml ──► Config (validated, derived rates)
                  │
                  ├─► Lattice (cells, rings, ghost sites)   ┐  built once per job,
                  ├─► Receptors (positions per target cell) ├─ plain NumPy arrays
                  │                                         ┘
                  ▼
          Numba kernel: one loop per molecule, parallel over molecules
                  │
                  ├─► per-molecule outcome table   (always)
                  └─► sampled trajectories          (optional, a few molecules)
                  ▼
          analysis/ (capture prob., FPT, checks)  ·  viz/ (figures, animations)
```

Every job is independent (molecules don't interact; receptors aren't consumed,
L19), so a sweep is many jobs whose output files are merged afterwards (L33).

## 2. Package layout

`src/fpsim/` (replaces the v0.1 draft; see §9):

| Module | Responsibility | Notes |
|---|---|---|
| `config.py` | Load YAML into a typed `Config`; validate; compute derived values (`k_off = 1/dwell`, `k_on = 19·k_off`, `k_layer = factor·k_on`, fixed-mode `dt`). | Single source of truth; the full resolved config is saved with every output. |
| `lattice/hex2d.py` | 2D hexagonal cluster: cell centers, ring (neighbor-step) distance, neighbor table, **ghost sites** (missing next ring, for the escape rule L15). | Ring k has 6k cells (check). |
| `lattice/hcp3d.py` | 3D HCP cluster: same interface. | Ring 3 = 96 cells, 5 rings = 587 (check, L26). |
| `lattice/layer.py` | `V_free / V_layer` for the lattice (L29): 79.2 (2D), 151.3 (3D). | Computed, not hard-coded; tested against these values. |
| `receptors.py` | Receptor positions per target cell: 2D equal angles, 3D Fibonacci sphere, each randomly rotated per cell (L18). Lookup table for "nearest receptor" on a cell. | 3D: 4000 per cell → per-cell angular bins so lookup is fast. |
| `kernel/common.py` | Numba helpers shared by 2D/3D: RNG seeding, time-step selection, rate probabilities, event recording. | |
| `kernel/geom2d.py`, `kernel/geom3d.py` | Numba geometry queries: nearest cell (search current cell + neighbors only), distance to membrane, reflection off a cell, escape test (nearest site is a ghost), nearest point on membrane, bound-state step on circle / sphere. | Dimension-specific so Numba can compile tight code. |
| `kernel/run2d.py`, `kernel/run3d.py` | The per-molecule state machine (§3), `prange` over molecules. | One code path for both stepping modes (L34). |
| `io.py` | Write/read outcome tables and trajectories; merge job outputs. | **Parquet** via `pyarrow` (compact, pandas-native). |
| `analysis/` | Capture/escape/timeout fractions; FPT distributions and survival curves with censoring; bootstrap CIs; free-fraction and cost diagnostics; adaptive-vs-fixed comparison (L31). | |
| `viz/` | Cluster drawing (from `sketch_2d_cluster.py`), single-molecule figure + GIF (from `demo_cell_hopping.py`), FPT/survival plots. | Demos become thin wrappers around these. |

## 3. The kernel: one molecule's life

States: `BOUND`, `FREE`, and terminal `CAPTURED`, `ESCAPED`, `TIMEOUT` (L13, L14).

```
init:  uniform point on source cell surface (L16)
       with prob 0.95 BOUND there, else FREE at depth U(0, δ) (L16, L25)

loop until terminal or t ≥ t_max:
  dt = choose_dt(state, position)          # §4; fixed mode: dt = dt_fixed
  BOUND:
    unbind with prob 1 − exp(−k_off·dt)    → FREE at depth U(0, δ), same spot (L25)
    else diffuse along membrane (D_bound)   # circle (2D) / sphere surface (3D)
    if within 10 nm of a receptor center   → CAPTURED (L19)
  FREE:
    diffuse (D_free); reflect off any cell (L21)
    if within 10 nm of a receptor center   → CAPTURED (L19)
    if nearest lattice site is a ghost     → ESCAPED (L15)
    if within δ of a membrane: bind with prob 1 − exp(−k_layer·dt)
                                           → BOUND at nearest membrane point,
                                             possibly a new cell (L22, L28)
  accumulate: time free, # free trips, # cell changes, # steps
```

Bound diffusion on the sphere: Gaussian step in the tangent plane, then
project back to the surface (error ~ (step/R)², negligible at nm steps on 10 µm).

Simplification: receptors sit on membranes, and their 10 nm capture zones lie
inside the 10 nm binding-layer region, so a free molecule near a receptor is
already taking near-membrane steps. Free molecules therefore need no separate
receptor-distance rule; bound molecules use their distance *along the membrane*
to the nearest receptor.

## 4. Time-step selection (L30, L34)

```
adaptive:
  BOUND: dt = min( (¼ · arc distance to nearest receptor)² / (2·D_bound),
                   0.05 / k_off,  dt_cap )
  FREE:  d  = distance to the nearest feature
              (binding layer edge, escape line; 0 if inside the layer)
         dt = min( (¼ · d)² / (2·D_free),  dt_cap )
         dt = max( dt, dt_layer )    where dt_layer = min( (δ/4)² / (2·D_free), 0.05 / k_layer )
         inside the layer: dt = dt_layer
fixed:
  dt = dt_fixed  (default: the strictest of the limits above for this sweep point)
```

`¼` (now ⅛ by default, L37) and `0.05` are config values (`stepping.distance_fraction`,
`stepping.max_rate_dt`), fixed across a sweep and halved for the L31 check.
Binding and unbinding *times* are exact regardless of dt (L37); dt only controls spatial accuracy.

## 5. Randomness and reproducibility

- Each molecule gets its own RNG seed derived from `(run_seed, sweep_index, molecule_id)`
  and re-seeded at the start of its loop, so results don't depend on the number
  of threads or how molecules are split across jobs.
- Receptor rotations are drawn **once per job** from `run_seed`, shared
  by all molecules in that job; different seeds give different layouts, so the
  layout varies across jobs (resolves the "redrawn each run?" open item).

## 6. Outputs

Per molecule (always), one row:

| Column | Meaning |
|---|---|
| `molecule_id`, `seed` | identity, reproducibility |
| `outcome` | captured / escaped / timeout |
| `t_end` | capture, escape or timeout time (s) |
| `final_cell`, `final_ring` | where it ended |
| `receptor_id` | which receptor captured it (if captured) |
| `captured_from` | free or bound at capture |
| `time_free`, `n_trips`, `n_cell_changes`, `n_steps` | diagnostics: free fraction, hopping, cost |

Plus a metadata block: the resolved config, stepping mode and accuracy
settings, code version (git commit), host, wall time.

Optional trajectories: positions/state/cell for the first `n_trajectories`
molecules at a fixed output interval (e.g. 50 ms; free trips additionally at a
finer interval), for animations. This sampling is independent of `dt`.

Storing per-molecule rows (not just summaries) means the choice of which
statistics to compare (open item) can be made later without re-running.

## 7. Running

Same entry point everywhere (L33):

```
python scripts/run.py configs/lattice2d.yaml --sweep-index 0 --chunk 0 --n-molecules 1000
```

- `--sweep-index` picks a point from the config's sweep list (dwell 5/71/1000 ms, …).
- `--chunk` splits one sweep point into independent molecule batches.
- `slurm/submit_sweep.sh` maps `SLURM_ARRAY_TASK_ID` → (sweep index, chunk);
  each task writes `results/<run>/sweep<i>_chunk<j>.parquet`.
- `scripts/merge.py results/<run>/` combines chunks and prints the checks (free
  fraction, outcome fractions).
- Locally: the same `run.py` with small `--n-molecules`; `NUMBA_NUM_THREADS`
  controls cores.

## 8. Testing and validation

Unit tests (fast, `pytest`):
- Lattice: ring sizes (2D 6k; 3D 12, 44, 96, …), neighbor distances = 20.5 µm, ghost ring correct.
- Layer factor: 79.2 (2D), 151.3 (3D).
- Receptors: counts 3 / 4000, coverage ≈ 0.1%, spacing, rotation changes positions but not spacing.
- Geometry: distance to membrane, reflection keeps molecules outside cells, nearest-point insertion.
- Kernel invariants: a bound molecule never changes cell without a free trip; mode toggle produces identical output schema.

Physics checks (slower, `pytest -m slow` or scripts):
- Free diffusion MSD = 2·d·D_free·t in open space; bound MSD on circle/sphere matches theory.
- Free fraction → 5% in a closed patch (no receptors, no escape).
- Reproduce the binding-layer demo (flat gap) and hop statistics (`demo_cell_hopping.py`).
- L31: adaptive vs fixed, and halved accuracy settings, at 5 ms and 1000 ms.

## 9. Changes to the existing code

- Remove the v0.1 draft (`agents/`, `geometry/`, `simulation.py`, `analysis/theory.py`,
  `scripts/run_simulation.py`, `configs/baseline_sphere.yaml`, `tests/test_spherical_shell.py`).
  It modeled a different problem (concentric shell) and none of it fits the kernel design.
  It stays in git history.
- Fold `scripts/sketch_2d_cluster.py` and `scripts/demo_cell_hopping.py` into
  `lattice/` and `viz/`; keep the scripts as thin wrappers so the figures can be regenerated.
- Add `numba` and `pyarrow` to `pyproject.toml` and `environment.yml`.

## 10. Build order (each step ends with something to look at)

| Step | Build | Check / visual |
|---|---|---|
| 1 | `config`, `lattice/hex2d`, `receptors` (2D), `layer` | Figure: 5-ring cluster with source, ring-3 targets, receptors, ghost ring. Unit tests. |
| 2 | 2D kernel: free + bound + layer (no receptors, no escape) | Free fraction 5%; reproduce hop demo; adaptive vs fixed agree. |
| 3 | Receptors + escape + outcomes | Pilot run: measure search times (replaces the ~1000 s guess) and cost per molecule. |
| 4 | `run.py`, SLURM script, `merge.py` | Small local sweep, then a cluster sweep at 5 / 71 / 1000 ms. |
| 5 | 3D: `hcp3d`, 3D receptors, 3D kernel | Same checks as 2–3 in 3D; 3D cluster figure. |
| 6 | Analysis + figures + animations | Capture probability and FPT vs dwell, 2D vs 3D; animations of the 2D version (L5). |
