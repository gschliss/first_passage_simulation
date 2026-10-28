# Handoff: resuming this project in a new Claude Code session

Written 2026-10-28, at decision-log entry **L87**. Start a new session by asking Claude to read
this file, then `docs/model_notes.md` (the decision log, L1–L87), then `CLAUDE.md`.

## 1. Setting up on a new machine

```bash
git clone https://github.com/gschliss/first_passage_simulation.git
cd first_passage_simulation
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
NUMBA_NUM_THREADS=6 .venv/bin/python -m pytest -q    # 73 tests; the first run compiles Numba kernels
```

- The repository is **private**, so the new machine needs GitHub access, for example `gh auth login`.
- The repository is named `first_passage_simulation`. On the original machine the local folder was renamed
  to `first_passage` (2026-10-19), at `~/Dropbox (Personal)/Lab/GS_lab/data/computation/first_passage`.
  If you work from that Dropbox folder rather than a fresh clone, rebuild `.venv` there: an editable
  install records absolute paths, and a moved or synced `.venv` points at the wrong place.
- **`results/` is gitignored and not on GitHub** (about 450 MB of Parquet runs and figures). It lives in
  the Dropbox folder. On a machine without that Dropbox, every result can be regenerated with the
  commands in section 4, but the figures named in the decision log will not exist until you do.

**Suggested first message to Claude on the new machine:** "Read docs/handoff.md, docs/model_notes.md
and CLAUDE.md, then summarize where the project stands before doing anything."

## 2. The question and the model in one paragraph

How does switching between dimensions affect first-passage time to receptors in a packed lattice of
cells?
- **Geometry:** HCP spheres in 3D, or hexagonally packed disks in 2D. Cells have a 10 µm radius with
  0.5 µm gaps.
- **Switching walker:** free 3D diffusion in the gaps alternates with 2D diffusion on a membrane (in 2D:
  free 2D diffusion and 1D diffusion on the cell outline).
- **Release and targets:** molecules start on a central source cell. The targets are every cell in one
  ring.
- **Comparison:** the switching walker is compared with free-only walkers. One is at **matched D_eff**
  (D_free = D_eff / φ, φ being the packing slow-down of a free walker). The other is at a **fixed
  free-only D**.
- **Pairing:** both walkers run over **10 shared receptor layouts × 20 molecules = 200 molecules per
  condition**, and switching ÷ free-only ratios get 95% CIs by resampling layouts.

## 3. Current parameters (since L84, L86)

| Parameter | Value | Where |
|---|---|---|
| Free D | **1 µm²/s** (was 10 until L83) | `--d-free 1` (config default is still 10) |
| Bound (membrane) D | 0.5 µm²/s | `configs/*.yaml` |
| Free fraction | 5% | `--free-fraction`, default 0.05 |
| Bound dwells | 50, 100, 300 ms, 1, 4, 10, 30 s | `--dwells` |
| Capture radius | 2 nm | `--sizes 2` |
| Receptors per target cell | 1000 in 3D, 1 in 2D | `--receptors-per-cell` |
| Target ring | 2 (also run at 1 and 3) | `--target-ring` |
| Capture rule | first find, or **5th find** with re-arm at 10 nm (3D only, L86) | `--finds 5 --rearm 0.010` |
| φ (free-only packing factor) | 0.691 (3D), 0.387 (2D) | `--phi` |
| Near-receptor step rule | receptor_fraction 1/32 ("rf32", L71) | `--tag rf32` in run names |
| Threads | **6** (leave 2 cores free) | `NUMBA_NUM_THREADS=6` |

## 4. How to run things

One command runs a full paired experiment: calibration, matched reference, capture runs, summary,
paired ratios and a per-run figure.

```bash
NUMBA_NUM_THREADS=6 .venv/bin/python scripts/dimension_experiment.py \
  --config configs/lattice3d.yaml --n 200 --layouts 10 --sizes 2 --target-ring 2 \
  --receptors-per-cell 1000 --tag rf32 --phi 0.691 --d-free 1 --dwells 0.05,0.1,0.3,1,4,10,30
```

Then a second run against a fixed-D free-only walker reuses the calibration and the switching chunks,
so only the free-only walker is simulated:

```bash
M=results/dimension/<the matched run's folder>
NUMBA_NUM_THREADS=6 .venv/bin/python scripts/dimension_experiment.py <same flags> \
  --d-ref 1 --reuse-calibration $M --reuse-switching $M
```

- **Output:** `results/dimension/<name>/`, holding `calibration.csv`, `calibration_reference.csv`,
  `reference.json`, `summary.csv`, `paired_ratios.csv`, `runs_a2nm/` and `dimension_experiment.png`.
- **Name pattern:** `lattice3d[_finds5][_Dfree1]_ring2_rec1000_dw<dwells in ms>_paired10[_ref1]_rf32`.
- **Finished chunks are reused:** rerunning the same command resumes where it stopped.
- **Calibration does not depend on targets or receptors**, since it runs receptor-free on its own
  12-ring cluster. Reuse it across rings with `--reuse-calibration` from any run with the same switching
  points and free D.
- **Combined figures** go in `results/figures/`:
  - `scripts/plot_paired_ratios.py DIR [DIR ...] --out FIG --t-min 10` combines runs, for example a
    6-dwell and a 30 s run.
  - `scripts/plot_finds_comparison.py --first DIRS --finds DIRS --out FIG` compares first passage with
    5 finds.
- **Validation:** `scripts/check_bp_dense.py --receptors N` (Berg–Purcell capture on one sphere) and
  `scripts/validate_3d_capture.py`.
- **Longer chains:** write a bash script and `nohup bash script.sh > log 2>&1 &`. zsh does not
  word-split a `$VAR` holding several flags, so use bash arrays instead.

## 5. Gotchas found the hard way

- **Config hash:** `run_task` refuses a results folder made with a different config (hash in
  `runs_a2nm/config.json`). New config fields must be left out of `Config.to_dict()` at their defaults
  (see `finds_to_capture`), or every earlier run stops matching.
- **Resuming a matched run** reuses the stored reference D exactly. A CSV round trip changed the last
  bit and with it the hash (fixed in L84).
- **The free-only reference at a fixed D is one walker**, repeated once per dwell column with different
  seeds. Differences between its columns are noise, not a dwell effect.
- **Mean times count only captured molecules.** This flatters free-only walkers, whose slow molecules
  escape instead of being counted (15–35% at first passage, about 70% at 5 finds). Report the fraction
  captured alongside the means.
- **`dimension_experiment.py` per-run figure** handles any number of dwells (fixed in L84). With
  `--finds`, the capture radius is set before validation (fixed in L87).
- **2D multi-find is not implemented.** The config validator rejects `finds_to_capture > 1` in 2D.

## 6. Results so far (details and figure paths in the decision log)

**3D, free D = 10, 5% free, 1000 receptors** (L67–L83; ratio = switching ÷ free-only mean time):

| Target ring | vs matched D_eff | vs free-only D = 10 |
|---|---|---|
| 1 | 0.06–0.17 | 0.56–1.62 (switching loses only at 10 s dwells) |
| 2 | 0.27–0.49 | 2.7–3.8 |
| 3 | 0.49–0.60 | 4.6–5.8 |

At ring 2, the receptor-density series against matched D_eff (1000 → 100,000 per cell) gave about
0.29 → 0.61 → 0.79 → 1.26. Long dwells (1–10 s) weaken switching as free trips and slides reach cell
scale (L81–L83).

**Free D = 1 µm²/s, bound D = 0.5** (L84–L85):

| | Ring 1 | Ring 2 | Ring 3 |
|---|---|---|---|
| 3D vs matched D_eff | 0.12–0.21 | 0.39–0.48 | 0.63–0.73 |
| 3D vs free-only D = 1 | 0.27–0.43 | 0.85–1.10 | 1.13–1.53 |
| 2D vs matched D_eff | | 0.88–1.04 (no difference) | |
| 2D vs free-only D = 1 | | 1.15–1.67 | |

Lowering free D moved the break-even against plain free diffusion out from about ring 1 to about
ring 2. In 2D, switching gives no advantage at D = 1.

**5 finds, 3D ring 2, free D = 1** (L86–L87):
- Switching still captures 100%, and the 4 extra finds take only 4–12 s.
- Both free-only walkers capture just 26–32%; the rest escape before their 5th find.
- Ratio vs matched D_eff: 0.16–0.18 (first passage 0.39–0.48).
- Ratio vs D = 1: 0.28–0.50 (first passage 0.85–1.44).
- At a 30 s dwell, first passage gives 0.48 vs matched and 1.44 vs D = 1.

## 7. How the user likes to work

- **Decide modeling choices together.** Propose options with their trade-offs and wait for agreement
  before coding physics or design changes. Use click-through questions for real choices, and make every
  assumption visible so it can be audited.
- **Conceptual questions get conceptual answers.** When asked "why" or "where would X matter", answer
  and stop; don't launch or set up simulations unless asked.
- **Keep runs small:** about 200 molecules per condition, 6 threads.
- **Figures:**
  - Always include the cumulative capture curves.
  - Use the stacked "where the time goes" bars: light for reaching a target cell, dark for finding the
    receptor.
  - Use log time axes, starting at 10 s for 3D (1 s for ring 1).
  - Show figures when they're ready, and make interim figures when asked for status.
- **Log every decision and result** as a new row in the `docs/model_notes.md` decision log, then commit
  and push. Commit messages end with a `Co-Authored-By` line for Claude.
- **Report problems plainly:** escape bias, noise, failed runs and fixes.

## 8. Open threads

- **5 finds at rings 1 and 3** (3D, free D = 1), to see whether the multi-find advantage changes with
  distance.
- **Multi-find in 2D**: the kernel change hasn't been made.
- **Alternative re-find rules** considered in L86 but not run:
  - 5 *different* receptors;
  - release to the free state after a non-signaling find;
  - a 1-in-5 chance per find instead of exactly 5.
- **A summary figure** across rings and free D.
- **A receptor density near the crossover** (for example 30,000 per cell at free D = 10), offered but not
  run.
