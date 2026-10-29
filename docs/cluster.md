# Running a larger sweep on a cluster

How to clone this project onto a SLURM cluster, reproduce the D = 0.5 matrix
(`docs/report_d05.md`), and then grow it. Read `docs/handoff.md` first for the model itself;
this file is only about getting it to run somewhere bigger.

**The shape of the problem.** One *cell* is one `dimension_experiment.py` invocation: a
switching walker and its free-only reference, over the same shared receptor layouts, at one
point in parameter space. Cells are completely independent, so a matrix of them is a plain
SLURM array. `scripts/sweep.py` enumerates the cells; `slurm/submit_matrix.sh` runs one per
array task.

Everything is resumable. Each (sweep point, layout) chunk is a separate Parquet file and
`run_task` skips chunks already on disk, so a failed or timed-out task is fixed by submitting
the same array again.

---

## 1. Clone and build the environment

```bash
git clone https://github.com/gschliss/first_passage_simulation.git
cd first_passage_simulation
```

Then either conda:

```bash
conda env create -f environment.yml
conda activate fpsim
```

or a venv against a Python ≥ 3.9 module:

```bash
module load python/3.11            # whatever your cluster calls it
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

```bash
mkdir -p slurm/logs                # SLURM will not create this, and jobs fail without it
```

Edit the two placeholder blocks in `slurm/submit_matrix.sh`: the partition/account `#SBATCH`
lines, and the environment activation line.

## 2. Check the install before spending an allocation

```bash
NUMBA_NUM_THREADS=8 python -m pytest -q        # 73 tests, ~15 s after Numba compiles
```

Then reproduce one published number end to end. Base task 1 is the ring-2 reference cell:

```bash
python scripts/sweep.py --plan matrix_d05 --stage base --array-task 1
```

About 10 minutes on 8 cores. It should print a `paired_ratios` line with **t20_switch ≈ 211.5
s**, **t20_ref ≈ 297.5 s**, captured **1.000** and **0.758** — the numbers in the report's
headline. If those match, the cluster build agrees with the machine the report was written on.

Results do not depend on the thread count: each molecule draws from its own seeded RNG, which
was checked bit-for-bit across 6 and 8 threads (L89). `--cpus-per-task` is therefore free to
differ from the 8 used locally.

## 3. Look at the plan

```bash
python scripts/sweep.py --plan matrix_d05 --list
```

87 cells in 19 *reuse groups*. A group is a set of cells that share one free-only reference:
the reference never binds, so it does not depend on the free fraction or the dwell value (L95,
verified bit-identical). It does depend on the ring, receptor count, `--finds`, `--d-ref`,
`--n`, `--layouts`, `t_max` and the *number* of dwells.

That matters because **the reference is ~97% of the compute** (L95: 9.8 min at N = 1 and 19.9
min at N = 5 per 400 molecules, against 0.3 min for the switching arm). So there are two ways
to run a matrix:

| | arrays | core-hours | wall time |
|---|---|---|---|
| `--stage all` | one, no dependency | ~5× | one slowest cell |
| `--stage base` then `--stage rest` | two, second depends on first | ~1× | two slowest cells |

**Use `--stage all` unless your allocation is tight.** On a cluster the extra core-hours buy
you a single dependency-free array, and the whole matrix lands in the time of its slowest cell
(~25 min for this plan). The two-stage path exists for when core-hours are the scarce thing.

## 4. Submit

Simple path:

```bash
N=$(python scripts/sweep.py --plan matrix_d05 --stage all --print-n-tasks)
sbatch --array=0-$((N-1))%40 slurm/submit_matrix.sh matrix_d05 all
```

The `%40` caps concurrent tasks; drop or raise it to suit the queue.

Two-stage path, with SLURM enforcing the ordering:

```bash
NB=$(python scripts/sweep.py --plan matrix_d05 --stage base --print-n-tasks)
NR=$(python scripts/sweep.py --plan matrix_d05 --stage rest --print-n-tasks)
JOB=$(sbatch --parsable --array=0-$((NB-1)) slurm/submit_matrix.sh matrix_d05 base)
sbatch --dependency=afterok:$JOB --array=0-$((NR-1)) slurm/submit_matrix.sh matrix_d05 rest
```

`--stage rest` resolves each cell's reference directory by asking the driver
(`--print-out-dir`), so the two stages agree on naming without duplicating the rules.

### Resubmitting

Tasks skip finished chunks, so after failures just submit the same array again. To find which
cells are incomplete:

```bash
python scripts/sweep.py --plan matrix_d05 --stage all --list | tail -n +2 | wc -l
ls -d results/dimension/*_d05/paired_ratios.csv | wc -l     # how many finished
```

## 5. Collect the results

```bash
bash scripts/make_figures.sh        # writes docs/figures/, prints a cell count per figure
```

`results/` is gitignored and never leaves the cluster on its own. Either run the figure script
there and copy back `docs/figures/`, or bring the raw output home:

```bash
rsync -av --info=progress2 cluster:first_passage_simulation/results/dimension/ results/dimension/
```

Output is small: ~1.5 MB per cell (40 Parquet chunks plus the per-run figure), 154 MB for the
whole 87-cell matrix.

## 6. Growing the sweep

Edit `PLANS` at the top of `scripts/sweep.py`. A plan is `common` (flags every cell gets) plus
`blocks`, and each block is a cross product of axis values. Cells that come out identical
across blocks are deduplicated, so overlapping blocks are fine and intentional.

To widen an existing axis, add values:

```python
FREES = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
```

To add a region, add a block — e.g. the dwell × capture-efficiency interaction the report flags
as unexplored:

```python
dict(free_fraction=[0.05], finds=[1, 2, 5, 10], target_ring=[2],
     receptors_per_cell=[1000], dwells=["0.1", "1", "10", "100"]),
```

Then re-check the size and the grouping before submitting:

```bash
python scripts/sweep.py --plan matrix_d05 --stage all --print-n-tasks
python scripts/sweep.py --plan matrix_d05 --list | head -30
```

Two things to keep in mind when you enlarge it:

- **Task indices are positional.** Adding values renumbers cells. That is harmless for a fresh
  submission (finished cells are skipped anyway) but means an array id is not a durable name
  for a cell.
- **The figure globs in `scripts/make_figures.sh` are tied to directory naming.** A new axis
  changes directory names and can silently widen a glob — exactly the bug that put `finds2` and
  `finds10` runs into the N = 1 figures in L96. Duplicate cells are now a hard error in
  `plot_grid.py` and `plot_line.py`, so a bad glob fails loudly, but you still have to add the
  globs for any new figure.

### One cell that is too big for one node

A cell runs its chunks in one process, so its ceiling is a single node. If you raise `--n` far
enough that one cell no longer fits in a time limit, split it with `--shard K/M`:

```bash
for k in 0 1 2 3; do
  python scripts/dimension_experiment.py <cell flags> --shard $k/4 &    # or 4 array tasks
done; wait
python scripts/dimension_experiment.py <cell flags> --replot            # summarise and plot
```

Shard `K/M` runs every M-th chunk starting at K. It is a scheduling flag only — it never
reaches the config, so it cannot change a hash or a result (checked bit-for-bit against an
unsharded run). The `--replot` pass at the end reads whatever is on disk and writes
`summary.csv`, `paired_ratios.csv` and the per-run figure.

Pass it through `sweep.py` with `--extra`:

```bash
python scripts/sweep.py --stage all --array-task 7 --extra --shard 0/4
```

## 7. Cluster-specific gotchas

The model gotchas are in `docs/handoff.md` §5 and still apply. These are the ones that only
show up on a cluster:

- **Numba's compile cache on a shared filesystem.** Concurrent tasks writing one cache
  directory corrupt it. `submit_matrix.sh` gives each task its own `NUMBA_CACHE_DIR` under
  `$TMPDIR` and deletes it on exit. Keep that if you write your own submit script.
- **`results/` must be on a filesystem that supports `os.link`.** `jobs.write_once` uses a
  hard link to make `config.json` creation atomic between concurrent tasks. Scratch and home
  filesystems are fine; some FUSE mounts are not.
- **One code version per results directory.** Every chunk records its git commit, and
  `jobs.merge` reports `chunks come from several code versions` if they disagree. Don't pull
  mid-sweep; finish, then pull.
- **Memory is small and flat:** peak RSS ~190 MB even at ring 4 with 10 000 receptors per
  cell. The 4 GB in the submit script is deliberate headroom, not a measured requirement. Ask
  for fewer cores per task and more tasks if the queue rewards it.
- **Wall-clock per cell is dominated by the reference arm and by `--finds`**, not by the free
  fraction. The 0.1%-free corner is cheap (L95), contrary to what you might expect.
- **`validate_3d_capture.py` step 4 is broken**: it hardcodes a run directory deleted in L91.
  Use `--skip-lattice` until it is repointed.

## 8. The older per-config array path

`slurm/submit_sweep.sh` and `scripts/run.py` predate the paired design. They run one *chunk*
per array task from a config file's own `sweeps:` block, writing to `results/runs/<name>/`, and
have no switching-vs-reference pairing, no shared layouts and no t_20. They are still the right
tool for a plain single-walker sweep defined entirely in YAML. Everything in the report came
through the `sweep.py` path described above.
