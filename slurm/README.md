# Running sweeps on SLURM

**Two paths. For the paired switching-vs-reference experiments behind `docs/report_d05.md`,
use `slurm/submit_matrix.sh` and read `docs/cluster.md` — not this file.** One array task
there is one matrix cell (`scripts/sweep.py`).

The rest of this file describes the older per-config path (`scripts/run.py`), which predates
the paired design: no switching/reference pairing, no shared layouts, no t_20. It is still the
right tool for a plain single-walker sweep defined entirely in a config's `sweeps:` block.

Each array task runs one chunk (a batch of molecules at one sweep point) with
`scripts/run.py`; chunks are independent, so tasks can run in any order and on any
nodes. Results land in `results/runs/<config name>/`.

## 1. One-time setup on the cluster

```bash
git clone https://github.com/gschliss/first_passage_simulation.git
cd first_passage_simulation
conda env create -f environment.yml      # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
mkdir -p slurm/logs                      # SLURM writes task logs here; it must exist before submitting
```

Edit the placeholders in `slurm/submit_sweep.sh`: partition/account (if your
cluster requires them) and the environment activation line.

## 2. Test locally first (same command, small numbers)

```bash
python scripts/run.py configs/mini2d.yaml --all --chunks-per-point 2 --n-molecules 20 --t-max 200
python scripts/merge.py results/runs/mini2d --chunks-per-point 2
```

## 3. Submit

```bash
N=$(python scripts/run.py configs/lattice2d.yaml --chunks-per-point 40 --print-n-tasks)   # 360 for 9 points
sbatch --array=0-$((N-1)) slurm/submit_sweep.sh configs/lattice2d.yaml 40 250
```

40 chunks × 250 molecules = 10⁴ molecules per sweep point. From the pilot (L40),
a chunk of 250 takes ~15 core-minutes on average (~4 min on 4 cores); the 2 h limit
leaves room for slow chunks. Add `%50` to `--array` (e.g. `0-359%50`) to cap how
many tasks run at once.

## 4. Resubmit failures, then merge

Tasks skip chunks that already have output, so resubmitting the same array only
redoes missing ones. Then:

```bash
python scripts/merge.py results/runs/lattice2d --chunks-per-point 40
```

This writes `merged.parquet` and `summary.csv` and exits non-zero if any chunk is
missing, duplicated, or came from a different config.
