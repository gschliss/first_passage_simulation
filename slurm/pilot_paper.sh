#!/bin/bash
# Pilot for the paper experiments (L121): tests, then shard 0 of the base cells most likely to be
# expensive at ring 4 (real production chunks: write-once, resumable), switching-only chunk 0 of
# the slowest switching cells, and small D_eff timings in a separate pilot/ folder.
#
#   sbatch slurm/pilot_paper.sh
#
#SBATCH --job-name=fpsim-pilot-paper
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_CACHE_DIR=${L_SCRATCH:-${TMPDIR:-/tmp}}/fpsim_numba_${SLURM_JOB_ID}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
echo "host $(hostname)  threads ${NUMBA_NUM_THREADS}  commit $(git rev-parse --short HEAD)"

echo "=== tests"
python -m pytest -q 2>&1 | tail -2

# Base cells (stage base, --shards 20; shard 0 of base cell b is task 20*b):
#   3 = ring 4, 1000 rec, N = 1 (the shared reference)   51 = ring 4, N = 1000
#  11 = ring 4, 10 receptors   47 = ring 4, 100 000 receptors   7 = ring 4, 278 s dwell, t_max 500 000
for b in 3 51 11 47 7; do
  echo "=== base cell $b shard 0/20 (switching + reference chunk, 50 molecules each)"
  /usr/bin/time -f "WALL %e s  MAXRSS %M kB" timeout 50m \
    python scripts/sweep.py --plan paper --stage base --shards 20 --array-task $((20 * b))
done

# Switching-only chunk 0 (stage all): 3 = ring 4 at 0.1% free, 79 = ring 4 at 1000 s dwell,
# 163 = ring 4, 50% free, N = 1000.
for c in 3 79 163; do
  echo "=== switching-only cell $c chunk 0 (50 molecules)"
  /usr/bin/time -f "WALL %e s  MAXRSS %M kB" timeout 50m \
    python scripts/sweep.py --plan paper --stage all --arm switch --chunks 0:1 --array-task $c
done

# D_eff timing at 50 molecules (production uses 1000), outside the production folders.
for c in 19 9; do
  echo "=== D_eff cell $c at n = 50"
  cmd=$(python scripts/sweep.py --plan paper_deff --array-task $c --dry-run | tail -1)
  cmd=${cmd/--n 1000/--n 50}
  /usr/bin/time -f "WALL %e s  MAXRSS %M kB" timeout 50m \
    $cmd --out results/paper_2026_11_02/pilot/deff_cell$c 2>&1 | grep -E "D_eff|escaped|Calibration only|Error"
done
