#!/bin/bash
# One-off probe before the ring3_t2x matrix (L114): check the cluster build reproduces the
# report's ring-2 numbers, then time shard 0 of the base cells most likely to be expensive.
# Every chunk written here is a real chunk of the final runs (write-once, resumable).
#
#   mkdir -p slurm/logs && sbatch slurm/probe_ring3_t2x.sh
#
#SBATCH --job-name=fpsim-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_CACHE_DIR=${L_SCRATCH:-${TMPDIR:-/tmp}}/fpsim_numba_${SLURM_JOB_ID}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT

echo "host $(hostname)  threads ${NUMBA_NUM_THREADS}  commit $(git rev-parse --short HEAD)"
python scripts/check_env.py

echo "=== reproduce report: matrix_d05 base task 1 (expect t20_switch ~211.5, t20_ref ~297.5)"
/usr/bin/time -f "WALL %e s" python scripts/sweep.py --plan matrix_d05 --stage base --array-task 1

# Base cells: 5 = 100 receptors x 5 finds, 12 = 10 finds, 10 = 100 s dwell at t_max 200 000 s,
# 0 = 100 receptors x 1 find, 4 = 10 000 receptors (cheap end). Shard 0 of base cell b is task 20*b.
for b in 5 12 10 0 4; do
  echo "=== probe base cell $b shard 0/20"
  /usr/bin/time -f "WALL %e s" timeout 25m \
    python scripts/sweep.py --plan ring3_t2x --stage base --shards 20 --array-task $((20 * b))
done
