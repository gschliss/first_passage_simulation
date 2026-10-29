#!/bin/bash
# SLURM array job: one task = one matrix cell (scripts/sweep.py). See docs/cluster.md.
#
# Submit from the project root:
#   N=$(python scripts/sweep.py --plan matrix_d05 --stage all --print-n-tasks)
#   sbatch --array=0-$((N-1))%40 slurm/submit_matrix.sh matrix_d05 all
#
# Arguments: PLAN STAGE [extra args passed through to sweep.py]
#
#SBATCH --job-name=fpsim-sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# --- Edit for your cluster (placeholders; uncomment as needed) ---
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

PLAN=${1:?plan name, e.g. matrix_d05}
STAGE=${2:?stage: base, rest or all}
shift 2

# --- Edit: activate the project environment ---
# module load python/3.11
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate fpsim
# source .venv/bin/activate

# Numba uses exactly the cores SLURM gave this task. Results do not depend on the thread
# count (one RNG seed per molecule), so this is free to differ from the 8 used locally (L89).
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Per-task compile cache on node-local disk: avoids concurrent writes on a shared filesystem.
# $L_SCRATCH is Sherlock's node-local SSD, purged when the job ends; TMPDIR elsewhere.
export NUMBA_CACHE_DIR=${L_SCRATCH:-${TMPDIR:-/tmp}}/fpsim_numba_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT

echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID}  threads ${NUMBA_NUM_THREADS}  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python scripts/sweep.py --plan "$PLAN" --stage "$STAGE" --array-task "${SLURM_ARRAY_TASK_ID}" "$@"
