#!/bin/bash
# SLURM array job: one task = one (sweep point, chunk). See slurm/README.md.
#
# Submit from the project root, with the array size printed by run.py:
#   N=$(python scripts/run.py configs/lattice2d.yaml --chunks-per-point 40 --print-n-tasks)
#   sbatch --array=0-$((N-1)) slurm/submit_sweep.sh configs/lattice2d.yaml 40 250
#
# Arguments: CONFIG CHUNKS_PER_POINT MOLECULES_PER_CHUNK [extra run.py args...]
#
#SBATCH --job-name=fpsim
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# --- Edit for your cluster (placeholders; uncomment as needed) ---
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

CONFIG=${1:?config path}
CHUNKS=${2:?chunks per point}
NMOL=${3:?molecules per chunk}
shift 3

# --- Edit: activate the project environment ---
# module load python/3.11            # if your cluster uses modules
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate fpsim
# source .venv/bin/activate

# Numba uses exactly the cores SLURM gave this task.
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Per-task compile cache on local disk: avoids concurrent writes on a shared filesystem.
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/fpsim_numba_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}

echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID}  threads ${NUMBA_NUM_THREADS}  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python scripts/run.py "$CONFIG" --array-task "${SLURM_ARRAY_TASK_ID}" \
    --chunks-per-point "$CHUNKS" --n-molecules "$NMOL" "$@"
rm -rf "$NUMBA_CACHE_DIR"
