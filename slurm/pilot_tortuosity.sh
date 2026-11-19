#!/bin/bash
# Cost of the narrowest gaps in experiment 6 (L127): free-only, receptor-free D runs at 40 molecules over
# the full 15 000 s, outside the production folders. Production cells 0-6 (1-15 nm) and 30 wait on this.
#SBATCH --job-name=pilot-tortuosity
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_CACHE_DIR=${L_SCRATCH_JOB:-${L_SCRATCH:-/tmp}}/fpsim_numba_$SLURM_JOB_ID; trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
for t in 0 2 4; do   # gaps 1, 3, 7.5 nm
  cmd=$(python scripts/sweep.py --plan paper2_tortuosity --array-task $t --dry-run | tail -1)
  echo "=== cell $t: $(echo "$cmd" | grep -o 'gap-nm [^ ]*')"
  /usr/bin/time -f "WALL %e s  MAXRSS %M kB" timeout 150m ${cmd/--n 1000/--n 40} \
    --out results/paper_2026_11_18/pilot_tortuosity/cell$t 2>&1 | grep -E "D_eff|WALL|Error|Traceback"
done
