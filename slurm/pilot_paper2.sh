#!/bin/bash
# Pilot for the revised paper experiments (paper_instructions/2026_11_18.txt, L125).
# (1) shard 0 of the costliest ring-4 base cells: real production chunks (write-once, resumable);
# (2) cost of the near-zero stand-ins (0.1 ms and 1 ms dwell, 0.01% free) over a 2000 s window with
#     8 molecules, outside the production folders, to extrapolate to t_max = 100 000 s;
# (3) the two slowest D_eff cells at n = 50.
#
#   sbatch slurm/pilot_paper2.sh
#
#SBATCH --job-name=fpsim-pilot-paper2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_CACHE_DIR=${L_SCRATCH_JOB:-${L_SCRATCH:-${TMPDIR:-/tmp}}}/fpsim_numba_${SLURM_JOB_ID}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
echo "host $(hostname)  threads ${NUMBA_NUM_THREADS}  commit $(git rev-parse --short HEAD)"
P=results/paper_2026_11_18/pilot

# Base cells of paper2 (shard 0/20 of base cell b is task 20*b): 3 = ring 4 shared 5000-receptor
# reference, 11 = ring 4 one receptor per cell, 91 = ring 4 N = 1000, 7 = ring 4 t_max 500 000 s.
for b in 3 11 91 7; do
  echo "=== base cell $b shard 0/20 (switching + reference chunk, 50 molecules each)"
  /usr/bin/time -f "WALL %e s  MAXRSS %M kB" timeout 60m \
    python scripts/sweep.py --plan paper2 --stage base --shards 20 --array-task $((20 * b)) 2>&1 | grep -E "WALL|Traceback|Error"
done 2>&1

COMMON=(--config configs/lattice3d.yaml --sizes 2 --d-ref 0.5 --skip-calibration --tag pilot --n 8 --layouts 1
        --finds 1 --receptors-per-cell 5000 --target-ring 4 --t-max 2000)
for spec in "0.05 0.0001 dw0.1ms" "0.05 0.001 dw1ms" "0.0001 1 f0.01pct" "0.05 1 baseline"; do
  set -- $spec
  echo "=== stand-in: free $1, dwell $2 s, 8 molecules, t_max 2000 s"
  /usr/bin/time -f "WALL %e s" timeout 90m python scripts/dimension_experiment.py "${COMMON[@]}" \
    --free-fraction $1 --dwells $2 --out $P/$3 2>&1 | grep -E "WALL|Error|Traceback|switch +[0-9]|captured" | head -5
done

for c in 26 44; do
  echo "=== D_eff cell $c at n = 50"
  cmd=$(python scripts/sweep.py --plan paper2_deff --array-task $c --dry-run | tail -1)
  /usr/bin/time -f "WALL %e s  MAXRSS %M kB" timeout 90m ${cmd/--n 1000/--n 50} --out $P/deff_cell$c 2>&1 \
    | grep -E "D_eff|WALL|Error"
done
