#!/bin/bash
# Pilot for the receptors-on-every-cell design (L117): tests, a bit-for-bit check that the
# default path is unchanged, then 400-molecule pilot cells to measure how many molecules reach
# rings 2 and 3 and what each costs. Pilot output goes to results/pilot_allrec/, never mixed
# with production chunks.
#
#   sbatch slurm/pilot_allrec.sh
#
#SBATCH --job-name=fpsim-pilot-allrec
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_CACHE_DIR=${L_SCRATCH:-${TMPDIR:-/tmp}}/fpsim_numba_${SLURM_JOB_ID}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
echo "host $(hostname)  threads ${NUMBA_NUM_THREADS}  commit $(git rev-parse --short HEAD)"

echo "=== tests"
python -m pytest -q 2>&1 | tail -3

echo "=== default path bit-for-bit: replay stored ring-3 molecules with this code"
R=results/dimension/lattice3d_finds5_ring3_rec1000_dw1000_tmax40000_paired20_ref0.5_d05/runs_a2nm
for spec in "0 0 0" "0 0 7" "1 3 60" "1 3 61"; do
  set -- $spec
  python scripts/replay_molecule.py $R --sweep-index $1 --chunk $2 --molecule $3 \
    --out results/pilot_allrec/replay_s$1_c$2_m$3 2>&1 | grep -E "MISMATCH|outcome|t_end|n_steps|does not" \
    || echo "REPLAY FAILED for $spec"
done

COMMON=(--config configs/lattice3d.yaml --sizes 2 --d-ref 0.5 --skip-calibration --tag d05
        --n 400 --layouts 20 --finds 5 --target-ring 3 --receptors-per-cell 1000 --receptors-on all)
P=results/pilot_allrec
BASE=$P/f5_dw1
echo "=== pilot: 5% free, 1 s (computes the reference)"
/usr/bin/time -f "WALL %e s" python scripts/dimension_experiment.py "${COMMON[@]}" --dwells 1 --t-max 40000 \
  --free-fraction 0.05 --out $BASE | tail -4
for spec in "0.001 1 40000 f0.1_dw1" "0.5 1 40000 f50_dw1" "0.05 600 200000 f5_dw600"; do
  set -- $spec
  echo "=== pilot: free $1, dwell $2 s, t_max $3"
  reuse=(); [ "$3" = 40000 ] && reuse=(--reuse-reference $BASE)
  /usr/bin/time -f "WALL %e s" python scripts/dimension_experiment.py "${COMMON[@]}" --dwells $2 --t-max $3 \
    --free-fraction $1 "${reuse[@]}" --out $P/$4 | tail -4
done

echo "=== arrivals by ring"
python scripts/plot_arrivals.py $P/f0.1_dw1 $P/f5_dw1 $P/f50_dw1 --x free_fraction \
  --out $P/arrivals_pilot_free_fraction.png --title "PILOT: receptors on every cell, 5 finds, 400 molecules"
python - <<'EOF'
import pandas as pd
t = pd.read_csv("results/pilot_allrec/arrivals_pilot_free_fraction.csv")
print(t[["x", "walker", "n", "n_ring_1", "n_ring_2", "n_ring_3", "n_ring_4plus", "n_escaped", "n_still_searching"]].to_string(index=False))
EOF
python scripts/plot_arrivals.py $P/f5_dw1 $P/f5_dw600 --x bound_dwell \
  --out $P/arrivals_pilot_dwell.png --title "PILOT: receptors on every cell, 5 finds, 400 molecules"
python -c "import pandas as pd; t = pd.read_csv('$P/arrivals_pilot_dwell.csv'); print(t[['x','walker','n','n_ring_1','n_ring_2','n_ring_3','n_ring_4plus','n_escaped','n_still_searching']].to_string(index=False))"
