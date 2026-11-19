#!/bin/bash
# Last step of the revised paper pipeline (L125): figures and tables, then commit and push them.
#SBATCH --job-name=paper2-finish
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
python scripts/paper2_figures.py
git add paper_figures/2026_11_18
if ! git diff --cached --quiet; then
  n=$(ls paper_figures/2026_11_18/*/*.pdf 2>/dev/null | wc -l)
  git -c user.name=gschliss -c user.email=4913489+gschliss@users.noreply.github.com commit -q \
    -m "Revised paper figures and tables (L125), built by slurm/paper2_finish.sh" \
    -m "$n PDF figures plus summary CSVs in paper_figures/2026_11_18/." \
    -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
  timeout 300 git push origin main && echo "pushed" || echo "PUSH FAILED (commit is local; push by hand)"
fi
