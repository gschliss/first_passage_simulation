#!/bin/bash
# Last step of the paper pipeline (L121): figures and tables, then commit and push them so they
# reach GitHub even if nobody is logged in. Submitted with a dependency on the summary jobs.
#SBATCH --job-name=fpsim-paper-finish
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
ml python/3.12.1
source .venv/bin/activate
bash scripts/make_figures_paper.sh
git add paper_figures/2026_11_02
if ! git diff --cached --quiet; then
  n=$(ls paper_figures/2026_11_02/*/*.png 2>/dev/null | wc -l)
  git -c user.name=gschliss -c user.email=4913489+gschliss@users.noreply.github.com commit -q \
    -m "Paper experiment figures and tables (L121), built by slurm/paper_finish.sh" \
    -m "$n PNG figures plus PDFs and summary CSVs in paper_figures/2026_11_02/." \
    -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
  timeout 300 git push origin main && echo "pushed" || echo "PUSH FAILED (commit is local; push by hand)"
fi
