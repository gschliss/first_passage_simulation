"""Run one chunk of a sweep, the same way locally and on SLURM (L33).

Pick the chunk explicitly:
    python scripts/run.py configs/lattice2d.yaml --sweep-index 0 --chunk 0 --n-molecules 250

or from a SLURM array id (points vary slowest):
    python scripts/run.py configs/lattice2d.yaml --array-task $SLURM_ARRAY_TASK_ID \\
        --chunks-per-point 40 --n-molecules 250

Run every chunk in sequence (local testing):
    python scripts/run.py configs/mini2d.yaml --all --chunks-per-point 2 --n-molecules 20 --t-max 200

Print the array size to pass to sbatch:
    python scripts/run.py configs/lattice2d.yaml --chunks-per-point 40 --print-n-tasks

Results go to results/runs/<config name>/ unless --out-dir is given. Chunks whose
output already exists are skipped (use --overwrite to redo them).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fpsim import load_config
from fpsim.jobs import Task, n_tasks, run_task, stem_for, task_from_array_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    which = p.add_mutually_exclusive_group()
    which.add_argument("--sweep-index", type=int)
    which.add_argument("--array-task", type=int)
    which.add_argument("--all", action="store_true", help="run every chunk of every sweep point in sequence")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--chunks-per-point", type=int, default=40)
    p.add_argument("--n-molecules", type=int, default=250, help="molecules per chunk")
    p.add_argument("--t-max", type=float, default=None, help="override run.t_max (s)")
    p.add_argument("--n-trajectories", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print-n-tasks", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.print_n_tasks:
        print(n_tasks(cfg, args.chunks_per_point))
        return
    out_dir = args.out_dir or PROJECT_ROOT / "results" / "runs" / cfg.name

    if args.all:
        tasks = [Task(i, c) for i in range(cfg.n_sweep_points) for c in range(args.chunks_per_point)]
    elif args.array_task is not None:
        tasks = [task_from_array_id(args.array_task, args.chunks_per_point, cfg)]
    elif args.sweep_index is not None:
        tasks = [Task(args.sweep_index, args.chunk)]
    else:
        p.error("give --sweep-index, --array-task or --all")

    overrides = {k: v for k, v in (("t_max", args.t_max), ("n_trajectories", args.n_trajectories)) if v is not None}
    for task in tasks:
        t0 = time.time()
        path = run_task(cfg, task, args.n_molecules, out_dir, overwrite=args.overwrite, **overrides)
        status = "exists, skipped" if path is None else f"done in {time.time() - t0:.0f}s"
        print(f"sweep {task.sweep_index} chunk {task.chunk}: {status} -> {stem_for(out_dir, task)}.parquet", flush=True)


if __name__ == "__main__":
    main()
