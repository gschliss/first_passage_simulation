"""Enumerate a parameter matrix as independent cells and run one of them (L97).

One cell is one `dimension_experiment.py` invocation: a switching walker and its free-only
reference over the same shared receptor layouts. Cells do not talk to each other, so a whole
matrix is a SLURM array over cell indices. See `docs/cluster.md`.

    python scripts/sweep.py --plan matrix_d05 --list
    python scripts/sweep.py --plan matrix_d05 --stage base --print-n-tasks
    python scripts/sweep.py --plan matrix_d05 --stage base --array-task 0

A plan is a list of blocks; each block is a cross product of axis values. Cells that come out
identical (blocks overlap by design -- the density and distance grids share a corner) are
deduplicated, first occurrence winning. To enlarge a sweep, add axis values or a block below.

Reference reuse and the two stages
----------------------------------
The free-only reference never binds, so its trajectories do not depend on the free fraction or
the bound dwell (L95, verified bit-identical). It also costs ~97% of the compute (L95). Cells
therefore fall into reuse groups: everything that shares a reference. Within a group, one
"base" cell computes the reference and the rest copy it.

    --stage base   the group representatives; no dependencies, run these first
    --stage rest   the others, each with --reuse-reference pointing at its base cell
    --stage all    every cell, each computing its own reference (no dependency, ~5x the
                   core-hours, one array instead of two -- usually the better trade)

Only `--stage rest` requires that `--stage base` has finished.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT_ROOT / "scripts" / "dimension_experiment.py"

# Axis values that appear in several blocks.
FREES = [0.001, 0.01, 0.05, 0.2, 0.5]
BASE_FREE = 0.05  # the free fraction whose cell computes the reference for its group

# Flags that do NOT change the free-only reference, so cells differing only in these share one.
# The reference never binds, so neither the free fraction nor the dwell *value* reaches it. The
# dwell *count* does, because molecule seeds depend on the sweep index, so group_key keeps it.
# t_max is deliberately absent: a different ceiling is a different reference run.
NOT_IN_REFERENCE = ("free_fraction", "dwells")

PLANS = {
    # The matrix behind docs/report_d05.md (L96). 87 cells.
    "matrix_d05": dict(
        common=dict(config="configs/lattice3d.yaml", sizes="2", d_ref=0.5, skip_calibration=True,
                    tag="d05", n=400, layouts=20, dwells="1"),
        blocks=[
            dict(free_fraction=FREES, finds=[1, 5], target_ring=[1, 2, 3, 4],
                 receptors_per_cell=[1000]),
            dict(free_fraction=FREES, finds=[1, 5], target_ring=[2],
                 receptors_per_cell=[100, 300, 1000, 3000, 10000]),
            # Dwell line. 100 s and 600 s need a higher ceiling, which changes the run (and the
            # reference), so they form their own groups rather than reusing the 20 000 s one.
            dict(free_fraction=[BASE_FREE], finds=[1], target_ring=[2], receptors_per_cell=[1000],
                 dwells=["0.01", "0.1", "1", "10"]),
            dict(free_fraction=[BASE_FREE], finds=[1], target_ring=[2], receptors_per_cell=[1000],
                 dwells=["100", "600"], t_max=100000),
            # Dwell cline at 5 finds. No t_max variants: L109 found nothing times out at the
            # default, so the 100 s and 600 s points need no raised ceiling (the N = 1 line
            # above uses one, which turned out to be unnecessary but is kept so its runs stay
            # addressable by their existing config hashes).
            dict(free_fraction=[BASE_FREE], finds=[5], target_ring=[2], receptors_per_cell=[1000],
                 dwells=["0.01", "0.1", "1", "10", "100", "600"]),
            dict(free_fraction=[BASE_FREE], finds=[1, 2, 5, 10], target_ring=[2],
                 receptors_per_cell=[1000]),
        ],
    ),
    # The ring-2 blocks of matrix_d05 (density grid, dwell line, efficiency line) moved to ring 3,
    # with every t_max doubled (L114). 57 cells in 13 reuse groups.
    "ring3_t2x": dict(
        common=dict(config="configs/lattice3d.yaml", sizes="2", d_ref=0.5, skip_calibration=True,
                    tag="d05", n=400, layouts=20, dwells="1", t_max=40000),
        blocks=[
            dict(free_fraction=FREES, finds=[1, 5], target_ring=[3],
                 receptors_per_cell=[100, 300, 1000, 3000, 10000]),
            dict(free_fraction=[BASE_FREE], finds=[1], target_ring=[3], receptors_per_cell=[1000],
                 dwells=["0.01", "0.1", "1", "10"]),
            dict(free_fraction=[BASE_FREE], finds=[1], target_ring=[3], receptors_per_cell=[1000],
                 dwells=["100", "600"], t_max=200000),
            dict(free_fraction=[BASE_FREE], finds=[1, 2, 5, 10], target_ring=[3],
                 receptors_per_cell=[1000]),
        ],
    ),
    # The ring3_t2x dwell line at N = 5 (L115). The 100/600 s block comes first so its new
    # reference is base task 0; the 1 s N = 5 cell is ring3_t2x's own base cell (already run).
    # Receptors on every cell but the source, 5 finds, arrivals read at rings 1-3 of a ring-3-sized
    # cluster (L117-L118). The switching walker almost never passes ring 1 (pilot: 0 of 1200 reached
    # ring 3) while the free-only one is ~200x costlier per molecule, so the two arms are sampled
    # separately: allrec_n5 runs only switching chunks (--arm switch) at 1 000 000 molecules over 200
    # layouts; allrec_n5_ref is two complete small cells that supply the reference, one per t_max.
    "allrec_n5": dict(
        common=dict(config="configs/lattice3d.yaml", sizes="2", d_ref=0.5, skip_calibration=True,
                    tag="d05", n=1000000, layouts=200, dwells="1", t_max=40000, finds=5,
                    target_ring=3, receptors_per_cell=1000, receptors_on="all"),
        blocks=[
            dict(free_fraction=FREES),
            dict(free_fraction=[BASE_FREE], dwells=["0.01", "0.1", "1", "10"]),
            dict(free_fraction=[BASE_FREE], dwells=["100", "600"], t_max=200000),
        ],
    ),
    "allrec_n5_ref": dict(
        common=dict(config="configs/lattice3d.yaml", sizes="2", d_ref=0.5, skip_calibration=True,
                    tag="d05", n=4000, layouts=20, dwells="1", t_max=40000, finds=5,
                    target_ring=3, receptors_per_cell=1000, receptors_on="all",
                    free_fraction=BASE_FREE),
        blocks=[dict(dwells=["1"]), dict(dwells=["100"], t_max=200000)],
    ),
    "ring3_t2x_n5dwell": dict(
        common=dict(config="configs/lattice3d.yaml", sizes="2", d_ref=0.5, skip_calibration=True,
                    tag="d05", n=400, layouts=20, dwells="1", t_max=40000),
        blocks=[
            dict(free_fraction=[BASE_FREE], finds=[5], target_ring=[3], receptors_per_cell=[1000],
                 dwells=["100", "600"], t_max=200000),
            dict(free_fraction=[BASE_FREE], finds=[5], target_ring=[3], receptors_per_cell=[1000],
                 dwells=["1", "0.01", "0.1", "10"]),
        ],
    ),
}


def expand(plan):
    """Blocks -> an ordered, deduplicated list of cells (each a dict of driver flags)."""
    cells, seen = [], set()
    for block in plan["blocks"]:
        axes = {k: (v if isinstance(v, list) else [v]) for k, v in block.items()}
        for combo in itertools.product(*axes.values()):
            cell = dict(plan["common"], **dict(zip(axes, combo)))
            key = tuple(sorted((k, str(v)) for k, v in cell.items()))
            if key not in seen:
                seen.add(key)
                cells.append(cell)
    return cells


def group_key(cell):
    """Cells with the same key share a free-only reference."""
    n_dwells = len(str(cell.get("dwells", "")).split(","))
    return (tuple(sorted((k, str(v)) for k, v in cell.items() if k not in NOT_IN_REFERENCE)),
            n_dwells)


def split_stages(cells):
    """(base cells, [(cell, its base cell), ...]). The base of a group is its BASE_FREE cell if
    the group has one, else its first cell."""
    groups = {}
    for cell in cells:
        groups.setdefault(group_key(cell), []).append(cell)
    base_of, bases = {}, []
    for key, members in groups.items():
        base = next((c for c in members if c.get("free_fraction") == BASE_FREE), members[0])
        bases.append(base)
        base_of[key] = base
    rest = [(c, base_of[group_key(c)]) for c in cells if c not in bases]
    # Keep the plan's order so a task index is stable as long as the plan is.
    bases = [c for c in cells if c in bases]
    return bases, rest


def flags(cell):
    out = []
    for k, v in cell.items():
        name = "--" + k.replace("_", "-")
        if v is True:
            out.append(name)
        elif v is not False and v is not None:
            out += [name, str(v)]
    return out


def out_dir(cell):
    """Ask the driver where this cell lands; the naming rules live there, not here."""
    r = subprocess.run([sys.executable, str(DRIVER), *flags(cell), "--print-out-dir"],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"could not resolve the run directory:\n{r.stderr}")
    return r.stdout.strip()


def label(cell):
    return " ".join(f"{k}={v}" for k, v in sorted(cell.items())
                    if k in ("free_fraction", "finds", "target_ring", "receptors_per_cell",
                             "dwells", "t_max"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="matrix_d05", choices=sorted(PLANS))
    p.add_argument("--stage", default="all", choices=("base", "rest", "all"))
    p.add_argument("--array-task", type=int, help="run this cell (0-based)")
    p.add_argument("--print-n-tasks", action="store_true", help="array size for sbatch")
    p.add_argument("--list", action="store_true", help="print every cell in this stage")
    p.add_argument("--dry-run", action="store_true", help="with --array-task, print the command only")
    p.add_argument("--shards", type=int, default=1,
                   help="split every cell into M tasks via the driver's --shard K/M (scheduling only, L114). "
                        "Sharded cells need a --replot pass once all their shards are done")
    p.add_argument("--pack", type=int, default=1,
                   help="one array task runs G consecutive tasks in turn, so short tasks add up to a "
                        "worthwhile job (L114)")
    p.add_argument("--arm", default="both", choices=("both", "switch"),
                   help="switch: run only the switching walker's chunks, one task per chunk (L118). "
                        "Valid for single-dwell cells; the reference comes from elsewhere")
    p.add_argument("--chunks", default=None, metavar="A:B",
                   help="with --arm switch, only chunks (layouts) A..B-1")
    p.add_argument("--replot", action="store_true",
                   help="summarise and plot each cell of the stage from the chunks on disk (no shards)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is appended to the driver command (e.g. --extra --shard 0/4)")
    args = p.parse_args()

    cells = expand(PLANS[args.plan])
    bases, rest = split_stages(cells)
    if args.stage == "base":
        tasks = [(c, None) for c in bases]
    elif args.stage == "rest":
        tasks = rest
    else:
        tasks = [(c, None) for c in cells]

    if args.arm == "switch":
        # A single-dwell cell has 2 sweep points, so its chunks enumerate as j = point * layouts
        # + layout and the switching ones are j < layouts. Shard j/(2 * layouts) is exactly chunk j.
        if args.replot or args.shards != 1:
            p.error("--arm switch picks its own shards; don't combine it with --shards or --replot")
        expanded = []
        for cell, base in tasks:
            if len(str(cell.get("dwells", "")).split(",")) != 1:
                p.error("--arm switch needs single-dwell cells")
            L = int(cell["layouts"])
            a, b = (int(v) for v in args.chunks.split(":")) if args.chunks else (0, L)
            expanded += [(cell, base, k, 2 * L) for k in range(max(a, 0), min(b, L))]
        tasks = expanded
        m = None
    else:
        m = 1 if args.replot else max(args.shards, 1)
        tasks = [(cell, base, k, m) for cell, base in tasks for k in range(m)]
    g = max(args.pack, 1)
    n_jobs = -(-len(tasks) // g)

    if args.print_n_tasks:
        print(n_jobs)
        return
    if args.list:
        print(f"plan {args.plan}: {len(cells)} cells, {len(bases)} reuse groups "
              f"-> stage {args.stage} has {len(tasks)} tasks in {n_jobs} array tasks")
        for i, (cell, base, k, mk) in enumerate(tasks):
            print(f"{i:4d}  {label(cell)}" + (f"  shard {k}/{mk}" if mk > 1 else "")
                  + (f"   [reuses {label(base)}]" if base else ""))
        return
    if args.array_task is None:
        p.error("give --array-task, --list or --print-n-tasks")
    if not 0 <= args.array_task < n_jobs:
        p.error(f"--array-task must be 0..{n_jobs - 1} for stage {args.stage}; got {args.array_task}")

    failed = 0
    for t in range(args.array_task * g, min((args.array_task + 1) * g, len(tasks))):
        cell, base, k, mk = tasks[t]
        cmd = [sys.executable, str(DRIVER), *flags(cell)]
        if base is not None:
            cmd += ["--reuse-reference", out_dir(base)]
        if args.replot:
            cmd += ["--replot"]
        elif mk > 1:
            cmd += ["--shard", f"{k}/{mk}"]
        cmd += args.extra
        print(f"task {t}/{len(tasks)}: {label(cell)}" + (f"  shard {k}/{mk}" if mk > 1 else ""), flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            failed += subprocess.run(cmd, cwd=PROJECT_ROOT).returncode != 0
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
