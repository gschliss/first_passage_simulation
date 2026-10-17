"""Merge the chunks of a run directory and print checks and a per-point summary.

Usage:
    python scripts/merge.py results/runs/lattice2d --chunks-per-point 40

Writes <run>/merged.parquet (all molecules) and <run>/summary.csv, and exits with a
non-zero status if any check fails (wrong config, duplicates, missing chunks).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fpsim.jobs import merge


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--chunks-per-point", type=int, default=None, help="expected chunks per point (to list missing ones)")
    args = p.parse_args()

    molecules, summary, problems = merge(args.run_dir, args.chunks_per_point)
    if len(molecules):
        molecules.to_parquet(args.run_dir / "merged.parquet", index=False)
        summary.to_csv(args.run_dir / "summary.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
            print(summary.drop(columns=["sweep"]).to_string(index=False))
        drift = summary.time_free / summary.free_fraction - 1
        print(f"\ntime-free / target free fraction - 1: min {drift.min():+.1%}, max {drift.max():+.1%} (L31 drift check)")
        print(f"{len(molecules)} molecules from {molecules.groupby(['sweep_index', 'chunk']).ngroups} chunks")
    for msg in problems:
        print(f"PROBLEM: {msg}", file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
