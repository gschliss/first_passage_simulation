"""Replay one molecule from a finished run, recording its path (L103).

Production runs store per-molecule summaries only (`n_trajectories: 0`), so there are no
positions to animate. But each molecule has its own RNG seed, derived from
(run seed, sweep index, molecule id) alone (`kernel.common.molecule_seeds`), so re-simulating
a single molecule reproduces *that* molecule's walk exactly — not a statistically similar one.

    python scripts/replay_molecule.py results/dimension/<run>/runs_a2nm \\
        --chunk 0 --molecule 15 --interval 0.05

The config is rebuilt from the run's own `config.json`, so the physics cannot drift from what
was run. The replay is checked against the stored row (seed, outcome, t_end, step count); a
mismatch is a hard error, because an animation of a different walk than the one summarised in
the results would be worse than no animation.

Writes `<out>.npy` (n_frames, 5: x, y, z, state, cell) and `<out>.json` (the stored row plus
what is needed to draw the scene: receptor cells, the capturing receptor, geometry).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpsim.config import from_dict
from fpsim.jobs import Task, stem_for
from fpsim.simulate import simulate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK = ("seed", "outcome", "t_end", "n_steps", "n_trips", "n_cell_changes", "t_first_target")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="a runs_aNNnm directory inside a dimension run")
    p.add_argument("--chunk", type=int, required=True, help="chunk (= receptor layout) the molecule is in")
    p.add_argument("--sweep-index", type=int, default=0, help="0 is the first switching point")
    p.add_argument("--molecule", type=int, required=True, help="molecule_id from the stored table")
    p.add_argument("--interval", type=float, default=None,
                   help="seconds between recorded frames (default: the config's own)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    stem = stem_for(args.run_dir, Task(args.sweep_index, args.chunk))
    stored = pd.read_parquet(stem.with_suffix(".parquet"))
    row = stored[stored.molecule_id == args.molecule]
    if row.empty:
        raise SystemExit(f"molecule {args.molecule} is not in {stem.with_suffix('.parquet').name}; "
                         f"it holds {stored.molecule_id.min()}..{stored.molecule_id.max()}")
    row = row.iloc[0]
    chunk_meta = json.loads(stem.with_suffix(".json").read_text())
    cfg = from_dict(json.loads((args.run_dir / "config.json").read_text())["config"])
    if args.interval is not None:  # recording cadence only; it must not touch the walk
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, trajectory_interval=args.interval))

    print(f"replaying molecule {args.molecule} of {args.run_dir.parent.name} "
          f"(sweep {args.sweep_index}, chunk {args.chunk})")
    r = simulate(cfg, args.sweep_index, n_molecules=1, first_id=args.molecule,
                 layout_seed=chunk_meta.get("layout_seed"), n_trajectories=1)
    got = r.molecules.iloc[0]

    def same(a, b):
        """Exact, not approximate: a replay either is the same walk or it is not."""
        if pd.isna(a) or pd.isna(b):
            return bool(pd.isna(a)) and bool(pd.isna(b))
        return str(a) == str(b) if isinstance(a, str) or isinstance(b, str) else a == b

    bad = []
    for k in CHECK:
        if k not in stored.columns:
            continue
        ok = same(row[k], got[k])
        bad += [] if ok else [k]
        print(f"  {k:16} stored {str(row[k]):<22} replay {str(got[k]):<22}{'' if ok else '  <-- MISMATCH'}")
    if bad:
        raise SystemExit(f"replay does not reproduce the stored molecule: {bad}. Refusing to "
                         f"write a trajectory that does not match the run it claims to come from.")

    traj = r.trajectories[0]
    # The buffer is sized for t_max and NaN-filled; keep only the frames actually written.
    used = np.flatnonzero(~np.isnan(traj[:, 0]))
    traj = traj[: used[-1] + 1] if used.size else traj[:0]
    out = args.out or PROJECT_ROOT / "results" / "trajectories" / f"{args.run_dir.parent.name}_c{args.chunk}_m{args.molecule}"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Not with_suffix: run names contain dots ("ref0.5"), which it would treat as an extension.
    npy, meta_path = out.parent / f"{out.name}.npy", out.parent / f"{out.name}.json"
    np.save(npy, traj)

    scene = dict(
        run=args.run_dir.parent.name, sweep_index=args.sweep_index, chunk=args.chunk,
        molecule=int(args.molecule), interval=cfg.run.trajectory_interval,
        n_frames=int(len(traj)), molecule_row={k: (None if pd.isna(v) else v) for k, v in
                                               got.items() if not isinstance(v, np.ndarray)},
        receptor_cells=r.meta["receptor_cells"], layout_seed=chunk_meta.get("layout_seed"),
        geometry=dict(dim=cfg.geometry.dim, cell_radius=cfg.geometry.cell_radius, gap=cfg.geometry.gap,
                      spacing=cfg.geometry.spacing, target_ring=cfg.geometry.target_ring,
                      rings_beyond_target=cfg.geometry.rings_beyond_target),
        capture_radius=cfg.receptors.capture_radius, n_per_cell=cfg.receptors.n_per_cell,
    )
    meta_path.write_text(json.dumps(scene, indent=2, default=float))
    print(f"\n{len(traj)} frames at {cfg.run.trajectory_interval} s -> {npy}")


if __name__ == "__main__":
    main()
