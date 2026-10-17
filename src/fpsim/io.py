"""Save and load simulation results: one Parquet table per job plus JSON metadata (L35)."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fpsim.simulate import Result

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def save_result(result: Result, stem) -> Path:
    """Write ``<stem>.parquet`` (molecules), ``<stem>.json`` (metadata) and, if any,
    ``<stem>_traj.npy`` (trajectories). Returns the Parquet path."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    result.molecules.to_parquet(stem.with_suffix(".parquet"), index=False)
    meta = dict(result.meta, git_commit=git_commit(), host=platform.node())
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    if result.trajectories.size:
        np.save(stem.parent / f"{stem.name}_traj.npy", result.trajectories)
    return stem.with_suffix(".parquet")


def load_result(stem) -> Result:
    stem = Path(stem)
    molecules = pd.read_parquet(stem.with_suffix(".parquet"))
    meta = json.loads(stem.with_suffix(".json").read_text())
    traj_path = stem.parent / f"{stem.name}_traj.npy"
    traj = np.load(traj_path) if traj_path.exists() else np.empty((0, 0, 4))
    return Result(molecules=molecules, trajectories=traj, meta=meta)
