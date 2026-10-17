"""Config loading and result saving."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from fpsim.simulation import SimulationResult


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_run_dir(results_root, name: str) -> Path:
    """Create ``results/<YYYYmmdd-HHMMSS>_<name>/`` and return it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(results_root) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_result(result: SimulationResult, run_dir, tag: str) -> Path:
    path = Path(run_dir) / f"{tag}.npz"
    np.savez_compressed(
        path, fpt=result.fpt, dt=result.dt, t_max=result.t_max, meta=json.dumps(result.meta)
    )
    return path
