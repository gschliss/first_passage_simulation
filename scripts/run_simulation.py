"""Run a dimension sweep from a YAML config.

Usage:
    python scripts/run_simulation.py configs/baseline_sphere.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from fpsim.analysis import fpt_summary, mfpt_spherical_shell
from fpsim.geometry import GEOMETRIES
from fpsim.io import load_config, make_run_dir, save_result
from fpsim.simulation import run_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    geo_cfg = dict(cfg["geometry"])
    geo_cls = GEOMETRIES[geo_cfg.pop("type")]
    sim_cfg = cfg["simulation"]

    run_dir = make_run_dir(args.results, cfg["name"])
    shutil.copy(args.config, run_dir / "config.yaml")

    summaries = {}
    for dim in cfg["dims"]:
        geometry = geo_cls(dim=dim, **geo_cfg)
        result = run_simulation(geometry, **sim_cfg)
        save_result(result, run_dir, f"dim{dim}")

        summary = fpt_summary(result.fpt)
        if geo_cls.__name__ == "SphericalShell":
            summary["theory"] = mfpt_spherical_shell(dim, sim_cfg["D"], **geo_cfg)
        summaries[dim] = summary
        print(f"d={dim}: " + ", ".join(f"{k}={v:.4g}" for k, v in summary.items()))

    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
