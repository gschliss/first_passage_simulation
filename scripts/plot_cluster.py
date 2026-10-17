"""Draw the check figure for a config's cluster, receptors and escape line.

Usage:
    python scripts/plot_cluster.py configs/lattice2d.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fpsim import load_config, make_cluster, make_receptors
from fpsim.viz.cluster import plot_cluster_check

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.geometry.dim != 2:
        raise SystemExit("check figure is 2D only for now")
    cluster = make_cluster(cfg.geometry)
    receptors = make_receptors(cluster, cfg.geometry.target_ring, cfg.receptors.n_per_cell,
                               cfg.receptors.capture_radius, np.random.default_rng(cfg.run.seed))
    out = args.out or PROJECT_ROOT / "results" / "figures" / f"{cfg.name}_cluster_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_cluster_check(cfg, cluster, receptors, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
