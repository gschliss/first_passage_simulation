"""Split a sweep into independent jobs, run them, and merge their outputs (L33).

A run directory holds one result per (sweep point, chunk):
    <out_dir>/config.json                  resolved config, written once
    <out_dir>/sweepSS_chunkCCCC.parquet    one row per molecule
    <out_dir>/sweepSS_chunkCCCC.json       metadata (config hash, git commit, host, timing)

Chunk ``c`` of a point simulates molecule ids ``c*n .. c*n + n-1`` (so seeds never
collide across chunks) with receptor layout ``c`` (one layout per job, L36).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from fpsim.config import Config
from fpsim.io import save_result
from fpsim.simulate import simulate


@dataclass(frozen=True)
class Task:
    sweep_index: int
    chunk: int


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True, default=float).encode()).hexdigest()[:12]


def n_tasks(cfg: Config, chunks_per_point: int) -> int:
    return cfg.n_sweep_points * chunks_per_point


def task_from_array_id(task_id: int, chunks_per_point: int, cfg: Config) -> Task:
    """SLURM array id -> (sweep point, chunk); points vary slowest."""
    if not 0 <= task_id < n_tasks(cfg, chunks_per_point):
        raise ValueError(f"array task {task_id} out of range 0..{n_tasks(cfg, chunks_per_point) - 1}")
    return Task(sweep_index=task_id // chunks_per_point, chunk=task_id % chunks_per_point)


def stem_for(out_dir, task: Task) -> Path:
    return Path(out_dir) / f"sweep{task.sweep_index:02d}_chunk{task.chunk:04d}"


def write_once(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` unless it exists; atomic so concurrent jobs can't clash."""
    if path.exists():
        return
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    try:
        os.link(tmp, path)  # fails if another job got there first
    except FileExistsError:
        pass
    finally:
        os.unlink(tmp)


def run_task(cfg: Config, task: Task, n_molecules: int, out_dir, overwrite: bool = False, **overrides) -> Path | None:
    """Run one chunk and save it. Returns the Parquet path, or None if it already existed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(cfg)
    manifest = dict(config=cfg.to_dict(), config_hash=chash, n_sweep_points=cfg.n_sweep_points,
                    sweep_labels=[p.label for p in cfg.sweep_points(1.0)])
    write_once(out_dir / "config.json", json.dumps(manifest, indent=2, default=float))
    existing = json.loads((out_dir / "config.json").read_text())["config_hash"]
    if existing != chash:
        raise ValueError(f"{out_dir} holds results for a different config ({existing} != {chash})")

    stem = stem_for(out_dir, task)
    if stem.with_suffix(".parquet").exists() and not overwrite:
        return None
    r = simulate(cfg, task.sweep_index, n_molecules=n_molecules, first_id=task.chunk * n_molecules,
                 layout_index=task.chunk, **overrides)
    r.meta.update(config_hash=chash, chunk=task.chunk, n_per_chunk=n_molecules, threads=numba.get_num_threads())
    return save_result(r, stem)


def merge(out_dir, expected_chunks: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Combine all chunks in ``out_dir``.

    Returns (molecules, summary per sweep point, problems). Problems include chunks from
    a different config, duplicate molecule ids, and missing chunks (if ``expected_chunks``).
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "config.json").read_text())
    chash = manifest["config_hash"]
    frames, problems, metas = [], [], []
    for meta_path in sorted(out_dir.glob("sweep*_chunk*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("config_hash") != chash:
            problems.append(f"{meta_path.name}: config hash {meta.get('config_hash')} != {chash}")
            continue
        m = pd.read_parquet(meta_path.with_suffix(".parquet"))
        sp = meta["sweep_point"]
        m.insert(0, "sweep_index", meta["sweep_index"])
        m.insert(1, "chunk", meta["chunk"])
        m.insert(2, "sweep", sp["sweep"])
        for k in ("free_fraction", "bound_dwell", "free_dwell"):
            m[k] = sp[k]
        frames.append(m)
        metas.append(meta)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), problems + ["no results found"]
    molecules = pd.concat(frames, ignore_index=True)

    dup = molecules.duplicated(["sweep_index", "molecule_id"])
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate (sweep_index, molecule_id) rows")
    commits = sorted({m.get("git_commit") or "unknown" for m in metas})
    if len(commits) > 1:
        problems.append(f"chunks come from several code versions: {commits}")

    n_points = manifest["n_sweep_points"]
    if expected_chunks is not None:
        have = {(m["sweep_index"], m["chunk"]) for m in metas}
        missing = [(i, c) for i in range(n_points) for c in range(expected_chunks) if (i, c) not in have]
        if missing:
            problems.append(f"{len(missing)} missing chunks, e.g. {missing[:5]}")

    wall = pd.DataFrame([dict(sweep_index=m["sweep_index"], wall_time_s=m["wall_time_s"],
                              threads=m.get("threads", np.nan)) for m in metas])
    summary = summarize(molecules, wall)
    return molecules, summary, problems


def summarize(molecules: pd.DataFrame, wall: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for i, m in molecules.groupby("sweep_index"):
        n = len(m)
        cap = m[m.outcome == "captured"]
        t = cap.t_end
        row = dict(
            sweep_index=i, sweep=m.sweep.iloc[0], free_fraction=m.free_fraction.iloc[0],
            bound_dwell=m.bound_dwell.iloc[0], free_dwell=m.free_dwell.iloc[0], n=n,
            n_chunks=m.chunk.nunique(),
            captured=len(cap) / n, escaped=(m.outcome == "escaped").mean(), timeout=(m.outcome == "timeout").mean(),
            capture_t_mean=t.mean(), capture_t_se=t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else np.nan,
            capture_t_median=t.median(),
            captured_while_bound=(cap.final_state == "bound").mean() if len(cap) else np.nan,
            time_free=m.time_free.sum() / m.t_end.sum(),
        )
        if wall is not None and len(wall):
            row["wall_s_total"] = wall.loc[wall.sweep_index == i, "wall_time_s"].sum()
        rows.append(row)
    return pd.DataFrame(rows)
