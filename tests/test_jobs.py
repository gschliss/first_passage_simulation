"""Job splitting, resuming and merging (L33)."""

import json

import numpy as np
import pytest

from fpsim.config import from_dict
from fpsim.jobs import Task, merge, n_tasks, run_task, task_from_array_id

# Scaled-down geometry so runs take a fraction of a second (test-only).
SMALL = from_dict({
    "name": "small_jobs",
    "geometry": {"cell_radius": 2.0, "gap": 0.5, "target_ring": 1, "rings_beyond_target": 1},
    "receptors": {"n_per_cell": 1},
    "run": {"t_max": 5.0},
    "sweeps": {"a": {"free_fraction": 0.05, "bound_dwell": [0.01, 1.0]}},
})


def test_array_ids_cover_every_chunk_once():
    tasks = [task_from_array_id(t, 3, SMALL) for t in range(n_tasks(SMALL, 3))]
    assert len(set(tasks)) == len(tasks) == 2 * 3
    assert tasks[0] == Task(0, 0) and tasks[-1] == Task(1, 2)
    with pytest.raises(ValueError):
        task_from_array_id(6, 3, SMALL)


@pytest.fixture()
def run_dir(tmp_path):
    for i in range(2):
        for c in range(2):
            run_task(SMALL, Task(i, c), 10, tmp_path)
    return tmp_path


def test_merge_combines_chunks(run_dir):
    molecules, summary, problems = merge(run_dir, expected_chunks=2)
    assert problems == []
    assert len(molecules) == 40 and set(summary.n) == {20}
    for _, m in molecules.groupby("sweep_index"):
        assert sorted(m.molecule_id) == list(range(20))  # chunks don't overlap
    assert molecules.seed.nunique() == 40


def test_chunks_get_different_receptor_layouts(run_dir):
    a = json.loads((run_dir / "sweep00_chunk0000.json").read_text())["receptor_angles"]
    b = json.loads((run_dir / "sweep00_chunk0001.json").read_text())["receptor_angles"]
    assert not np.allclose(a, b)  # one layout per job (L36)


def test_existing_chunks_are_skipped(run_dir):
    assert run_task(SMALL, Task(0, 0), 10, run_dir) is None
    assert run_task(SMALL, Task(0, 0), 10, run_dir, overwrite=True) is not None


def test_different_config_refused(run_dir):
    other = from_dict({**{"name": "small_jobs"}, "geometry": {"cell_radius": 2.0, "gap": 0.5, "target_ring": 1,
                       "rings_beyond_target": 1}, "receptors": {"n_per_cell": 2}, "run": {"t_max": 5.0},
                       "sweeps": {"a": {"free_fraction": 0.05, "bound_dwell": [0.01, 1.0]}}})
    with pytest.raises(ValueError, match="different config"):
        run_task(other, Task(0, 0), 10, run_dir)


def test_merge_reports_missing_and_foreign_chunks(run_dir):
    (run_dir / "sweep01_chunk0001.parquet").unlink()
    (run_dir / "sweep01_chunk0001.json").unlink()
    meta = json.loads((run_dir / "sweep00_chunk0001.json").read_text())
    meta["config_hash"] = "deadbeef0000"
    (run_dir / "sweep00_chunk0001.json").write_text(json.dumps(meta))
    _, _, problems = merge(run_dir, expected_chunks=2)
    assert any("missing" in p for p in problems)
    assert any("config hash" in p for p in problems)
