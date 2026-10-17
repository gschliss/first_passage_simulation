"""Constants and helpers shared by the 2D and 3D kernels."""

from __future__ import annotations

import numpy as np

# Molecule state (docs/architecture.md §3)
BOUND = 1
FREE = 0

# Outcome codes. Step 2 has no receptors or escape, so every molecule times out.
TIMEOUT = 0
CAPTURED = 1
ESCAPED = 2
OUTCOME_NAMES = {TIMEOUT: "timeout", CAPTURED: "captured", ESCAPED: "escaped"}

MODE_ADAPTIVE = 0
MODE_FIXED = 1


def molecule_seeds(run_seed: int, sweep_index: int, n: int, first_id: int = 0) -> np.ndarray:
    """One RNG seed per molecule, so results don't depend on threads or job splitting (§5)."""
    ids = np.arange(first_id, first_id + n, dtype=np.int64)
    return (np.int64(run_seed) * 1_000_003 + np.int64(sweep_index) * 7_919_000_001 + ids * 2_654_435_761) % (2**32)
