"""Summaries of first-passage-time samples."""

from __future__ import annotations

import numpy as np


def fpt_summary(fpt: np.ndarray) -> dict:
    """Mean/median/SEM over absorbed agents, plus the absorbed fraction.

    Note: if some agents were not absorbed (NaN), the mean is biased low;
    check ``absorbed_fraction`` before trusting it.
    """
    hit = fpt[~np.isnan(fpt)]
    n = hit.size
    return {
        "n_total": int(fpt.size),
        "absorbed_fraction": n / fpt.size if fpt.size else float("nan"),
        "mean": float(hit.mean()) if n else float("nan"),
        "sem": float(hit.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "median": float(np.median(hit)) if n else float("nan"),
    }


def survival_curve(fpt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """S(t) = fraction of agents not yet absorbed at each time in ``t``."""
    fpt = np.where(np.isnan(fpt), np.inf, fpt)
    return (fpt[None, :] > np.asarray(t)[:, None]).mean(axis=1)
