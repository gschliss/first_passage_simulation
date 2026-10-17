"""Statistics on first-passage times and analytical benchmarks."""

from fpsim.analysis.statistics import fpt_summary, survival_curve
from fpsim.analysis.theory import mfpt_spherical_shell

__all__ = ["fpt_summary", "survival_curve", "mfpt_spherical_shell"]
