"""Shared figure style for the report (L96).

One place for the palette so the heatmaps and the line plots agree. Two series
(switching, non-switching) use categorical slots 1 and 2; t_20 magnitude uses a
single-hue sequential blue ramp, light = fast, dark = slow. The pair was checked
for colour-vision separation (worst-pair CVD dE 24.7, normal-vision 33.6).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8a8982"
GRID = "#e6e6e6"

SWITCH = "#2a78d6"   # categorical slot 1
REFERENCE = "#eb6834"  # categorical slot 2
ROLE = {"switch": SWITCH, "reference": REFERENCE}
LABEL = {"switch": "switching", "reference": "non-switching (free-only, D = 0.5)"}

# Sequential blue, 100 -> 700. Light means fast (small t_20).
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
SEQ.set_bad("#f0efec")  # cells with no t_20


def use():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": SECONDARY, "axes.edgecolor": GRID,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.labelsize": 10, "axes.titlesize": 11, "font.size": 10,
        "axes.grid": False, "legend.frameon": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 2, "lines.markersize": 8,
    })


def style(ax, grid="y"):
    """Recessive axes: no box, a single faint grid direction."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def fmt_time(v):
    """Compact seconds: 83, 211, 2.9k, 31k."""
    if not np.isfinite(v):
        return "n/a"
    if v >= 9.5e4:
        return f"{v / 1e3:.0f}k"
    if v >= 950:
        return f"{v / 1e3:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def log_ticks(axis, lo, hi, want=7):
    """Readable labels on a log axis whose range is under a decade or two.

    Matplotlib labels only whole powers of ten, so a 80-1100 s range gets a single
    "10^3" and nothing can be read off it. Step through 1/2/5 per decade, thinning
    to 1/5 then to whole decades if that gives too many labels.
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    for subs in ((1, 2, 5), (1, 5), (1,)):
        ticks = [s * 10.0 ** e for e in range(-12, 13) for s in subs]
        ticks = [t for t in sorted(ticks) if lo <= t <= hi]
        if len(ticks) <= want:
            break
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_time(v)))


AXIS_LABEL = {
    "free_fraction": "fraction of time free",
    "bound_dwell": "mean bound dwell (s)",
    "target_ring": "distance from source (ring)",
    "n_per_cell": "receptors per target cell",
    "finds_to_capture": "receptor finds needed (capture efficiency = 1/N)",
}

AXIS_TICK = {
    "free_fraction": lambda v: f"{v * 100:g}%",
    "bound_dwell": lambda v: f"{v:g}",
    "target_ring": lambda v: f"{v:.0f}",
    "n_per_cell": lambda v: f"{v:,.0f}",
    "finds_to_capture": lambda v: f"{v:.0f}",
}
