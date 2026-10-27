"""Combine paired dimension-experiment runs (L57-L59) into one figure across bound dwells.

Each run directory must come from scripts/dimension_experiment.py with --layouts > 1 and
the same layouts, free fraction and geometry, so the points are comparable.

Usage:
    python scripts/plot_paired_ratios.py results/dimension/lattice2d_ring2_rec2_dw5-16-50_paired10 \\
        results/dimension/lattice2d_ring2_rec2_dw100-300_paired10 --out results/figures/paired_ratios_lattice2d.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim.jobs import merge

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e6e6e6"
REF = "free-only, same D_eff"
D_BOUND = 0.5  # µm²/s, for the slide-length axis (L20)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load_curves(dirs, size_nm=2):
    """Per bound dwell: (switching molecules, reference molecules), from each run's capture runs."""
    out = {}
    for d in dirs:
        m = merge(d / f"runs_a{size_nm}nm")[0]
        dwells = sorted(m[m.sweep == "switch"].bound_dwell.unique())
        for k, dwell in enumerate(dwells):
            out[dwell] = (m[m.sweep_index == k], m[m.sweep_index == len(dwells) + k])
    return dict(sorted(out.items()))


def load(dirs):
    paired, summary, cal, check = [], [], [], []
    for d in dirs:
        paired.append(pd.read_csv(d / "paired_ratios.csv"))
        summary.append(pd.read_csv(d / "summary.csv"))
        c = pd.read_csv(d / "calibration.csv")
        cal.append(c[c.role == "switch"])
        check.append(pd.read_csv(d / "calibration_reference.csv"))
    sort = lambda df: df.sort_values("bound_dwell").reset_index(drop=True)  # noqa: E731
    return sort(pd.concat(paired)), pd.concat(summary), sort(pd.concat(cal)), sort(pd.concat(check))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--t-min", type=float, default=100.0, help="start of the capture-curve time axis (s)")
    args = p.parse_args()

    import json
    ref_info = json.loads((args.run_dirs[0] / "reference.json").read_text())
    global REF
    REF = ref_info.get("label", "free-only, same D_eff")
    matched = ref_info.get("matched", True)
    eq = "at equal D_eff" if matched else f"({REF.split(', ')[1]}, not matched; times are for captured molecules)"
    paired, s, cal, check = load(args.run_dirs)
    curves = load_curves(args.run_dirs)
    dw = paired.bound_dwell.to_numpy() * 1e3  # ms
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})
    n_c = len(curves)
    fig = plt.figure(figsize=(max(14, 3.6 * n_c), 15))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 2], hspace=0.28, top=0.95)
    top_gs = outer[0].subgridspec(1, n_c, wspace=0.12)
    t_max = max(max(a.t_end.max(), b.t_end.max()) for a, b in curves.values())
    t = np.logspace(np.log10(args.t_min), np.log10(t_max), 400)
    for k, (dwell, (msw, mref)) in enumerate(curves.items()):
        ax = fig.add_subplot(top_gs[0, k])
        for m, color, label in ((msw, BLUE, "switching (5% free)"), (mref, ORANGE, REF)):
            times = np.sort(m.t_end[m.outcome == "captured"].to_numpy())
            f = np.searchsorted(times, t, side="right") / len(m)
            half = 1.96 * np.sqrt(f * (1 - f) / len(m))
            ax.fill_between(t, f - half, f + half, color=color, alpha=0.15, lw=0)
            ax.plot(t, f, color=color, lw=2, label=f"{label} (n = {len(m)})")
        ax.set_xscale("log"); ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1.02)
        ax.set_xlabel("time since release (s)")
        if k == 0:
            ax.set_ylabel("fraction captured")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"bound {dwell * 1e3:g} ms (bands: 95% CI)", loc="left", fontsize=10)
        style(ax)
    bottom = outer[1].subgridspec(2, 2, hspace=0.35, wspace=0.18)
    axes = np.array([[fig.add_subplot(bottom[i, j]) for j in range(2)] for i in range(2)])

    ax = axes[0, 0]
    for off, col, mk, color, label in ((0.97, "total", "o", BLUE, "total capture time"),
                                       (1.03, "local", "D", INK, "local search (at a target cell)")):
        ax.errorbar(dw * off, paired[col], yerr=[paired[col] - paired[f"{col}_lo"], paired[f"{col}_hi"] - paired[col]],
                    fmt=mk + "-", color=color, ms=8, capsize=3, lw=1.5, label=label)
    ax.axhline(1, color=MUTED, lw=1)
    ax.annotate("1 = no difference", (dw.max(), 1), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(dw, [f"{v:g}" for v in dw]); ax.minorticks_off()
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel(("switching / free-only at equal D_eff" if matched else "switching / free-only") + " (mean time)")
    ax.set_title(f"A. Paired over {int(paired.n_layouts.iloc[0])} shared receptor layouts (95% CI)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    top = ax.secondary_xaxis("top", functions=(lambda ms: np.sqrt(2 * D_BOUND * np.maximum(ms, 1e-9) / 1e3),
                                               lambda lam: lam**2 / (2 * D_BOUND) * 1e3))
    top.set_xlabel("typical slide per binding, √(2·D_bound·τ_bound) (µm)", color=MUTED)
    top.set_xticks(np.sqrt(2 * D_BOUND * dw / 1e3), [f"{v:.2f}" for v in np.sqrt(2 * D_BOUND * dw / 1e3)])
    top.xaxis.set_minor_locator(plt.NullLocator())
    top.tick_params(colors=MUTED)
    style(ax)

    def by_role(ax, col, se, ylabel, title):
        for off, role, color, mk, label in ((0.97, "switch", BLUE, "o", "switching (5% free)"),
                                            (1.03, "reference", ORANGE, "s", REF)):
            r = s[s.role == role].sort_values("bound_dwell")
            ax.errorbar(r.bound_dwell * 1e3 * off, r[col], yerr=1.96 * r[se], fmt=mk + "-", color=color, ms=8,
                        capsize=3, lw=1.5, label=label)
        ax.set_xscale("log"); ax.set_xticks(dw, [f"{v:g}" for v in dw]); ax.minorticks_off()
        ax.set_xlabel("mean bound dwell of the switching walker (ms)")
        ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left"); ax.legend(frameon=False, fontsize=9, loc="lower left")
        style(ax)

    by_role(axes[0, 1], "local_mean", "local_se", "mean time from first arrival to capture (s, 95% CI)",
            "B. Local search: finding the receptor at a target cell")
    # C. Where the time goes: stacked bars, transport (light) + local search (dark); bar height = total
    ax = axes[1, 0]
    from matplotlib.patches import Patch
    xb = np.arange(len(dw))
    for off, role, color in ((-0.19, "switch", BLUE), (0.19, "reference", ORANGE)):
        r = s[s.role == role].sort_values("bound_dwell")
        ax.bar(xb + off, r.transport_mean, 0.36, color=color, alpha=0.4, edgecolor="white", lw=2)
        ax.bar(xb + off, r.local_mean, 0.36, bottom=r.transport_mean, color=color, edgecolor="white", lw=2)
        ax.errorbar(xb + off, r.t_mean, yerr=1.96 * r.t_se, fmt="none", ecolor=INK, capsize=3, lw=1)
    handles = [Patch(color=MUTED, alpha=0.4, label="light: reaching a target cell"),
               Patch(color=MUTED, label="dark: then finding the receptor"),
               Patch(color=BLUE, label="switching (5% free)"), Patch(color=ORANGE, label=REF)]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_xticks(xb, [f"{v:g}" for v in dw])
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("mean time (s)")
    top_y = (s.t_mean + 1.96 * s.t_se).max()
    ax.set_ylim(0, top_y * 1.35)
    ax.set_title("C. Where the time goes (left bar: switching, right: free-only)\nbar height = total capture time; error bar = its 95% CI", loc="left")
    style(ax)

    ax = axes[1, 1]
    ax.errorbar(cal.bound_dwell * 1e3 * 0.97, cal.d_eff, yerr=1.96 * cal.d_eff_se, fmt="o", color=BLUE, ms=8, capsize=3,
                label="switching (5% free)")
    ax.errorbar(check.bound_dwell * 1e3 * 1.03, check.d_eff, yerr=1.96 * check.d_eff_se, fmt="s", color=ORANGE, ms=8,
                capsize=3, label=REF)
    ax.set_xscale("log"); ax.set_xticks(dw, [f"{v:g}" for v in dw]); ax.minorticks_off()
    ax.set_xlabel("mean bound dwell of the switching walker (ms)")
    ax.set_ylabel("measured D_eff (µm²/s, 95% CI)")
    if matched:
        ax.set_ylim(0, 0.8)
    else:
        ax.set_yscale("log"); ax.set_ylim(0.3, 2 * check.d_eff.max())
    ax.set_title("D. Long-range spreading is matched" if matched else "D. Long-range spreading (free-only: φ·D, predicted)",
                 loc="left"); ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)

    run_cfg = json.loads((args.run_dirs[0] / "runs_a2nm" / "config.json").read_text())["config"]
    f = next(sw["free_fraction"] for sw in run_cfg["sweeps"] if sw["name"] == "switch")
    n_mol = int(s[s.role == "switch"].n.iloc[0])
    n_rec = run_cfg["receptors"]["n_per_cell"]
    plural = "" if n_rec == 1 else "s"
    fig.suptitle(f"Switching vs non-switching diffusion {eq}\n({run_cfg['geometry']['dim']}D, {f * 100:g}% free, "
                 f"ring-{run_cfg['geometry']['target_ring']} targets with {n_rec} receptor{plural} per cell, "
                 f"D_free {run_cfg['diffusion']['d_free']:g}, D_bound {run_cfg['diffusion']['d_bound']:g} µm²/s, "
                 f"{run_cfg['receptors']['capture_radius'] * 1e3:g} nm; {n_mol} molecules per condition)", fontsize=12, y=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(paired.to_string(index=False))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
