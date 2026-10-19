"""Is switching between dimensions different from plain diffusion at the same D_eff? (L47-L51)

Steps (each saved under results/dimension/<name>/; finished steps are reused on rerun):
  1. Calibrate D_eff at each switching point (5% free; bound dwell 5, 71, 285, 1000 ms)
     and the gap-geometry factor phi = D_eff / D for a free-only walker (L50).
  2. Reference walkers: free-only, D_free = D_eff(switching point) / phi, so their long-range
     spreading matches; verify by calibrating them too (L47).
  3. Capture runs for switching and reference points at 2 nm and 10 nm targets (L48),
     with arrival-time outputs (L49).
  4. Figure + summary table.

Run with 6 threads to leave cores free:
    NUMBA_NUM_THREADS=6 python scripts/dimension_experiment.py --n 100
    python scripts/dimension_experiment.py --replot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim import load_config
from fpsim.config import Receptors, Sweep
from fpsim.deff import measure_deff
from fpsim.jobs import Task, merge, run_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DWELLS = (0.005, 0.0707, 0.285, 1.0)  # s, bound dwell at 5% free (same points as the 'dwell' sweep)
FREE_FRACTION = 0.05
SIZES = (0.002, 0.010)  # µm capture radius
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e6e6e6"


def base_config(path):
    cfg = load_config(path)
    return dataclasses.replace(cfg, sweeps=(Sweep("switch", free_fraction=FREE_FRACTION, bound_dwell=DWELLS),
                                            Sweep("free_unit", free_fraction=1.0, d_free=1.0)))


def calibrate(cfg, out, n, t_max, short_lag, long_lag):
    path = out / "calibration.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for i in range(cfg.n_sweep_points):
        r = measure_deff(cfg, i, n, t_max, short_lag, long_lag)
        rows.append(dict(dataclasses.asdict(r), role="switch" if i < len(DWELLS) else "free_unit",
                         bound_dwell=DWELLS[i] if i < len(DWELLS) else np.nan))
        print(f"  D_eff {r.label}: {r.d_eff:.4f} ± {r.d_eff_se:.4f} (short lag {r.d_eff_short:.4f}), escaped {r.escaped}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def experiment_config(cfg, d_ref, capture_radius):
    return dataclasses.replace(
        cfg, name=f"{cfg.name}_dim_a{capture_radius * 1e3:.0f}nm",
        receptors=Receptors(n_per_cell=cfg.receptors.n_per_cell, capture_radius=capture_radius),
        sweeps=(Sweep("switch", free_fraction=FREE_FRACTION, bound_dwell=DWELLS),
                Sweep("reference", free_fraction=1.0, d_free=tuple(float(d) for d in d_ref))))


def verify_reference(cfg, d_ref, out, n, t_max, short_lag, long_lag):
    path = out / "calibration_reference.csv"
    if path.exists():
        return pd.read_csv(path)
    ref = experiment_config(cfg, d_ref, SIZES[0])
    rows = []
    for k in range(len(DWELLS)):
        r = measure_deff(ref, len(DWELLS) + k, n, t_max, short_lag, long_lag)
        rows.append(dict(dataclasses.asdict(r), bound_dwell=DWELLS[k]))
        print(f"  check {r.label}: D_eff {r.d_eff:.4f} ± {r.d_eff_se:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def capture_runs(cfg, d_ref, out, n):
    dirs = {}
    for a in SIZES:
        ecfg = experiment_config(cfg, d_ref, a)
        run_dir = out / f"runs_a{a * 1e3:.0f}nm"
        for i in range(ecfg.n_sweep_points):
            path = run_task(ecfg, Task(i, 0), n, run_dir)
            print(f"  a = {a * 1e3:.0f} nm, point {i}: {'done' if path else 'exists'}", flush=True)
        dirs[a] = run_dir
    return dirs


def summarize(out):
    rows = []
    for a in SIZES:
        molecules, _, problems = merge(out / f"runs_a{a * 1e3:.0f}nm")
        if problems:
            print(f"merge problems (a = {a * 1e3:.0f} nm): {problems}")
        for (sweep, i), m in molecules.groupby(["sweep", "sweep_index"]):
            k = i if sweep == "switch" else i - len(DWELLS)
            cap = m[m.outcome == "captured"]
            t, arr = cap.t_end, cap.t_first_target
            rows.append(dict(size_nm=a * 1e3, role=sweep, bound_dwell=DWELLS[k], n=len(m),
                             captured=len(cap) / len(m), escaped=(m.outcome == "escaped").mean(),
                             timeout=(m.outcome == "timeout").mean(),
                             t_mean=t.mean(), t_se=t.std(ddof=1) / np.sqrt(len(t)), t_median=t.median(),
                             transport_mean=arr.mean(), local_mean=(t - arr).mean(),
                             local_se=(t - arr).std(ddof=1) / np.sqrt(len(t)),
                             from_bound=(cap.final_state == "bound").mean(),
                             visits_mean=cap.n_target_visits.mean()))
    return pd.DataFrame(rows), {a: merge(out / f"runs_a{a * 1e3:.0f}nm")[0] for a in SIZES}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ratio(a, b, sa, sb):
    r = a / b
    return r, r * np.hypot(sa / a, sb / b)


def plot(out, cal, check, s, molecules, title, size=SIZES[0], t_min=100.0):
    """Switching vs free-only reference at one target size (L52 replot: size comparison dropped)."""
    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK})
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    sw = cal[cal.role == "switch"].reset_index(drop=True)
    m = molecules[size]
    d = s[s.size_nm == round(size * 1e3)]
    roles = (("switch", BLUE, "o", "switching (5% free)"), ("reference", ORANGE, "s", "free-only, same D_eff"))

    # Top: cumulative capture per switching point, from t_min on
    t = np.logspace(np.log10(t_min), np.log10(m.t_end.max()), 400)
    for k, dwell in enumerate(DWELLS):
        ax = fig.add_subplot(gs[0, k])
        for role, color, _, label in roles:
            sel = m[m.sweep_index == (k if role == "switch" else len(DWELLS) + k)]
            times = np.sort(sel.t_end[sel.outcome == "captured"].to_numpy())
            f = np.searchsorted(times, t, side="right") / len(sel)
            half = 1.96 * np.sqrt(f * (1 - f) / len(sel))
            ax.fill_between(t, f - half, f + half, color=color, alpha=0.15, lw=0)
            ax.plot(t, f, color=color, lw=2, label=f"{label} (n = {len(sel)})")
        ax.set_xscale("log")
        ax.set_xlim(t_min, t[-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("time since release (s)")
        if k == 0:
            ax.set_ylabel("fraction captured")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"bound {dwell * 1e3:.0f} ms, free {dwell * FREE_FRACTION / (1 - FREE_FRACTION) * 1e3:.2g} ms\n"
                     f"D_eff = {sw.d_eff[k]:.2f} µm²/s (bands: 95% CI)", loc="left", fontsize=10)
        style(ax)

    x = np.arange(len(DWELLS))
    xt = [f"{v * 1e3:.0f}" for v in DWELLS]

    def paired(ax, ycol, secol, ylabel, panel_title):
        for off, (role, color, mk, label) in zip((-0.09, 0.09), roles):
            r = d[d.role == role].sort_values("bound_dwell")
            ax.errorbar(x + off, r[ycol], yerr=1.96 * r[secol], fmt=mk, color=color, ms=8, capsize=3, lw=1.5, label=label)
        ax.set_xticks(x, xt)
        ax.set_xlabel("bound dwell of the switching walker (ms)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_title(panel_title, loc="left", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        style(ax)

    # Bottom 1: D_eff matching
    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(x - 0.09, sw.d_eff, yerr=1.96 * sw.d_eff_se, fmt="o", color=BLUE, ms=8, capsize=3, label=roles[0][3])
    ax.errorbar(x + 0.09, check.d_eff, yerr=1.96 * check.d_eff_se, fmt="s", color=ORANGE, ms=8, capsize=3, label=roles[1][3])
    ax.set_xticks(x, xt); ax.set_xlabel("bound dwell of the switching walker (ms)")
    ax.set_ylabel("measured D_eff (µm²/s, 95% CI)"); ax.set_ylim(0, 0.8)
    ax.set_title("A. Long-range spreading (D_eff) is matched", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax)

    # Bottom 2 and 3: total and local-search time
    paired(fig.add_subplot(gs[1, 1]), "t_mean", "t_se", "mean capture time (s, 95% CI)",
           f"B. Total capture time ({size * 1e3:.0f} nm target)")
    paired(fig.add_subplot(gs[1, 2]), "local_mean", "local_se", "mean time from first arrival to capture (s, 95% CI)",
           "C. Local search: at a target cell, finding the receptor")

    # Bottom 4: where the time goes
    ax = fig.add_subplot(gs[1, 3])
    for off, (role, color, _, label) in zip((-0.18, 0.18), roles):
        r = d[d.role == role].sort_values("bound_dwell")
        ax.bar(x + off, r.transport_mean, 0.34, color=color, alpha=0.4, edgecolor="white", lw=2)
        ax.bar(x + off, r.local_mean, 0.34, bottom=r.transport_mean, color=color, edgecolor="white", lw=2)
    from matplotlib.patches import Patch
    handles = [Patch(color=MUTED, alpha=0.4, label="light: reaching a target cell"),
               Patch(color=MUTED, label="dark: then finding the receptor")]
    handles += [Patch(color=color, label=label) for _, color, _, label in roles]
    ax.set_xticks(x, xt); ax.set_xlabel("bound dwell of the switching walker (ms)")
    ax.set_ylabel("mean time (s)")
    ax.set_title("D. Where the time goes (left bar: switching, right: free-only)", loc="left", fontsize=10)
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, (d.transport_mean + d.local_mean).max() * 1.3)
    style(ax)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "dimension_experiment.png", dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "lattice2d.yaml")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--cal-t-max", type=float, default=3000.0)
    p.add_argument("--short-lag", type=float, default=50.0)
    p.add_argument("--long-lag", type=float, default=500.0)
    p.add_argument("--replot", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="output directory (default results/dimension/<config name>)")
    args = p.parse_args()

    cfg = base_config(args.config)
    out = args.out or PROJECT_ROOT / "results" / "dimension" / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    print("1. D_eff calibration", flush=True)
    cal = calibrate(cfg, out, args.n, args.cal_t_max, args.short_lag, args.long_lag)
    phi = float(cal[cal.role == "free_unit"].d_eff.iloc[0])  # free-only D_eff per unit D
    d_eff = cal[cal.role == "switch"].d_eff.to_numpy()
    d_ref = d_eff / phi
    (out / "reference.json").write_text(json.dumps(dict(phi=phi, d_eff=d_eff.tolist(), d_free_reference=d_ref.tolist()), indent=2))
    print(f"   phi = {phi:.4f}; reference D_free = {np.round(d_ref, 4).tolist()}", flush=True)

    print("2. verify reference D_eff", flush=True)
    check = verify_reference(cfg, d_ref, out, args.n, args.cal_t_max, args.short_lag, args.long_lag)

    if not args.replot:
        print("3. capture runs", flush=True)
        capture_runs(cfg, d_ref, out, args.n)

    s, molecules = summarize(out)
    s.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print(s.to_string(index=False))
    plot(out, cal, check, s, molecules,
         f"Switching vs non-switching diffusion at matched D_eff ({cfg.geometry.dim}D, {args.n} molecules per run, "
         f"2 nm target; free-only D_free = D_eff / φ, φ = {phi:.3f})")
    print(f"Saved {out / 'dimension_experiment.png'}")


if __name__ == "__main__":
    main()
