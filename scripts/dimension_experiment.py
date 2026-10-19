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


def plot(out, cal, check, s, molecules, title):
    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK})
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    sw = cal[cal.role == "switch"].reset_index(drop=True)

    # Top: cumulative capture per switching point
    for k, dwell in enumerate(DWELLS):
        ax = fig.add_subplot(gs[0, k])
        t = np.logspace(-1, np.log10(max(m.t_end.max() for m in molecules.values())), 400)
        for a, ls in zip(SIZES, ("-", "--")):
            m = molecules[a]
            for role, color, idx in (("switch", BLUE, k), ("reference", ORANGE, len(DWELLS) + k)):
                sel = m[m.sweep_index == idx]
                times = np.sort(sel.t_end[sel.outcome == "captured"].to_numpy())
                f = np.searchsorted(times, t, side="right") / len(sel)
                ax.plot(t, f, ls, color=color, lw=2,
                        label=f"{'switching' if role == 'switch' else 'free-only reference'}, {a * 1e3:.0f} nm")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("time since release (s)")
        if k == 0:
            ax.set_ylabel("fraction captured")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"bound {dwell * 1e3:.0f} ms (5% free)\nD_eff = {sw.d_eff[k]:.2f} µm²/s", loc="left", fontsize=10)
        style(ax)

    x = np.arange(len(DWELLS))
    xt = [f"{d * 1e3:.0f}" for d in DWELLS]

    # Bottom 1: D_eff matching
    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(x - 0.08, sw.d_eff, yerr=1.96 * sw.d_eff_se, fmt="o", color=BLUE, ms=8, capsize=3, label="switching")
    ax.errorbar(x + 0.08, check.d_eff, yerr=1.96 * check.d_eff_se, fmt="s", color=ORANGE, ms=8, capsize=3, label="free-only reference")
    naive = FREE_FRACTION * 10.0 + (1 - FREE_FRACTION) * 0.5
    ax.axhline(naive, color=MUTED, ls=":", lw=1)
    ax.annotate(f"naive f·D_free + (1−f)·D_bound = {naive:.2f}", (x[-1], naive), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x, xt); ax.set_xlabel("bound dwell (ms)"); ax.set_ylabel("measured D_eff (µm²/s, 95% CI)")
    ax.set_ylim(0, max(naive, sw.d_eff.max()) * 1.25)
    ax.set_title("A. Long-range spreading is matched", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax)

    # Bottom 2: T_switch / T_reference
    ax = fig.add_subplot(gs[1, 1])
    for off, a, mk, ls in zip((-0.08, 0.08), SIZES, ("o", "s"), ("-", "--")):
        sa, sr = s[(s.size_nm == a * 1e3) & (s.role == "switch")], s[(s.size_nm == a * 1e3) & (s.role == "reference")]
        r, e = ratio(sa.t_mean.to_numpy(), sr.t_mean.to_numpy(), sa.t_se.to_numpy(), sr.t_se.to_numpy())
        ax.errorbar(x + off, r, yerr=1.96 * e, fmt=mk + ls, color=BLUE, ms=8, capsize=3, lw=1.5, label=f"{a * 1e3:.0f} nm target")
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xticks(x, xt); ax.set_xlabel("bound dwell (ms)"); ax.set_ylabel("mean capture time: switching / reference")
    ax.set_title("B. Switching vs plain diffusion at equal D_eff", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    # Bottom 3: target-size sensitivity
    ax = fig.add_subplot(gs[1, 2])
    for off, role, color, mk in ((-0.08, "switch", BLUE, "o"), (0.08, "reference", ORANGE, "s")):
        a2 = s[(s.size_nm == 2) & (s.role == role)]
        a10 = s[(s.size_nm == 10) & (s.role == role)]
        r, e = ratio(a2.t_mean.to_numpy(), a10.t_mean.to_numpy(), a2.t_se.to_numpy(), a10.t_se.to_numpy())
        ax.errorbar(x + off, r, yerr=1.96 * e, fmt=mk + "-", color=color, ms=8, capsize=3, lw=1.5,
                    label="switching" if role == "switch" else "free-only reference")
    ax.axhline(1, color=MUTED, lw=1)
    ax.annotate("1 = insensitive to target size (1D-like)", (x[-1], 1), xytext=(0, -12), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x, xt); ax.set_xlabel("bound dwell (ms)"); ax.set_ylabel("capture time ratio, 2 nm / 10 nm target")
    ax.set_title("C. Sensitivity to target size (dimension signature)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    # Bottom 4: decomposition at 2 nm
    ax = fig.add_subplot(gs[1, 3])
    for off, role, color in ((-0.18, "switch", BLUE), (0.18, "reference", ORANGE)):
        d = s[(s.size_nm == 2) & (s.role == role)]
        ax.bar(x + off, d.transport_mean, 0.34, color=color, alpha=0.45, edgecolor="white", lw=2,
               label=f"{'switching' if role == 'switch' else 'reference'}: reach a target cell")
        ax.bar(x + off, d.local_mean, 0.34, bottom=d.transport_mean, color=color, edgecolor="white", lw=2,
               label=f"{'switching' if role == 'switch' else 'reference'}: then find the receptor")
    ax.set_xticks(x, xt); ax.set_xlabel("bound dwell (ms)"); ax.set_ylabel("mean time (s), 2 nm target")
    ax.set_title("D. Where the time goes: transport vs local search", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
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
         f"Dimension switching vs plain diffusion at matched D_eff ({cfg.geometry.dim}D, {args.n} molecules per run, "
         f"5% free; reference D_free = D_eff / φ, φ = {phi:.3f})")
    print(f"Saved {out / 'dimension_experiment.png'}")


if __name__ == "__main__":
    main()
