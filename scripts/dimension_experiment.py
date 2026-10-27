"""Is switching between dimensions different from plain diffusion at the same D_eff? (L47-L51)

Steps (each saved under results/dimension/<name>/; finished steps are reused on rerun):
  1. Calibrate D_eff at each switching point (free fraction --free-fraction, default 5%;
     bound dwell 5, 71, 285, 1000 ms)
     and the gap-geometry factor phi = D_eff / D for a free-only walker (L50).
  2. Reference walkers: free-only, D_free = D_eff(switching point) / phi, so their long-range
     spreading matches; verify by calibrating them too (L47).
  3. Capture runs for switching and reference points at 2 nm and 10 nm targets (L48),
     with arrival-time outputs (L49).
  4. Figure + summary table.

--d-ref D replaces step 2's matched reference with a free-only walker at a fixed D (L79);
its D_eff is then phi * D (from the calibration), not measured again. --reuse-switching DIR
copies the switching chunks from an earlier run with the same switching points, layouts and
seeds (molecule seeds depend only on run seed, sweep index and molecule id), so only the
reference walkers are simulated.

Run with 6 threads to leave cores free:
    NUMBA_NUM_THREADS=6 python scripts/dimension_experiment.py --n 100
    python scripts/dimension_experiment.py --replot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpsim import load_config
from fpsim.config import Receptors, Sweep
from fpsim.deff import measure_deff
from fpsim.jobs import Task, config_hash, merge, run_task, stem_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DWELLS = (0.005, 0.0707, 0.285, 1.0)  # s, bound dwell (same values as the 'dwell' sweep)
FREE_FRACTION = 0.05
SIZES = (0.002, 0.010)  # µm capture radius
REF_LABEL = "free-only, same D_eff"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e6e6e6"


def base_config(path, target_ring=None, n_per_cell=None, d_free=None):
    cfg = load_config(path)
    suffix = "" if FREE_FRACTION == 0.05 else f"_f{FREE_FRACTION * 100:g}"
    if d_free is not None:  # free-state D of the switching walker (L84)
        cfg = dataclasses.replace(cfg, diffusion=dataclasses.replace(cfg.diffusion, d_free=d_free))
        suffix += f"_Dfree{d_free:g}"
    if target_ring is not None:
        cfg = dataclasses.replace(cfg, geometry=dataclasses.replace(cfg.geometry, target_ring=target_ring))
        suffix += f"_ring{target_ring}"
    if n_per_cell is not None:
        cfg = dataclasses.replace(cfg, receptors=dataclasses.replace(cfg.receptors, n_per_cell=n_per_cell))
        suffix += f"_rec{n_per_cell}"
    return dataclasses.replace(cfg, name=cfg.name + suffix, sweeps=(Sweep("switch", free_fraction=FREE_FRACTION, bound_dwell=DWELLS),
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


def copy_switching(ecfg, src_dir, run_dir, layouts):
    """Copy switching chunks from an earlier run (same switching points and layouts) and restamp
    them with this run's config hash; the reference points differ, the switching ones do not."""
    run_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(ecfg)
    for i in range(len(DWELLS)):
        for c in range(layouts):
            src, dst = stem_for(src_dir, Task(i, c)), stem_for(run_dir, Task(i, c))
            if dst.with_suffix(".parquet").exists():
                continue
            meta = json.loads(src.with_suffix(".json").read_text())
            assert meta["sweep_point"]["sweep"] == "switch" and abs(meta["sweep_point"]["bound_dwell"] - DWELLS[i]) < 1e-12, meta
            meta.update(config_hash=chash, copied_from=str(src))
            shutil.copy(src.with_suffix(".parquet"), dst.with_suffix(".parquet"))
            dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def capture_runs(cfg, d_ref, out, n, layouts=1, reuse_switching=None):
    """``layouts`` > 1: each point runs ``layouts`` chunks of n // layouts molecules, chunk c on
    shared receptor layout c, so switching and reference points see the same layouts (L57)."""
    dirs = {}
    for a in SIZES:
        ecfg = experiment_config(cfg, d_ref, a)
        run_dir = out / f"runs_a{a * 1e3:.0f}nm"
        if reuse_switching is not None:
            (run_dir).mkdir(parents=True, exist_ok=True)
            manifest = dict(config=ecfg.to_dict(), config_hash=config_hash(ecfg), n_sweep_points=ecfg.n_sweep_points,
                            sweep_labels=[p.label for p in ecfg.sweep_points(1.0)])
            if not (run_dir / "config.json").exists():
                (run_dir / "config.json").write_text(json.dumps(manifest, indent=2, default=float))
            copy_switching(ecfg, reuse_switching / f"runs_a{a * 1e3:.0f}nm", run_dir, layouts)
        for i in range(ecfg.n_sweep_points):
            for c in range(layouts):
                extra = {"layout_seed": c} if layouts > 1 else {}
                path = run_task(ecfg, Task(i, c), n // layouts, run_dir, **extra)
                print(f"  a = {a * 1e3:.0f} nm, point {i}, layout {c}: {'done' if path else 'exists'}", flush=True)
        dirs[a] = run_dir
    return dirs


def paired_ratios(m, n_boot=4000, seed=0):
    """Switching / reference ratio of mean total and local-search time per switching point,
    with 95% CIs from resampling layouts (chunks) jointly for both walkers (L57)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, dwell in enumerate(DWELLS):
        per = []
        for idx in (k, len(DWELLS) + k):
            c = m[(m.sweep_index == idx) & (m.outcome == "captured")]
            g = c.assign(local=c.t_end - c.t_first_target).groupby("chunk")
            per.append(pd.DataFrame({"t_sum": g.t_end.sum(), "l_sum": g.local.sum(), "n": g.size()}))
        sw, rf = per
        layouts = sw.index.intersection(rf.index).to_numpy()
        def ratio_of(sel):
            a, b = sw.loc[sel], rf.loc[sel]
            return (a.t_sum.sum() / a.n.sum()) / (b.t_sum.sum() / b.n.sum()), (a.l_sum.sum() / a.n.sum()) / (b.l_sum.sum() / b.n.sum())
        est = ratio_of(layouts)
        boot = np.array([ratio_of(rng.choice(layouts, len(layouts))) for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows.append(dict(bound_dwell=dwell, n_layouts=len(layouts), total=est[0], total_lo=lo[0], total_hi=hi[0],
                         local=est[1], local_lo=lo[1], local_hi=hi[1]))
    return pd.DataFrame(rows)


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


def plot(out, cal, check, s, molecules, title, size=None, t_min=100.0, paired=None):
    """Switching vs free-only reference at one target size (L52 replot: size comparison dropped)."""
    size = SIZES[0] if size is None else size
    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK})
    fig = plt.figure(figsize=(max(18, 4.5 * len(DWELLS)), 10))
    gs = fig.add_gridspec(2, max(4, len(DWELLS)), height_ratios=[1, 1])
    sw = cal[cal.role == "switch"].reset_index(drop=True)
    m = molecules[size]
    d = s[s.size_nm == round(size * 1e3)]
    roles = (("switch", BLUE, "o", f"switching ({FREE_FRACTION * 100:g}% free)"),
             ("reference", ORANGE, "s", REF_LABEL))

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

    # Top right (when there is room): paired switching / free-only ratios over shared layouts
    if paired is not None and len(DWELLS) < 4:
        ax = fig.add_subplot(gs[0, 3])
        for off, col, mk, label in ((-0.09, "total", "o", "total capture time"), (0.09, "local", "D", "local search")):
            ax.errorbar(x + off, paired[col], yerr=[paired[col] - paired[f"{col}_lo"], paired[f"{col}_hi"] - paired[col]],
                        fmt=mk, color=BLUE if col == "total" else INK, ms=8, capsize=3, label=label)
        ax.axhline(1, color=MUTED, lw=1)
        ax.annotate("1 = no difference", (x[-1] + 0.3, 1), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=MUTED)
        ax.set_xticks(x, xt); ax.set_xlabel("bound dwell of the switching walker (ms)")
        ax.set_ylabel("switching / free-only (mean time)")
        ax.set_title(f"Paired over {int(paired.n_layouts.iloc[0])} shared receptor layouts\n(95% CI, layouts resampled)",
                     loc="left", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        style(ax)

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
    ax.set_ylabel("measured D_eff (µm²/s, 95% CI)"); ax.set_ylim(0, max(0.8, 1.2 * (sw.d_eff + 2 * sw.d_eff_se).max()))
    if REF_LABEL != "free-only, same D_eff":
        ax.set_yscale("log"); ax.set_ylim(0.3, 2 * check.d_eff.max())
    ax.set_title("A. Long-range spreading (D_eff)" + (" is matched" if REF_LABEL == "free-only, same D_eff"
                 else "; free-only = φ·D"), loc="left", fontsize=10)
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
    p.add_argument("--target-ring", type=int, default=None, help="override geometry.target_ring")
    p.add_argument("--receptors-per-cell", type=int, default=None, help="override receptors.n_per_cell")
    p.add_argument("--free-fraction", type=float, default=0.05, help="free fraction of the switching walker")
    p.add_argument("--tag", default="", help="suffix for the run name / output folder (e.g. a code-version label)")
    p.add_argument("--dwells", default="0.005,0.0707,0.285,1.0", help="bound dwells (s), comma-separated")
    p.add_argument("--layouts", type=int, default=1, help="shared receptor layouts per point (n is split across them)")
    p.add_argument("--phi", type=float, default=None, help="override the calibrated phi (free-only D_eff per unit D)")
    p.add_argument("--sizes", default="2,10", help="capture radii in nm, comma-separated (first one is plotted)")
    p.add_argument("--d-free", type=float, default=None, help="override diffusion.d_free (µm²/s) of the switching walker")
    p.add_argument("--d-ref", type=float, default=None,
                   help="free-only reference at this fixed D (µm²/s) instead of matched D_eff (L79)")
    p.add_argument("--reuse-switching", type=Path, default=None,
                   help="copy switching chunks from this earlier run directory (same points, layouts, seeds)")
    p.add_argument("--reuse-calibration", type=Path, default=None,
                   help="copy calibration CSVs from this run directory; valid because calibration runs have no "
                        "receptors and their own enlarged cluster, so they don't depend on targets or receptor count")
    args = p.parse_args()

    global SIZES, FREE_FRACTION, DWELLS, REF_LABEL
    SIZES = tuple(float(v) * 1e-3 for v in args.sizes.split(","))
    FREE_FRACTION = args.free_fraction
    DWELLS = tuple(float(v) for v in args.dwells.split(","))
    cfg = base_config(args.config, args.target_ring, args.receptors_per_cell, args.d_free)
    if args.dwells != "0.005,0.0707,0.285,1.0":
        cfg = dataclasses.replace(cfg, name=cfg.name + "_dw" + "-".join(f"{d * 1e3:g}" for d in DWELLS))
    if args.layouts > 1:
        cfg = dataclasses.replace(cfg, name=cfg.name + f"_paired{args.layouts}")
    if args.d_ref is not None:
        cfg = dataclasses.replace(cfg, name=cfg.name + f"_ref{args.d_ref:g}")
        REF_LABEL = f"free-only, D = {args.d_ref:g} µm²/s"
    if args.tag:
        cfg = dataclasses.replace(cfg, name=cfg.name + f"_{args.tag}")
    out = args.out or PROJECT_ROOT / "results" / "dimension" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_calibration:
        for name in ("calibration.csv",) + (("calibration_reference.csv",) if args.d_ref is None else ()):
            if not (out / name).exists():
                shutil.copy(args.reuse_calibration / name, out / name)
        print(f"   reusing calibration from {args.reuse_calibration}", flush=True)

    print("1. D_eff calibration", flush=True)
    cal = calibrate(cfg, out, args.n, args.cal_t_max, args.short_lag, args.long_lag)
    phi = args.phi if args.phi is not None else float(cal[cal.role == "free_unit"].d_eff.iloc[0])  # D_eff per unit D
    d_eff = cal[cal.role == "switch"].d_eff.to_numpy()
    d_ref = d_eff / phi if args.d_ref is None else np.full(len(DWELLS), args.d_ref)
    stored = out / f"runs_a{SIZES[0] * 1e3:.0f}nm" / "config.json"
    if stored.exists():  # resume: reuse the exact stored values (the CSV round trip can change the last bit)
        prev = next(s for s in json.loads(stored.read_text())["config"]["sweeps"] if s["name"] == "reference")["d_free"]
        prev = np.atleast_1d(np.asarray(prev, dtype=float))
        if prev.shape == d_ref.shape and np.allclose(prev, d_ref, rtol=1e-9, atol=0):
            d_ref = prev
    (out / "reference.json").write_text(json.dumps(dict(phi=phi, d_eff=d_eff.tolist(), d_free_reference=d_ref.tolist(),
                                                        matched=args.d_ref is None, label=REF_LABEL), indent=2))
    print(f"   phi = {phi:.4f}; reference D_free = {np.round(d_ref, 4).tolist()}", flush=True)

    print("2. verify reference D_eff", flush=True)
    if args.d_ref is None:
        check = verify_reference(cfg, d_ref, out, args.n, args.cal_t_max, args.short_lag, args.long_lag)
    else:  # not matched, so not measured: D_eff of a free-only walker is phi * D
        check = pd.DataFrame(dict(bound_dwell=DWELLS, d_eff=phi * d_ref, d_eff_se=0.0, label="predicted"))
        check.to_csv(out / "calibration_reference.csv", index=False)

    if not args.replot:
        print("3. capture runs", flush=True)
        capture_runs(cfg, d_ref, out, args.n, args.layouts, args.reuse_switching)

    s, molecules = summarize(out)
    s.to_csv(out / "summary.csv", index=False)
    paired = None
    if args.layouts > 1:
        paired = paired_ratios(molecules[SIZES[0]])
        paired.to_csv(out / "paired_ratios.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print(s.to_string(index=False))
        if paired is not None:
            print("\npaired switching / free-only (95% CI over layouts):")
            print(paired.to_string(index=False))
    plot(out, cal, check, s, molecules, paired=paired, title=
         f"Switching ({FREE_FRACTION * 100:g}% free) vs non-switching diffusion "
         + ("at matched D_eff" if args.d_ref is None else f"at D_free = {args.d_ref:g} µm²/s (not matched)")
         + f" ({cfg.geometry.dim}D, {args.n} molecules per condition, "
         f"{SIZES[0] * 1e3:.0f} nm target, ring-{cfg.geometry.target_ring} targets with {cfg.receptors.n_per_cell} "
         f"receptors each; " + (f"free-only D_free = D_eff / φ, φ = {phi:.3f}" if args.d_ref is None else f"φ = {phi:.3f}")
         + (f"; {args.layouts} shared layouts)" if args.layouts > 1 else ")"))
    print(f"Saved {out / 'dimension_experiment.png'}")


if __name__ == "__main__":
    main()
