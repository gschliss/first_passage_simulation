"""Figures and tables for the revised paper experiments (paper_instructions/2026_11_18.txt, L125).

    python scripts/paper2_figures.py            # run from the project root, ideally in a Slurm job

Reads every finished cell under results/paper_2026_11_18/ once and writes, under
paper_figures/2026_11_18/<NN_experiment>/, PDF only, every figure with a log-x and a linear-x
panel side by side:

  t20_<axis>_ring<k>.pdf      t_20 vs the swept variable, switching vs free-only, 95% CIs over
                              layouts; fraction ever captured underneath
  rate_<axis>_ring<k>.pdf     the same as an apparent diffusion rate L^2 / (6 t_20) (um^2/s), L the
                              source-to-target-ring distance (ring k x centre spacing)
  arrival_<axis>_ring<k>.pdf  cumulative arrival curves, log and linear time
  ring_t20_<axis>.pdf, ring_rate_<axis>.pdf   x = target ring, one trace per swept value
  deff_<axis>.pdf             D_eff (long lag) and instantaneous D (2 s lag), experiments 1-2
  fgf_vs_hh_<stat>.pdf        75 / 5 / 1% free and free-only vs capture efficiency, rings 1-4
  06_tortuosity/deff_vs_gap   D vs cell-cell gap, solid vs invisible cells (L127)
  summary_<group>.csv         every cell

Within one experiment every figure of a kind shares its axes across rings and series (L124).
Capture efficiency is plotted as a percentage, 100 / N finds. The 0.01%-free and 0.1 ms dwell
points are stand-ins for 0 and are labelled as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpsim.jobs import merge  # noqa: E402
from plot_line import dig, layout_table, level  # noqa: E402
import vizstyle as vz  # noqa: E402

import os  # noqa: E402
# Overridable so the script can be tested on another set (e.g. the 11-02 runs) without touching these.
D = Path(os.environ.get("PAPER2_DATA", "results/paper_2026_11_18"))
F = Path(os.environ.get("PAPER2_FIGS", "paper_figures/2026_11_18"))
N_BOOT = 1000
# (experiment folder, sub-folder or "", x column, axis name, x label, series label)
GROUPS = [
    ("01_confinement", "", "free_pct", "fraction_free", "fraction of time free (%)", None),
    ("02_dwell_time", "", "bound_dwell", "dwell", "mean bound dwell (s)", None),
    ("03_receptor_density", "", "receptors", "receptor_density", "receptors per target cell", None),
    ("04_capture_efficiency", "", "efficiency_pct", "capture_efficiency", "capture efficiency (% = 100 / N finds)", "5% free"),
    ("05_fgf_vs_hh", "75pct_free", "efficiency_pct", "capture_efficiency_75pct_free", "capture efficiency (% = 100 / N finds)", "75% free"),
    ("05_fgf_vs_hh", "1pct_free", "efficiency_pct", "capture_efficiency_1pct_free", "capture efficiency (% = 100 / N finds)", "1% free"),
]
STAND_IN = {"free_pct": 0.01, "bound_dwell": 0.0001}  # simulated in place of 0


def load_cell(run, rng):
    cj = run / "runs_a2nm" / "config.json"
    if not (run / "paired_ratios.csv").exists() or not cj.exists():
        return None
    cfg = json.loads(cj.read_text())["config"]
    g, r = cfg["geometry"], cfg["receptors"]
    mol, _, _ = merge(cj.parent)
    sw, rf = layout_table(mol, "switch", 0), layout_table(mol, "reference", 1)
    n_lay = min(len(sw), len(rf))
    out = dict(run=run.name, ring=g["target_ring"], L=g["target_ring"] * (2 * g["cell_radius"] + g["gap"]),
               free_pct=100 * dig(cfg, ("sweeps", "switch", "free_fraction")),
               bound_dwell=float(np.atleast_1d(dig(cfg, ("sweeps", "switch", "bound_dwell")) or [np.nan])[0]),
               receptors=r["n_per_cell"], finds=r.get("finds_to_capture", 1),
               efficiency_pct=100 / r.get("finds_to_capture", 1), t_max=cfg["run"]["t_max"])
    for role, rows in (("switch", sw), ("ref", rf)):
        t20, cap = level(rows, range(n_lay))
        boot = np.array([level(rows, rng.integers(0, n_lay, n_lay))[0] for _ in range(N_BOOT)], dtype=float)
        lo, hi = (np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan))
        n_all = sum(rows[i][1] for i in range(n_lay))
        times = np.sort(np.concatenate([rows[i][2] for i in range(n_lay)]))
        out.update({f"t20_{role}": t20, f"t20_{role}_lo": lo, f"t20_{role}_hi": hi, f"captured_{role}": cap,
                    f"n_{role}": n_all, f"_curve_{role}": (times, np.arange(1, len(times) + 1) / n_all)})
    for role in ("switch", "ref"):
        k = 6.0
        out[f"rate_{role}"] = out["L"] ** 2 / (k * out[f"t20_{role}"])
        out[f"rate_{role}_lo"] = out["L"] ** 2 / (k * out[f"t20_{role}_hi"])  # long t_20 = slow rate
        out[f"rate_{role}_hi"] = out["L"] ** 2 / (k * out[f"t20_{role}_lo"])
    return out


def tick(xcol, v):
    if xcol in STAND_IN and np.isclose(v, STAND_IN[xcol]):
        return f"{v:g} (~0)"
    return f"{v:g}" if xcol != "receptors" else f"{v:,.0f}"


def span(values, pad=1.25, log=True):
    v = np.asarray([x for x in values if np.isfinite(x) and (x > 0 or not log)], dtype=float)
    if not len(v):
        return None
    return (v.min() / pad, v.max() * pad) if log else (0, v.max() * 1.05)


def both_x(fig, nrows=1, **kw):
    """Columns: log x, linear x (the instructions ask for both scales in one file)."""
    return fig.subplots(nrows, 2, squeeze=False, **kw)


def cline_figure(t, xcol, xlabel, stat, ylim, xr, title, out):
    fig = plt.figure(figsize=(13, 6.6))
    ax = both_x(fig, 2, sharey="row", gridspec_kw=dict(height_ratios=[1.7, 1], hspace=0.1, wspace=0.08))
    for j, scale in enumerate(("log", "linear")):
        for role, colour, mk, lab in (("switch", vz.SWITCH, "o-", "switching"), ("ref", vz.REFERENCE, "s-", "free-only (D = 0.5)")):
            ok = t[t[f"{stat}_{role}"].notna()]
            ax[0, j].errorbar(ok[xcol], ok[f"{stat}_{role}"],
                              yerr=[ok[f"{stat}_{role}"] - ok[f"{stat}_{role}_lo"], ok[f"{stat}_{role}_hi"] - ok[f"{stat}_{role}"]],
                              fmt=mk, color=colour, ms=5, lw=1.6, elinewidth=1, label=lab)
            ax[1, j].plot(t[xcol], t[f"captured_{role}"], mk, color=colour, ms=4, lw=1.4)
        ax[0, j].set_yscale("log")
        for a in ax[:, j]:
            a.set_xscale(scale)
            a.set_xlim(*xr[scale])
            vz.style(a)
        ax[0, j].set_title(f"{scale} x", loc="left", fontsize=10, color=vz.SECONDARY)
        ax[1, j].set_xlabel(xlabel)
        ax[1, j].set_ylim(0, 1.02)
        ax[0, j].tick_params(labelbottom=False)
    if ylim:
        ax[0, 0].set_ylim(*ylim)
    ax[0, 0].set_ylabel("t_20 (s)" if stat == "t20" else "L² / (6 t_20)  (µm²/s)")
    ax[1, 0].set_ylabel("fraction ever captured")
    ax[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle(title + "   (missing points: under 20% ever captured)", x=0.06, ha="left", fontsize=11.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def arrival_figure(t, xcol, xlabel, tr, title, out):
    fig = plt.figure(figsize=(13, 7.4))
    ax = both_x(fig, 2, sharey=True, gridspec_kw=dict(hspace=0.3, wspace=0.08))
    shades = [vz.SEQ(v) for v in np.linspace(0.18, 0.95, len(t))]
    for i, (role, name) in enumerate((("switch", "Switching"), ("ref", "Non-switching (free-only, D = 0.5)"))):
        for j, scale in enumerate(("log", "linear")):
            for (_, r), c in zip(t.iterrows(), shades):
                tt, yy = r[f"_curve_{role}"]
                ax[i, j].step(np.r_[tr[scale][0], tt], np.r_[0, yy], where="post", color=c, lw=1.4,
                              label=tick(xcol, r[xcol]) if (i, j) == (0, 0) else None)
            ax[i, j].axhline(0.2, color=vz.INK, lw=0.8, ls=(0, (4, 3)))
            ax[i, j].set_xscale(scale)
            ax[i, j].set_xlim(*tr[scale])
            ax[i, j].set_ylim(0, 1.02)
            ax[i, j].set_title(f"{name}, {scale} time", loc="left", fontsize=10, color=vz.INK)
            vz.style(ax[i, j])
        ax[i, 0].set_ylabel("fraction of all molecules captured")
    for j in range(2):
        ax[1, j].set_xlabel("time (s)")
    ax[0, 0].legend(fontsize=7.5, title=xlabel, title_fontsize=8, ncol=2 if len(t) > 9 else 1, loc="upper left")
    fig.suptitle(title + "   (dashed: 0.2, where t_20 is read)", x=0.06, ha="left", fontsize=11.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def ring_figure(t, xcol, xlabel, stat, ylim, title, out):
    fig = plt.figure(figsize=(13, 7.4))
    ax = both_x(fig, 2, sharey=True, gridspec_kw=dict(hspace=0.3, wspace=0.08))
    levels = sorted(t[xcol].unique())
    shades = [vz.SEQ(v) for v in np.linspace(0.18, 0.95, len(levels))]
    for i, (role, name) in enumerate((("switch", "Switching"), ("ref", "Non-switching (free-only, D = 0.5)"))):
        for j, scale in enumerate(("log", "linear")):
            for lv, c in zip(levels, shades):
                r = t[t[xcol] == lv].sort_values("ring")
                ax[i, j].plot(r.ring, r[f"{stat}_{role}"], "o-", color=c, lw=1.4, ms=4,
                              label=tick(xcol, lv) if (i, j) == (0, 0) else None)
            ax[i, j].set_xscale(scale)
            ax[i, j].set_yscale("log")
            ax[i, j].set_xticks([1, 2, 3, 4], ["1", "2", "3", "4"])
            ax[i, j].minorticks_off() if scale == "log" else None
            ax[i, j].set_title(f"{name}, {scale} x", loc="left", fontsize=10, color=vz.INK)
            vz.style(ax[i, j])
        ax[i, 0].set_ylabel("t_20 (s)" if stat == "t20" else "L² / (6 t_20)  (µm²/s)")
    if ylim:
        ax[0, 0].set_ylim(*ylim)
    for j in range(2):
        ax[1, j].set_xlabel("target ring (distance from source)")
    ax[0, 0].legend(fontsize=7.5, title=xlabel, title_fontsize=8, ncol=2 if len(levels) > 9 else 1, loc="best")
    fig.suptitle(title, x=0.06, ha="left", fontsize=11.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def deff_figure(exp, xcol, xlabel, out):
    rows = []
    for d in sorted((D / exp / "deff").glob("lattice3d*")):
        f = d / "calibration.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        c = c[c.role == "switch"].iloc[0]
        free = 0.05
        if "_f" in d.name:
            try:
                free = float(d.name.split("_f")[1].split("_")[0]) / 100
            except ValueError:
                pass
        rows.append(dict(free_pct=100 * free, bound_dwell=c.bound_dwell, d_eff=c.d_eff, d_eff_se=c.d_eff_se,
                         d_short=c.d_eff_short, escaped=c.escaped, n=c.n))
    if len(rows) < 2:
        print(f"  {exp}: D_eff skipped ({len(rows)} runs)")
        return None
    t = pd.DataFrame(rows).sort_values(xcol)
    fig = plt.figure(figsize=(13, 4.8))
    ax = both_x(fig, 1, sharey=True, gridspec_kw=dict(wspace=0.08))[0]
    for j, scale in enumerate(("log", "linear")):
        ax[j].errorbar(t[xcol], t.d_eff, yerr=1.96 * t.d_eff_se, fmt="o-", color=vz.SWITCH, ms=5, lw=1.6,
                       label="effective D (very long lag)")
        ax[j].plot(t[xcol], t.d_short, "^--", color=vz.SEQ(0.45), ms=5, lw=1.2, label="instantaneous D (2 s lag)")
        ax[j].axhline(0.5, color=vz.MUTED, lw=1, ls=(0, (4, 3)))
        ax[j].set_xscale(scale)
        ax[j].set_xlabel(xlabel)
        ax[j].set_title(f"{scale} x", loc="left", fontsize=10, color=vz.SECONDARY)
        vz.style(ax[j])
    ax[0].set_ylabel("D (µm²/s)")
    ax[0].set_ylim(0, max(0.55, float(t[["d_eff", "d_short"]].max().max()) * 1.08))
    ax[0].legend(fontsize=9, loc="best")
    fig.suptitle(f"Diffusion vs {xlabel}  (receptor-free 20-ring cluster, {int(t.n.iloc[0])} molecules; "
                 "dashed: microscopic D = 0.5)", x=0.06, ha="left", fontsize=11.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    t.to_csv(out.with_suffix(".csv"), index=False)
    return out


def tortuosity_figure(out):
    """Experiment 6 (L127): D of the pure-3D walker vs the cell-cell gap, solid vs invisible cells."""
    import re
    rows = []
    for kind in ("solid", "invisible", "stepcheck"):
        for d in sorted((D / "06_tortuosity" / kind).glob("lattice3d*")):
            f = d / "calibration.csv"
            m = re.search(r"_gap([\d.]+)nm", d.name)
            if not f.exists() or not m:
                continue
            c = pd.read_csv(f).iloc[0]
            gap = float(m.group(1)) - (20000 - 2 if kind == "invisible" else 0)  # nominal gap at the same spacing
            rows.append(dict(kind=kind, gap_nm=gap, d_eff=c.d_eff, d_eff_se=c.d_eff_se, d_short=c.d_eff_short,
                             escaped=c.escaped, n=c.n))
    # Cost pilots (L128): small-n runs outside the production folders, drawn as separate, labelled
    # points so their wide error bars are not mistaken for production ones. Cell index -> gap.
    pilot_gap = {"cell4": 7.5, "full10_cell0": 1, "full10_cell1": 2, "full10_cell2": 3, "full10_cell30": 1}
    for d in sorted((D / "pilot_tortuosity").glob("*cell*")):
        f = d / "calibration.csv"
        if f.exists() and d.name in pilot_gap:
            c = pd.read_csv(f).iloc[0]
            rows.append(dict(kind="pilot_halfstep" if d.name.endswith("cell30") else "pilot", gap_nm=pilot_gap[d.name],
                             d_eff=c.d_eff, d_eff_se=c.d_eff_se, d_short=c.d_eff_short, escaped=c.escaped, n=c.n))
    if not rows:
        print("  06_tortuosity: no runs yet")
        return
    t = pd.DataFrame(rows).sort_values(["kind", "gap_nm"])
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out.with_suffix(".csv"), index=False)
    fig = plt.figure(figsize=(13, 4.8))
    ax = both_x(fig, 1, sharey=True, gridspec_kw=dict(wspace=0.08))[0]
    for j, scale in enumerate(("log", "linear")):
        for kind, colour, lab in (("solid", vz.SWITCH, "cells solid (molecules reflect)"),
                                  ("invisible", vz.REFERENCE, "cells invisible (pass through)")):
            r = t[t.kind == kind]
            ax[j].errorbar(r.gap_nm, r.d_eff, yerr=1.96 * r.d_eff_se, fmt="o-", color=colour, ms=5, lw=1.6,
                           label=f"{lab}: effective D (5000 s lag)")
            ax[j].plot(r.gap_nm, r.d_short, "^--", color=colour, ms=4, lw=1, alpha=0.7,
                       label=f"{lab}: instantaneous D (2 s lag)")
        for kind, mk, lab in (("pilot", "s", "solid, cost pilot (n = {n}): effective D"),
                              ("pilot_halfstep", "D", "solid, cost pilot, half step (n = {n}): effective D")):
            pr = t[t.kind == kind]
            for n_mol, g in pr.groupby("n"):
                ax[j].errorbar(g.gap_nm, g.d_eff, yerr=1.96 * g.d_eff_se, fmt=mk, color=vz.SWITCH, ms=6,
                               mfc="white", elinewidth=1, capsize=2, label=lab.format(n=int(n_mol)))
        sc = t[t.kind == "stepcheck"]
        if len(sc):
            ax[j].errorbar(sc.gap_nm, sc.d_eff, yerr=1.96 * sc.d_eff_se, fmt="D", color=vz.INK, ms=6,
                           mfc="none", label="1 nm gap, half the step size (accuracy check)")
        ax[j].axhline(0.5, color=vz.MUTED, lw=1, ls=(0, (4, 3)))
        ax[j].set_xscale(scale)
        ax[j].set_xlabel("cell-cell gap (nm)")
        ax[j].set_title(f"{scale} x", loc="left", fontsize=10, color=vz.SECONDARY)
        vz.style(ax[j])
    ax[0].set_ylabel("D (µm²/s)")
    ax[0].set_ylim(0, 0.56)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Tortuosity: pure 3D diffusion vs cell-cell gap  (no binding, no receptors, 20-ring "
                 "cluster, 1000 molecules; dashed: D = 0.5)", x=0.06, ha="left", fontsize=11.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  06_tortuosity: {len(t)} runs")


def main():
    vz.use()
    plt.rcParams["pdf.fonttype"] = 42
    rng = np.random.default_rng(0)
    cells = {}
    for exp, sub, *_ in GROUPS:
        base = D / exp / sub if sub else D / exp
        loaded = [c for c in (load_cell(r, rng) for r in sorted(base.glob("lattice3d*"))) if c is not None]
        cells[(exp, sub)] = pd.DataFrame(loaded)
        print(f"  {exp}/{sub or '.'}: {len(loaded)} cells")

    for exp in dict.fromkeys(g[0] for g in GROUPS):
        mine = [g for g in GROUPS if g[0] == exp and len(cells[(g[0], g[1])])]
        if not mine:
            continue
        (F / exp).mkdir(parents=True, exist_ok=True)
        allt = pd.concat([cells[(g[0], g[1])] for g in mine])
        xcol = mine[0][2]
        xs = allt[xcol][allt[xcol] > 0]
        xr = {"log": (xs.min() / 1.4, xs.max() * 1.4), "linear": (0 - 0.03 * xs.max(), xs.max() * 1.05)}
        tmin = min((c[0][0] for col in ("_curve_switch", "_curve_ref") for c in allt[col] if len(c[0])), default=1.0)
        tmax = max(allt.t_max)
        tr = {"log": (max(tmin / 1.5, 0.1), tmax * 1.1), "linear": (0, tmax * 1.02)}
        ylims = {s: span(pd.concat([allt[f"{s}_{r}{e}"] for r in ("switch", "ref") for e in ("", "_lo", "_hi")]))
                 for s in ("t20", "rate")}
        for e, sub, xcol, name, xlabel, series in mine:
            t = cells[(e, sub)]
            tag = f"{exp}{' / ' + sub if sub else ''}"
            for k in sorted(t.ring.unique()):
                tk = t[t.ring == k].sort_values(xcol)
                for stat in ("t20", "rate"):
                    cline_figure(tk, xcol, xlabel, stat, ylims[stat], xr,
                                 f"{'t_20' if stat == 't20' else 'Apparent diffusion rate L²/(6 t_20)'} vs {xlabel}, "
                                 f"target ring {k}  ({tag})", F / exp / f"{stat}_{name}_ring{k}.pdf")
                arrival_figure(tk, xcol, xlabel, tr, f"Cumulative arrival, target ring {k}  ({tag})",
                               F / exp / f"arrival_{name}_ring{k}.pdf")
            for stat in ("t20", "rate"):
                ring_figure(t, xcol, xlabel, stat, ylims[stat],
                            f"{'t_20' if stat == 't20' else 'L²/(6 t_20)'} vs target ring, one trace per {xlabel}  ({tag})",
                            F / exp / f"ring_{stat}_{name}.pdf")
            t.drop(columns=[c for c in t.columns if c.startswith("_")]).to_csv(
                F / exp / f"summary_{exp}{'__' + sub if sub else ''}.csv", index=False)
            print(f"  {tag}: figures for {t.ring.nunique()} rings")
    for exp, xcol, xlabel in (("01_confinement", "free_pct", "fraction of time free (%)"),
                              ("02_dwell_time", "bound_dwell", "mean bound dwell (s)")):
        if (D / exp).exists():
            deff_figure(exp, xcol, xlabel, F / exp / f"deff_{'fraction_free' if exp.startswith('01') else 'dwell'}.pdf")

    tortuosity_figure(F / "06_tortuosity" / "deff_vs_gap.pdf")

    # Pseudo-FGF vs pseudo-HH: 75 / 5 / 1% free and free-only, all four rings, both x scales.
    series = [(cells.get(("05_fgf_vs_hh", "75pct_free")), vz.SEQ(0.95), "high 3D: 75% free (pseudo-FGF)"),
              (cells.get(("04_capture_efficiency", "")), vz.SEQ(0.6), "medium 3D: 5% free (pseudo-HH)"),
              (cells.get(("05_fgf_vs_hh", "1pct_free")), vz.SEQ(0.3), "low 3D: 1% free")]
    series = [s for s in series if s[0] is not None and len(s[0])]
    if series:
        (F / "05_fgf_vs_hh").mkdir(parents=True, exist_ok=True)
        allt = pd.concat([s[0] for s in series])
        for stat in ("t20", "rate"):
            fig, ax = plt.subplots(2, 4, figsize=(17, 7.6), sharey=True, gridspec_kw=dict(hspace=0.32, wspace=0.06))
            for j, k in enumerate((1, 2, 3, 4)):
                for i, scale in enumerate(("log", "linear")):
                    a = ax[i, j]
                    for t, colour, lab in series:
                        r = t[t.ring == k].sort_values("efficiency_pct")
                        a.plot(r.efficiency_pct, r[f"{stat}_switch"], "o-", color=colour, ms=4, lw=1.5, label=lab)
                    r = series[0][0]
                    r = r[r.ring == k].sort_values("efficiency_pct")
                    a.plot(r.efficiency_pct, r[f"{stat}_ref"], "s--", color=vz.REFERENCE, ms=4, lw=1.3, label="free-only (D = 0.5)")
                    a.set_xscale(scale)
                    a.set_yscale("log")
                    a.set_title(f"target ring {k}, {scale} x", loc="left", fontsize=10, color=vz.INK)
                    vz.style(a)
                    if i == 1:
                        a.set_xlabel("capture efficiency (%)")
            lim = span(pd.concat([allt[f"{stat}_{r}"] for r in ("switch", "ref")]))
            if lim:
                ax[0, 0].set_ylim(*lim)
            for i in range(2):
                ax[i, 0].set_ylabel("t_20 (s)" if stat == "t20" else "L² / (6 t_20)  (µm²/s)")
            ax[0, 0].legend(fontsize=8, loc="best")
            fig.suptitle(f"Pseudo-FGF vs pseudo-HH: {'t_20' if stat == 't20' else 'apparent diffusion rate'} vs capture "
                         "efficiency (1 s dwell, 5000 receptors; missing points: under 20% captured)",
                         x=0.05, ha="left", fontsize=12)
            fig.savefig(F / "05_fgf_vs_hh" / f"fgf_vs_hh_{stat}.pdf", bbox_inches="tight")
            plt.close(fig)
        print("  05_fgf_vs_hh: overlays")


if __name__ == "__main__":
    main()
