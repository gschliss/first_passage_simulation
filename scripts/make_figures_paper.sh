#!/usr/bin/env bash
# Figures and tables for the paper experiments (paper_instructions/2026_11_02.txt, L121).
# Safe to run mid-matrix: cells without paired_ratios.csv are skipped. From the project root:
#   bash scripts/make_figures_paper.sh
#
# Raw runs:  results/paper_2026_11_02/<NN_experiment>/<run>/   (gitignored, on Sherlock)
# Output:    paper_figures/2026_11_02/<NN_experiment>/         (tracked, pushed to GitHub)
#   t20_<axis>_ring<k>.png/.pdf       t_20 levels, switching vs free-only (plot_line.py)
#   arrival_<axis>_ring<k>.png/.pdf   cumulative arrival curves behind them (plot_capture_cline.py)
#   deff_<axis>.png/.pdf/.csv         effective diffusion (experiments 1-2, plot_deff.py)
#   summary_<NN_experiment>.csv       every cell: axis value, ring, t_20s, capture fractions, CIs
#
# Axes (L124): within one experiment, every figure of one type (t_20, arrival, D_eff) shares its
# x and y ranges across rings 1-4 and across clines, so they can be compared side by side. Pass 1
# draws everything and records the ranges matplotlib chose; pass 2 redraws with their union.
set -uo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

PY=.venv/bin/python
D=results/paper_2026_11_02
F=paper_figures/2026_11_02
mkdir -p "$F"
LIM=$(mktemp -d "${TMPDIR:-/tmp}/paper_limits.XXXXXX")
trap 'rm -rf "$LIM"' EXIT

pick() { A=(); for d in "$@"; do [ -f "$d/paired_ratios.csv" ] && A+=("$d"); done; }
lim() { [ -f "$LIM/$1__$2.lim" ] && cat "$LIM/$1__$2.lim"; }  # lim <exp> <kind> -> "xmin xmax ymin ymax"

cline() {  # cline <exp> <x field> <axis name> <ring> <title> <run dirs...>
  local exp=$1 x=$2 name=$3 k=$4 title=$5; shift 5
  pick "$@"
  mkdir -p "$F/$exp"
  [ "$PASS" = 2 ] && printf '%-24s ring %d  %2d points  ' "$exp" "$k" "${#A[@]}"
  if [ "${#A[@]}" -lt 2 ]; then [ "$PASS" = 2 ] && echo "(skipped)"; return; fi
  local tl=() al=() x0 x1 y0 y1 r1 r2
  if [ "$PASS" = 1 ]; then
    tl=(--limits-out "$LIM/${exp}__t20__${name}_$k.json")
    al=(--limits-out "$LIM/${exp}__arrival__${name}_$k.json")
  else
    read -r x0 x1 y0 y1 <<< "$(lim "$exp" t20)"; [ -n "${y1:-}" ] && tl=(--xlim "$x0" "$x1" --ylim "$y0" "$y1")
    read -r x0 x1 y0 y1 <<< "$(lim "$exp" arrival)"; [ -n "${x1:-}" ] && al=(--tlim "$x0" "$x1")
  fi
  $PY scripts/plot_line.py "${A[@]}" --x "$x" --out "$F/$exp/t20_${name}_ring$k.png" ${tl[@]+"${tl[@]}"} \
      --title "t_20 vs $title  (target ring $k)" > /dev/null 2>&1 && r1="t20 ok" || r1="t20 FAILED"
  $PY scripts/plot_capture_cline.py "${A[@]}" --x "$x" --out "$F/$exp/arrival_${name}_ring$k.png" ${al[@]+"${al[@]}"} \
      --title "Cumulative arrival vs $title  (target ring $k)" > /dev/null 2>&1 && r2="curves ok" || r2="curves FAILED"
  [ "$PASS" = 2 ] && echo "$r1 + $r2"
  return 0
}

deff() {  # deff <exp> <x field> <out name> <title> <run dirs...> [-- <comparison dirs...>]
  local exp=$1 x=$2 name=$3 title=$4; shift 4
  local runs=() cmp=() seen=0 a x0 x1 y0 y1 out
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; elif [ $seen = 0 ]; then runs+=("$a"); else cmp+=("$a"); fi
  done
  if [ "${#runs[@]}" -lt 2 ]; then [ "$PASS" = 2 ] && echo "$exp $name (skipped)"; return 0; fi
  local extra=()
  [ "${#cmp[@]}" -gt 0 ] && extra+=(--compare "${cmp[@]}")
  local fo=($D/01_confinement/deff/*_f100_*)
  [ "$exp" = 02_dwell_time ] && [ "${#cmp[@]}" -eq 0 ] && [ "${#fo[@]}" -gt 0 ] && extra+=(--free-only "${fo[0]}")
  if [ "$PASS" = 1 ]; then
    extra+=(--limits-out "$LIM/${exp}__deff__${name}.json")
  else
    read -r x0 x1 y0 y1 <<< "$(lim "$exp" deff)"; [ -n "${y1:-}" ] && extra+=(--xlim "$x0" "$x1" --ylim "$y0" "$y1")
  fi
  out=$($PY scripts/plot_deff.py "${runs[@]}" --x "$x" --out "$F/$exp/$name.png" "${extra[@]}" --title "$title" 2>&1 | tail -1)
  [ "$PASS" = 2 ] && echo "$out"
  return 0
}

draw_all() {
  local CONV="3D, 1000 molecules per walker, t_max 100 000 s" k long keep d f l skip
  for k in 1 2 3 4; do
    cline 01_confinement free_fraction fraction_free $k "fraction of time free  ($CONV, 1 s dwell, 1000 receptors, first passage)" \
      $D/01_confinement/lattice3d*_ring${k}_*
    cline 02_dwell_time bound_dwell dwell $k "bound dwell  ($CONV; 500 000 s for 278 s and 1000 s; 5% free, 1000 receptors)" \
      $D/02_dwell_time/lattice3d*_ring${k}_*
    cline 03_receptor_density n_per_cell receptor_density $k "receptors per target cell  ($CONV, 5% free, 1 s dwell)" \
      $D/03_receptor_density/lattice3d*_ring${k}_*
    cline 04_capture_efficiency finds_to_capture capture_efficiency $k "capture efficiency 1/N  ($CONV, 5% free, 1 s dwell)" \
      $D/04_capture_efficiency/lattice3d*_ring${k}_*
    cline 05_fgf_vs_hh finds_to_capture capture_efficiency_50pct_free $k "capture efficiency 1/N, 50% free (pseudo-FGF)  ($CONV)" \
      $D/05_fgf_vs_hh/lattice3d*_ring${k}_*
    cline 05_fgf_vs_hh finds_to_capture capture_efficiency_1pct_free $k "capture efficiency 1/N, 1% free (intermediate)  ($CONV)" \
      $D/05_fgf_vs_hh/1pct_free/lattice3d*_ring${k}_*
  done
  # The low-free cells rerun at t_max 500 000 s (L122), in place of their 100 000 s versions.
  for k in 3 4; do
    long=($D/01_confinement/tmax500000/lattice3d*_ring${k}_*)
    [ "${#long[@]}" -eq 0 ] && continue
    keep=()
    for d in $D/01_confinement/lattice3d*_ring${k}_*; do
      f=$(basename "$d" | sed -n 's/^lattice3d_f\([0-9.]*\)_.*/\1/p')
      skip=0; for l in "${long[@]}"; do [ -n "$f" ] && [[ $(basename "$l") == lattice3d_f${f}_* ]] && skip=1; done
      [ $skip -eq 0 ] && keep+=("$d")
    done
    cline 01_confinement free_fraction fraction_free_longtail $k \
      "fraction of time free, lowest points at t_max 500 000 s  (3D, 1000 molecules per walker, 1 s dwell)" \
      "${long[@]}" "${keep[@]}"
  done
  deff 01_confinement free_fraction deff_fraction_free \
    "Effective diffusion vs fraction free  (receptor-free, 12-ring cluster, 1000 molecules)" $D/01_confinement/deff/*
  deff 02_dwell_time bound_dwell deff_dwell \
    "Effective diffusion vs dwell  (receptor-free, 12-ring cluster, 1000 molecules)" $D/02_dwell_time/deff/*
  deff 01_confinement free_fraction deff_fraction_free_20rings \
    "Effective diffusion vs fraction free  (20-ring cluster; faded: 12-ring)" $D/01_confinement/deff_20rings/* -- $D/01_confinement/deff/*
  deff 02_dwell_time bound_dwell deff_dwell_20rings \
    "Effective diffusion vs dwell  (20-ring cluster; faded: 12-ring)" $D/02_dwell_time/deff_20rings/* -- $D/02_dwell_time/deff/*
}

PASS=1; draw_all
# Union of the recorded ranges, per experiment and figure type.
$PY - "$LIM" <<'PYEOF'
import json, sys
from collections import defaultdict
from pathlib import Path
lim = Path(sys.argv[1])
box = defaultdict(lambda: [float("inf"), float("-inf"), float("inf"), float("-inf")])
for f in lim.glob("*.json"):
    exp, kind = f.stem.split("__")[:2]
    d, b = json.loads(f.read_text()), box[(f.stem.split("__")[0], f.stem.split("__")[1])]
    b[0], b[1] = min(b[0], d["x"][0]), max(b[1], d["x"][1])
    b[2], b[3] = min(b[2], d["y"][0]), max(b[3], d["y"][1])
for (exp, kind), b in sorted(box.items()):
    (lim / f"{exp}__{kind}.lim").write_text(" ".join(repr(v) for v in b))
    print(f"  shared axes {exp:24} {kind:8} x {b[0]:.4g}-{b[1]:.4g}  y {b[2]:.4g}-{b[3]:.4g}")
PYEOF
echo "== figures (shared axes within each experiment) =="
PASS=2; draw_all

echo "== tables and the FGF vs HH overlay =="
$PY - <<'PYEOF'
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import vizstyle as vz

D, F = Path("results/paper_2026_11_02"), Path("paper_figures/2026_11_02")
rows = []
# Every folder holding run directories: the experiments, and sub-folders such as
# 05_fgf_vs_hh/1pct_free or 01_confinement/tmax500000 (L122-L123), labelled by relative path.
groups = sorted({r.parent for r in D.glob("0*/**/lattice3d*") if (r / "paired_ratios.csv").exists()}) if D.exists() else []
for exp in groups:
    label = str(exp.relative_to(D)).replace("/", "__")
    for run in sorted(exp.glob("lattice3d*")):
        pr, cj = run / "paired_ratios.csv", run / "runs_a2nm" / "config.json"
        if not (pr.exists() and cj.exists()):
            continue
        cfg = json.loads(cj.read_text())["config"]
        sw = next(s for s in cfg["sweeps"] if s["name"] == "switch")
        p = pd.read_csv(pr).iloc[0]
        rows.append(dict(experiment=label, target_ring=cfg["geometry"]["target_ring"],
                         free_fraction=sw.get("free_fraction"), bound_dwell=p.bound_dwell,
                         receptors_per_cell=cfg["receptors"]["n_per_cell"],
                         finds_to_capture=cfg["receptors"].get("finds_to_capture", 1),
                         capture_efficiency=1 / cfg["receptors"].get("finds_to_capture", 1),
                         t_max=cfg["run"]["t_max"], t20_switch=p.t20_switch, t20_ref=p.t20_ref,
                         t20_ratio=p.t20, t20_ratio_lo=p.t20_lo, t20_ratio_hi=p.t20_hi,
                         captured_switch=p.captured_switch, captured_ref=p.captured_ref, run=run.name))
t = pd.DataFrame(rows)
for exp, g in (t.groupby("experiment") if len(t) else []):
    top = exp.split("__")[0]
    (F / top).mkdir(parents=True, exist_ok=True)
    g.to_csv(F / top / f"summary_{exp}.csv", index=False)
    print(f"  {exp}: {len(g)} cells -> {top}/summary_{exp}.csv")

# t_20 against distance (L124): x = target ring, one trace per sweep value. Every ring plot of one
# experiment shares its y ranges (the 05 folder has two: 50% and 1% free).
import numpy as np
AXIS = {"01_confinement": ("free_fraction", "free_fraction", "fraction_free"),
        "02_dwell_time": ("bound_dwell", "bound_dwell", "dwell"),
        "03_receptor_density": ("receptors_per_cell", "n_per_cell", "receptor_density"),
        "04_capture_efficiency": ("finds_to_capture", "finds_to_capture", "capture_efficiency"),
        "05_fgf_vs_hh": ("finds_to_capture", "finds_to_capture", "capture_efficiency_50pct_free"),
        "05_fgf_vs_hh__1pct_free": ("finds_to_capture", "finds_to_capture", "capture_efficiency_1pct_free")}
vz.use()
for top in sorted({e.split("__")[0] for e in AXIS}):
    labels = [e for e in AXIS if e.split("__")[0] == top and e in set(t.experiment)] if len(t) else []
    if not labels:
        continue
    sub = t[t.experiment.isin(labels)]
    vals = pd.concat([sub.t20_switch, sub.t20_ref]).dropna()
    ylim = (vals.min() / 1.4, vals.max() * 1.4) if len(vals) else None
    for e in labels:
        col, field, name = AXIS[e]
        g = t[t.experiment == e]
        levels = sorted(g[col].unique())
        shades = [vz.SEQ(v) for v in np.linspace(0.18, 0.95, len(levels))]
        ref_same = all(np.allclose(g[g[col] == lv].sort_values("target_ring").t20_ref.fillna(-1).to_numpy(),
                                   g[g[col] == levels[0]].sort_values("target_ring").t20_ref.fillna(-1).to_numpy())
                       for lv in levels)
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True, sharey="row",
                                 gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12, wspace=0.07))
        for j, (role, title) in enumerate((("switch", "Switching"), ("ref", "Non-switching (free-only, D = 0.5)"))):
            drawn = levels[:1] if (role == "ref" and ref_same) else levels
            for lv, c in zip(drawn, shades if len(drawn) > 1 else [vz.SEQ(0.62)]):
                r = g[g[col] == lv].sort_values("target_ring")
                lab = vz.AXIS_TICK[field](lv) if len(drawn) > 1 else "every point (identical)"
                axes[0, j].plot(r.target_ring, r[f"t20_{role}"], "o-", color=c, lw=1.8, ms=6, label=lab)
                axes[1, j].plot(r.target_ring, r[f"captured_{role}"], "o-", color=c, lw=1.8, ms=6)
            axes[0, j].set_title(title, loc="left", fontsize=11, color=vz.INK)
            axes[0, j].set_yscale("log")
            axes[1, j].set_xlabel("target ring (distance from source)")
            axes[1, j].set_xticks([1, 2, 3, 4])
            for a in axes[:, j]:
                vz.style(a)
            axes[0, j].legend(fontsize=8, title=vz.AXIS_LABEL[field], title_fontsize=8, loc="upper left",
                              ncol=2 if len(drawn) > 5 else 1, alignment="left")
        if ylim:
            axes[0, 0].set_ylim(*ylim)
        axes[1, 0].set_ylim(0, 1.02)
        axes[0, 0].set_ylabel("t_20 (s)")
        axes[1, 0].set_ylabel("fraction ever captured")
        fig.suptitle(f"t_20 vs distance, one trace per {vz.AXIS_LABEL[field]}  ({e.replace('__', ' / ')}; "
                     "missing points = under 20% ever captured)", x=0.07, ha="left", fontsize=12)
        print("  " + ", ".join(str(q) for q in vz.save(fig, F / top / f"ring_t20_{name}.png")))
        plt.close(fig)

# Pseudo-FGF (50% free) vs 5% free (pseudo-HH, experiment 4) vs 1% free vs free-only.
# sharey by row (L124): the four rings are compared on one t_20 axis and one capture axis.
vz.use()
both = t[t.experiment.isin(["04_capture_efficiency", "05_fgf_vs_hh", "05_fgf_vs_hh__1pct_free"])] if len(t) else t
if len(both):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.25))
    for j, k in enumerate((1, 2, 3, 4)):
        for exp, colour, lab in (("05_fgf_vs_hh", vz.SEQ(0.95), "high 3D: 50% free (pseudo-FGF)"),
                                 ("04_capture_efficiency", vz.SEQ(0.6), "low 3D: 5% free (pseudo-HH)"),
                                 ("05_fgf_vs_hh__1pct_free", vz.SEQ(0.3), "lower 3D: 1% free")):
            g = both[(both.experiment == exp) & (both.target_ring == k)].sort_values("capture_efficiency")
            axes[0, j].plot(g.capture_efficiency, g.t20_switch, "o-", color=colour, label=lab)
            axes[1, j].plot(g.capture_efficiency, g.captured_switch, "o-", color=colour)
        g = both[(both.experiment == "04_capture_efficiency") & (both.target_ring == k)].sort_values("capture_efficiency")
        axes[0, j].plot(g.capture_efficiency, g.t20_ref, "s--", color=vz.REFERENCE, label="free-only (D = 0.5)")
        axes[1, j].plot(g.capture_efficiency, g.captured_ref, "s--", color=vz.REFERENCE)
        axes[0, j].set_title(f"target ring {k}", loc="left", fontsize=11)
        for a in axes[:, j]:
            a.set_xscale("log"); vz.style(a)
        axes[0, j].set_yscale("log")
        axes[1, j].set_xlabel("capture efficiency (1/N finds)")
        axes[1, j].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("t_20 (s)")
    axes[1, 0].set_ylabel("fraction ever captured")
    axes[0, 0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Pseudo-FGF vs pseudo-HH: t_20 and capture fraction vs capture efficiency "
                 "(1 s dwell, 1000 receptors, t_max 100 000 s; missing points = under 20% ever captured)",
                 x=0.06, ha="left", fontsize=12)
    print("  " + ", ".join(str(q) for q in vz.save(fig, F / "05_fgf_vs_hh" / "fgf_vs_hh_overlay.png")))
PYEOF
echo
echo "paper cells complete: $(ls -d $D/0*/lattice3d*/ $D/0*/*/lattice3d*/ 2>/dev/null | while read -r d; do [ -f "$d/paired_ratios.csv" ] && echo x; done | wc -l | tr -d ' ') of 243"
