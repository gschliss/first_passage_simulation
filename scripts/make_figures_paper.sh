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
set -uo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

PY=.venv/bin/python
D=results/paper_2026_11_02
F=paper_figures/2026_11_02
mkdir -p "$F"

pick() { A=(); for d in "$@"; do [ -f "$d/paired_ratios.csv" ] && A+=("$d"); done; }

cline() {  # cline <exp> <x field> <axis name> <ring> <title> <run dirs...>
  local exp=$1 x=$2 name=$3 k=$4 title=$5; shift 5
  pick "$@"
  mkdir -p "$F/$exp"
  printf '%-24s ring %d  %2d points  ' "$exp" "$k" "${#A[@]}"
  if [ "${#A[@]}" -lt 2 ]; then echo "(skipped)"; return; fi
  $PY scripts/plot_line.py "${A[@]}" --x "$x" --out "$F/$exp/t20_${name}_ring$k.png" \
      --title "t_20 vs $title  (target ring $k)" > /dev/null 2>&1 && printf "t20 ok" || printf "t20 FAILED"
  $PY scripts/plot_capture_cline.py "${A[@]}" --x "$x" --out "$F/$exp/arrival_${name}_ring$k.png" \
      --title "Cumulative arrival vs $title  (target ring $k)" > /dev/null 2>&1 \
      && echo " + curves ok" || echo " + curves FAILED"
}

CONV="3D, 1000 molecules per walker, t_max 100 000 s"
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

echo "== effective diffusion =="
for spec in "01_confinement free_fraction fraction_free" "02_dwell_time bound_dwell dwell"; do
  set -- $spec
  runs=($D/$1/deff/*)
  if [ "${#runs[@]}" -lt 2 ]; then echo "$1 D_eff (skipped)"; continue; fi
  extra=(); fo=($D/01_confinement/deff/*_f100_*); [ "$1" = 02_dwell_time ] && [ "${#fo[@]}" -gt 0 ] && extra=(--free-only "${fo[0]}")
  $PY scripts/plot_deff.py "${runs[@]}" --x $2 --out "$F/$1/deff_$3.png" ${extra[@]+"${extra[@]}"} \
    --title "Effective diffusion vs ${3//_/ }  (receptor-free, 1000 molecules)" 2>&1 | tail -1
done

# L122: D_eff again in a 20-ring cluster, and the low-free cells rerun at t_max 500 000 s.
for spec in "01_confinement free_fraction fraction_free" "02_dwell_time bound_dwell dwell"; do
  set -- $spec
  runs=($D/$1/deff_20rings/*)
  if [ "${#runs[@]}" -lt 2 ]; then echo "$1 D_eff 20 rings (skipped)"; continue; fi
  $PY scripts/plot_deff.py "${runs[@]}" --x $2 --out "$F/$1/deff_$3_20rings.png" --compare $D/$1/deff/* \
    --title "Effective diffusion vs ${3//_/ }  (20-ring cluster; faded: 12-ring)" 2>&1 | tail -1
done
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

echo "== tables and the FGF vs HH overlay =="
$PY - <<'EOF'
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

# Pseudo-FGF (50% free, experiment 5) vs pseudo-HH (5% free, experiment 4) vs free-only.
vz.use()
both = t[t.experiment.isin(["04_capture_efficiency", "05_fgf_vs_hh", "05_fgf_vs_hh__1pct_free"])] if len(t) else t
if len(both):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True, gridspec_kw=dict(hspace=0.25))
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
EOF
echo
echo "paper cells complete: $(ls -d $D/0*/lattice3d*/ 2>/dev/null | while read -r d; do [ -f "$d/paired_ratios.csv" ] && echo x; done | wc -l | tr -d ' ') of 200"
