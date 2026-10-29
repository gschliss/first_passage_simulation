#!/usr/bin/env bash
# Rebuild every report figure from whatever runs have finished (L96).
# Safe to run mid-matrix: cells without paired_ratios.csv are skipped, partial grids
# just have blank cells. Run from the project root:  bash scripts/make_figures.sh
set -uo pipefail
shopt -s extglob
cd "$(dirname "$0")/.."

PY=.venv/bin/python
D=results/dimension
F=docs/figures   # committed alongside docs/report_d05.md, which links to them
mkdir -p "$F"

# Run dirs are lattice3d[_f<pct>][_finds<N>]_ring<R>_rec<n>_dw<ms>[_tmax<t>]_paired<L>_ref0.5_d05,
# so a bare lattice3d prefix means 5% free and N = 1. The N = 1 globs must exclude every
# _findsN, not just finds5 (L96: grep -v finds5 let finds2/finds10 into the N = 1 cells).
# Collect run dirs matching a listing command, keeping only finished ones.
pick() { A=(); while IFS= read -r d; do [ -f "$d/paired_ratios.csv" ] && A+=("$d"); done < <(eval "$1"); }

grid() {  # grid <listing> <x> <y> <out> <title>
  pick "$1"
  printf '%-34s %2d cells  ' "$(basename "$4" .png)" "${#A[@]}"
  if [ "${#A[@]}" -lt 2 ]; then echo "(skipped)"; return; fi
  $PY scripts/plot_grid.py "${A[@]}" --x "$2" --y "$3" --out "$F/$4" --title "$5" > /dev/null 2>&1 \
    && echo "ok" || echo "FAILED"
}

# Each cline yields two figures from ONE glob: the t_20 levels, and the cumulative capture
# curves those levels are read off (L107). Deriving both here keeps them from drifting apart.
line() {  # line <listing> <x> <out> <title>
  pick "$1"
  printf '%-34s %2d points ' "$(basename "$3" .png)" "${#A[@]}"
  if [ "${#A[@]}" -lt 2 ]; then echo "(skipped)"; return; fi
  $PY scripts/plot_line.py "${A[@]}" --x "$2" --out "$F/$3" --title "$4" > /dev/null 2>&1 \
    && printf "ok" || printf "FAILED"
  $PY scripts/plot_capture_cline.py "${A[@]}" --x "$2" --out "$F/${3/line_/capture_}" \
    --title "${4/t_20 vs/Cumulative capture vs}" > /dev/null 2>&1 \
    && echo " + curves ok" || echo " + curves FAILED"
}

BASE="1 s dwell, 1000 receptors"
echo "== grids =="
grid "ls -d $D/lattice3d@(|_f+([0-9.]))_ring[1234]_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     target_ring free_fraction grid_free_x_distance_N1.png \
     "t_20 across fraction free x distance  (3D, $BASE, first passage)"
grid "ls -d $D/lattice3d@(|_f+([0-9.]))_finds5_ring[1234]_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     target_ring free_fraction grid_free_x_distance_N5.png \
     "t_20 across fraction free x distance  (3D, $BASE, 5 finds)"
grid "ls -d $D/lattice3d@(|_f+([0-9.]))_ring2_rec*_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell free_fraction grid_free_x_density_N1.png \
     "t_20 across fraction free x receptor density  (3D, ring 2, 1 s dwell, first passage)"
grid "ls -d $D/lattice3d@(|_f+([0-9.]))_finds5_ring2_rec*_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell free_fraction grid_free_x_density_N5.png \
     "t_20 across fraction free x receptor density  (3D, ring 2, 1 s dwell, 5 finds)"

echo "== lines =="
line "ls -d $D/lattice3d@(|_f+([0-9.]))_ring2_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     free_fraction line_free_fraction.png \
     "t_20 vs fraction free  (3D, ring 2, $BASE, first passage)"
line "ls -d $D/lattice3d@(|_f+([0-9.]))_finds5_ring2_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     free_fraction line_free_fraction_N5.png \
     "t_20 vs fraction free  (3D, ring 2, $BASE, 5 finds)"
line "ls -d $D/lattice3d_ring[1234]_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     target_ring line_distance.png \
     "t_20 vs distance  (3D, 5% free, $BASE, first passage)"
line "ls -d $D/lattice3d_finds5_ring[1234]_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     target_ring line_distance_N5.png \
     "t_20 vs distance  (3D, 5% free, $BASE, 5 finds)"
line "ls -d $D/lattice3d_ring2_rec*_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell line_density.png \
     "t_20 vs receptor density  (3D, ring 2, 5% free, 1 s dwell, first passage)"
line "ls -d $D/lattice3d_finds5_ring2_rec*_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell line_density_N5.png \
     "t_20 vs receptor density  (3D, ring 2, 5% free, 1 s dwell, 5 finds)"
# The dwell line spans two t_max settings, so it is assembled from both naming patterns.
line "ls -d $D/lattice3d_ring2_rec1000_dw+([0-9.])@(|_tmax100000)_paired20_ref0.5_d05 2>/dev/null" \
     bound_dwell line_dwell.png \
     "t_20 vs bound dwell  (3D, ring 2, 5% free, 1000 receptors, first passage)"
# N = 5 needs no t_max variants: nothing times out at the default (L109).
line "ls -d $D/lattice3d_finds5_ring2_rec1000_dw+([0-9.])_paired20_ref0.5_d05 2>/dev/null" \
     bound_dwell line_dwell_N5.png \
     "t_20 vs bound dwell  (3D, ring 2, 5% free, 1000 receptors, 5 finds)"
# 5% free only: the bare name is N = 1, the _findsN names are the rest. A plain
# lattice3d*_ring2_... glob would sweep in every free fraction as well.
line "ls -d $D/lattice3d@(|_finds+([0-9]))_ring2_rec1000_dw1000_paired20_ref0.5_d05 2>/dev/null" \
     finds_to_capture line_efficiency.png \
     "t_20 vs capture efficiency  (3D, ring 2, 5% free, $BASE)"

echo
echo "cells complete: $(ls -d $D/*d05*/ 2>/dev/null | while read -r d; do [ -f "$d/paired_ratios.csv" ] && echo x; done | wc -l | tr -d ' ')"
