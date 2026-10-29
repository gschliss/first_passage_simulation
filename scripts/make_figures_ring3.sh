#!/usr/bin/env bash
# Ring-3, double-t_max figures (plans ring3_t2x and ring3_t2x_n5dwell, L102-L103).
# Safe to run mid-matrix: cells without paired_ratios.csv are skipped, partial grids
# just have blank cells. Run from the project root:  bash scripts/make_figures_ring3.sh
set -uo pipefail
shopt -s extglob
cd "$(dirname "$0")/.."

PY=.venv/bin/python
D=results/dimension
F=docs/figures/ring3
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

line() {  # line <listing> <x> <out> <title>
  pick "$1"
  printf '%-34s %2d points ' "$(basename "$3" .png)" "${#A[@]}"
  if [ "${#A[@]}" -lt 2 ]; then echo "(skipped)"; return; fi
  $PY scripts/plot_line.py "${A[@]}" --x "$2" --out "$F/$3" --title "$4" > /dev/null 2>&1 \
    && echo "ok" || echo "FAILED"
}

BASE="1 s dwell"
R="ring 3, t_max x2"
echo "== grids =="
grid "ls -d $D/lattice3d@(|_f+([0-9.]))_ring3_rec*_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell free_fraction grid_free_x_density_N1.png \
     "t_20 across fraction free x receptor density  (3D, $R, $BASE, first passage)"
grid "ls -d $D/lattice3d@(|_f+([0-9.]))_finds5_ring3_rec*_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell free_fraction grid_free_x_density_N5.png \
     "t_20 across fraction free x receptor density  (3D, $R, $BASE, 5 finds)"

echo "== lines =="
line "ls -d $D/lattice3d@(|_f+([0-9.]))_ring3_rec1000_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     free_fraction line_free_fraction.png "t_20 vs fraction free  (3D, $R, $BASE, 1000 receptors, first passage)"
line "ls -d $D/lattice3d@(|_f+([0-9.]))_finds5_ring3_rec1000_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     free_fraction line_free_fraction_N5.png "t_20 vs fraction free  (3D, $R, $BASE, 1000 receptors, 5 finds)"
line "ls -d $D/lattice3d_ring3_rec*_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell line_density.png "t_20 vs receptor density  (3D, $R, 5% free, $BASE, first passage)"
line "ls -d $D/lattice3d_finds5_ring3_rec*_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     n_per_cell line_density_N5.png "t_20 vs receptor density  (3D, $R, 5% free, $BASE, 5 finds)"
# The dwell line spans two t_max settings (40 000 s; 200 000 s for 100 s and 600 s dwells).
line "ls -d $D/lattice3d_ring3_rec1000_dw+([0-9.])_tmax@(40000|200000)_paired20_ref0.5_d05 2>/dev/null" \
     bound_dwell line_dwell.png "t_20 vs bound dwell  (3D, $R, 5% free, 1000 receptors, first passage)"
line "ls -d $D/lattice3d_finds5_ring3_rec1000_dw+([0-9.])_tmax@(40000|200000)_paired20_ref0.5_d05 2>/dev/null" \
     bound_dwell line_dwell_N5.png "t_20 vs bound dwell  (3D, $R, 5% free, 1000 receptors, 5 finds)"
line "ls -d $D/lattice3d@(|_finds+([0-9]))_ring3_rec1000_dw1000_tmax40000_paired20_ref0.5_d05 2>/dev/null" \
     finds_to_capture line_efficiency.png "t_20 vs capture efficiency  (3D, $R, 5% free, $BASE, 1000 receptors)"

echo
echo "ring-3 cells complete: $(ls -d $D/*ring3*d05*/ 2>/dev/null | while read -r d; do [ -f "$d/paired_ratios.csv" ] && echo x; done | wc -l | tr -d ' ')"
