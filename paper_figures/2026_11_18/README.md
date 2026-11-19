# Paper experiments, revised 2026-11-18

Figures and tables for `paper_instructions/2026_11_18.txt` (decision log L125 onward in
`docs/model_notes.md`). Rebuild with `python scripts/paper2_figures.py` (in a Slurm job). Raw runs
are in `results/paper_2026_11_18/` on Sherlock, not in git. The earlier set, `2026_11_02/`, is kept.

**Common settings unless a folder sweeps them:** 3D HCP lattice, 1000 molecules per walker over 20
shared receptor layouts, 5% of time free, 1 s mean bound dwell, D = 0.5 µm²/s in both states,
**5000 receptors** per target-ring cell (receptors only on the target ring), first passage, 2 nm
capture radius, t_max 100 000 s (500 000 s for dwells of 100 s and longer). Every experiment runs
at target rings 1-4 (cluster extends 2 rings past the target). The comparison walker is free-only
(never binds) at the same D.

| folder | sweep |
|---|---|
| `01_confinement/` | % free: 0.01 (stand-in for 0), 0.1-0.9 in 0.1 steps, 1, 1.5, 2, 2.5, 5, 7.5, 10, 15, 20, 30 ... 90, 100; plus D |
| `02_dwell_time/` | dwell: 0.1 ms (stand-in for 0), 1, 10, 50, 100, 500 ms, 1, 5, 10, 25, 40, 50, 60, 75, 100, 250, 500, 1000 s; plus D |
| `03_receptor_density/` | receptors per cell: 1, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000, 7500, 10 000, 25 000, 50 000, 100 000 |
| `04_capture_efficiency/` | efficiency 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25, 50, 100% = capture on exactly N = 1000 ... 1 receptor contacts (any receptor) |
| `05_fgf_vs_hh/` | the same efficiencies at 75% free (pseudo-FGF) and 1% free, against `04` at 5% free (pseudo-HH) |

**Files (PDF only; every figure has a log-x and a linear-x panel side by side)**

- `t20_<axis>_ring<k>` - t_20 (time by which 20% of all molecules are captured), switching vs
  free-only, 95% CIs over layouts; fraction ever captured underneath. A missing point means fewer
  than 20% were ever captured.
- `rate_<axis>_ring<k>` - the same as an apparent diffusion rate L² / (6 t_20) in µm²/s, with L the
  distance from the source to the target ring (ring k x 20.5 µm centre spacing), comparable across
  rings and with the D plots.
- `arrival_<axis>_ring<k>` - the cumulative arrival curves behind them, log and linear time.
- `ring_t20_<axis>`, `ring_rate_<axis>` - x = target ring, one trace per swept value.
- `deff_<axis>` (01, 02) - effective D from a very long lag, max(5000 s, 20 x dwell), and
  instantaneous D from a 2 s lag; receptor-free 20-ring cluster.
- `05_fgf_vs_hh/fgf_vs_hh_t20`, `fgf_vs_hh_rate` - 75 / 5 / 1% free and free-only, rings 1-4.
- `summary_<folder>.csv` - one row per cell.

Within one experiment every figure of a kind shares its x and y axes across rings and series.
