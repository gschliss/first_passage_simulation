# Paper experiments, 2026-11-02

Figures and tables for the five experiments in `paper_instructions/2026_11_02.txt`
(decision log L121 onward in `docs/model_notes.md`). Rebuild with
`bash scripts/make_figures_paper.sh`. Raw runs are in `results/paper_2026_11_02/` on Sherlock
(not in git).

**Common settings unless a folder sweeps them:** 3D HCP lattice, 1000 molecules per walker over
20 shared receptor layouts, 5% of time free, 1 s mean bound dwell, D = 0.5 µm²/s in both states,
1000 receptors per target-ring cell (receptors only on the target ring), first passage (N = 1),
2 nm capture radius, t_max 100 000 s (500 000 s for the 278 s and 1000 s dwells). Every axis
has 10 log-spaced steps and every experiment is run at target rings 1, 2, 3 and 4 (cluster
extends 2 rings past the target). The comparison walker is free-only (never binds) at the
same D = 0.5.

| folder | sweep |
|---|---|
| `01_confinement/` | fraction of time free 0.1% -> 100% (100% = no binding); D_eff with no receptors |
| `02_dwell_time/` | mean bound dwell 10 ms -> 1000 s; D_eff with no receptors |
| `03_receptor_density/` | receptors per target cell 10 -> 100 000 |
| `04_capture_efficiency/` | capture efficiency 0.1% -> 100%, as capture on the N-th receptor find, N = 1000 -> 1 |
| `05_fgf_vs_hh/` | the same efficiency sweep at 50% free (pseudo-FGF), against `04` at 5% free (pseudo-HH) |

**Files in each folder**

- `t20_<axis>_ring<k>` - t_20 (time by which 20% of *all* molecules are captured) for the
  switching and free-only walkers, with bootstrap 95% CIs over layouts; the lower panel is
  the fraction ever captured. t_20 is undefined (point missing) where under 20% are captured.
- `arrival_<axis>_ring<k>` - the cumulative arrival curves those t_20 values are read from.
- `deff_<axis>` (01, 02) - effective diffusion D_eff from the time-averaged MSD at a long lag
  of max(2500 s, 10 x dwell), receptor-free 12-ring cluster.
- `summary_<folder>.csv` - one row per cell: axis values, ring, t_20s, ratio and CI, capture fractions.
- `05_fgf_vs_hh/fgf_vs_hh_overlay` - high- vs low-3D walker and free-only, all four rings.

Every figure is written as PNG and as editable PDF.
