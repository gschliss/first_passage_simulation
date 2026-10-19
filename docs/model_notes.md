# Model notes

## Question

How does the dimensionality of diffusive search (1D, 2D, 3D, and mixtures —
cf. Adam & Delbrück's "reduction of dimensionality") change the first-passage
time to a target in a biologically inspired geometry?

## Draft model (v0.1, validation only)

- **Agents:** independent overdamped Brownian walkers,
  `x(t+dt) = x(t) + sqrt(2 D dt) * N(0, I_d)`.
- **Boundaries:** reflecting outer wall (radial reflection `r -> 2R - r`).
- **Target:** absorbing; an agent's FPT is the first step at which it is inside the target.
- **Censoring:** agents not absorbed by `t_max` get `NaN`.

### Validation geometry: `SphericalShell`

Absorbing ball radius `a` inside reflecting ball radius `R`, start at radius `r0`.
MFPT solves `D ∇²T = -1`, `T(a)=0`, `T'(R)=0`:

```
T(r0) = 1/(d D) [ R^d ∫_a^r0 s^(1-d) ds - (r0² - a²)/2 ]
```

Implemented in `fpsim.analysis.theory.mfpt_spherical_shell` and checked in
`tests/test_spherical_shell.py`.

## Known limitations

- Fixed-step Euler scheme detects target crossings only at step ends, so FPTs
  are biased slightly high when `sqrt(2 D dt)` is not small compared to `a`.
- Agents are non-interacting.

## Decision log

Decisions agreed together (2026-10-17). The v0.1 code above predates these
and is a draft/validation case only.

| # | Topic | Decision |
|---|-------|----------|
| G1 | Meaning of "dimension" | **Mixed**: ligand lives in 3D but can bind the membrane and diffuse in 2D, then unbind. Question: how 3D/2D partitioning affects time to reach a receptor. |
| G2 | Biological system | **Ligand reaching membrane receptors.** |
| G3 | Side of membrane | **Outside the cell** (extracellular ligand). |
| G4 | Cell shape | **Sphere** (first version; realistic shapes later). |
| G5 | Bare membrane | **Stick, slide, let go**: bind with some probability, diffuse in 2D with `D_2D`, unbind at rate `k_off`. Pure-3D (binding prob. 0) and stick-forever (`k_off = 0`) are limits of the same model. |
| G6 | Outer boundary | **Configurable**: reflecting outer shell *or* escape (absorbing, ligand lost). |
| G7 | Receptors | **N circular patches placed uniformly at random** on the sphere. |
| G8 | Initial condition | **Point source** (all ligands released from one point). |

**Revision (2026-10-17): packed lattice of cells.** Replaces the single-cell picture
(G4, G6 superseded; G7, G8 need re-deciding for the lattice).

| # | Topic | Decision |
|---|-------|----------|
| L1 | Domain | **Explicit packed lattice of cells.** 3D: spheres in a hexagonal close-packed arrangement. 2D: disks in a hexagonal (triangular) lattice. |
| L2 | Units & sizes | **µm.** Cell radius **10 µm**; gap **0.5 µm** at closest approach, so center-to-center spacing is **20.5 µm**. |
| L3 | 3D version | Molecules diffuse in **3D in the gaps** between cells, or **2D on a cell surface**; they switch between the two (bind/unbind as in G5). |
| L4 | 2D version | Molecules diffuse in **2D in the gaps**, or **1D along a cell circumference**; same switching rule. |
| L5 | Use of versions | **Implement both.** Use the 2D version for figures/animations; report statistics from both. |

| L6 | 3D packing | **HCP** (hexagonal layers, ABAB stacking). |
| L7 | Extent | **Finite cluster** of cells (not periodic). |
| L8 | Receptors | Receptor patches on **one target cell** capture molecules. The design must also support **decoy receptors** on other cells (reversible binding with their own `k_on`/`k_off`); for now, decoy `k_on = 0`. |
| L9 | Source | Molecules released from the **surface of a source cell**, some number of cells from the target cell. |
| L10 | Separation | **Sweep 1–4 cells** (hex/ring distance) between source and target. |
| L11 | Direction | Target is a **random cell at that ring distance** from the source (averages over lattice direction). |
| L12 | Placement & size | **Source cell at the cluster center**; cluster has **at least 2 rings beyond the target** (≥ 6 rings / 127 cells in 2D for the 1–4 sweep). |
| L13 | Cluster edge | **Escape**: molecules that leave the cluster are lost. Report **capture probability** alongside FPT (FPT is conditional on capture). |
| L14 | Time limit | Molecules still searching at `t_max` are a **third outcome** (captured / escaped / still searching), each reported as a fraction; choose `t_max` so the last is small. |
| L15 | Escape boundary | **Empty-neighbor (Voronoi) line** (option A): a molecule is lost once it is closer to a lattice site in the missing next ring/shell than to any real cell. In 2D the line runs 0.25 µm past outer cell surfaces and bisects outer pores. No extra length scale; same rule in 2D and 3D. (Briefly chose option B, fixed distance `d_esc`; reversed.) |
| L16 | Release | All molecules start at the **source cell's surface**, at positions **uniform over that surface**, already at the equilibrium split: **95% membrane-bound** on the source membrane and **5% free**, placed just off the source membrane. Source-cell molecules are not special: they switch with the same `k_on`/`k_off` as on any cell. |
| L17 | Target cells | **Every cell in one ring** around the source is a target; initially **ring 3** (18 cells in 2D). Replaces L11's single random target cell. Cluster therefore needs ≥ 5 rings (91 cells in 2D). |
| L18 | Receptors | Per target cell: **3 receptors (2D)** or **4000 (3D)**, **evenly spaced** on the membrane, **immobile**, chosen so receptors cover **~0.1% of the membrane** in both versions (2D: 3 × 20 nm of a 62.8 µm circumference = 0.095%, spacing 20.9 µm; 3D: 4000 × π(10 nm)² of 1257 µm² = 0.100%, spacing ≈ 0.56 µm). Revised from 50 / 2500. Non-target cells have no receptors for now (decoys later, L8). Layout is **randomly rotated per target cell** in both versions (2D: 3 equally spaced angles; 3D: **Fibonacci sphere**). |
| L19 | Capture | Molecules are **points**. Capture when within **10 nm of a receptor's center**. Capture can happen **from the gap or from the membrane**. Captured molecules are **removed and counted**; receptors are **not consumed** (no saturation). |
| L20 | Diffusion | **Free: D = 10 µm²/s** in the gaps. **Membrane-bound: D = 0.5 µm²/s** on the surface (1D along the circumference in 2D, 2D on the sphere in 3D). |
| L21 | Membranes | **Reflective** for free molecules: touching a membrane never binds it. A bound molecule **stays on its own cell's membrane**; changing cells requires unbind → diffuse free → rebind. Replaces G5's stick-on-contact rule. |
| L22 | Switching | Free → bound at rate **k_on**, bound → free at rate **k_off**, both in **s⁻¹**, applied per step as probability `1 − exp(−k·dt)`. On binding, the molecule **inserts at the nearest point of the nearest membrane**. On unbinding, it is placed **just off the membrane at the same spot**. | *(Binding location refined by L28.)*
| L23 | Fixed partition | Equilibrium free fraction fixed at **5%**: `k_off/(k_on + k_off) = 0.05`, so `k_on = 19·k_off`. The sweep is over the switching rate `k_on + k_off` alone. |
| L24 | Sweep limit | ~~Cap k_on below the pocket-mixing rate.~~ **Superseded by L28**: the binding layer removes binding at a distance, so no cap is needed. |
| L25 | Unbinding position | ~~1 nm off the membrane.~~ **Revised (L28):** unbinding molecules land at a **uniformly random depth within the binding layer** (0–10 nm), which keeps the 5% free split exact. |
| L26 | 3D rings | Ring k = cells **k neighbor steps** from the source (12 touching neighbors per cell in HCP; same rule as 2D). 3D ring 3 = **96 cells** at 45.8–61.5 µm; 5-ring cluster = **587 cells**. |
| L27 | Switching sweep | Sweep the **mean bound dwell 1/k_off over 5–1000 ms**, with k_on = 19·k_off (L23). Sparse first pass: **5, 71, 1000 ms** (k_on = 3800, 269, 19 s⁻¹). |
| L28 | Binding layer | **Option 3.** Free molecules can bind only within **δ = 10 nm** of a membrane, at rate `k_layer = k_on · V_free / V_layer` (for a flat gap of width W: `k_on · W / (2δ)`), then insert at the nearest membrane point. Because the layer's share of free space is geometric, `k_layer` is computed, not tuned, and the free fraction stays exactly 5%. Demo (`scripts/demo_binding_models.py`, `results/sketches/binding_models_demo.png`): same mean trip as the rate model, but a heavy tail of rare long trips (99th-percentile displacement up to ~10× farther at 5 ms); 10 nm and 1 nm (≈ contact) layers give nearly identical results. |
| L29 | Layer factor | `V_free / V_layer` per cell of the infinite lattice: **79.2 in 2D** (49.79 µm² free, 0.6286 µm² layer) and **151.3 in 3D** (1903 µm³ free, 12.58 µm³ layer), so `k_layer = 79.2·k_on` (2D) and `151.3·k_on` (3D). Checked in 2D with `scripts/demo_cell_hopping.py`: 300 s runs gave 5.0% and 6.3% time free (target 5%; heavy-tailed trips make this noisy). In those runs (1000 ms dwell, started at a slit), 156 of 571 free trips (27%) ended on a different cell. |
| L30 | Time stepping | **Adaptive time steps.** Each step's dt is the largest value satisfying all of: (a) typical move √(2·D·dt) ≤ **¼ of the distance to the nearest feature** (membrane/binding layer, receptor, escape line); (b) **k·dt ≤ 0.05** for every rate active at the molecule's location (k_layer inside the layer, k_off when bound); (c) a global cap. Unbinding uses the exact per-step probability 1 − e^(−k_off·dt). The two accuracy settings (¼, 0.05) are **fixed across the sweep**, so residual error has the same relative size at every sweep point. |
| L31 | Validation of L30 | Before using results: (1) **halve the accuracy settings** (⅛, 0.025) at both sweep ends (5 ms, 1000 ms); capture probability, MFPT and free fraction must agree within statistical error; (2) compare against a **fixed-small-step reference** on a small problem (single slit / 7-cell patch); (3) report the **time-averaged free fraction** on every run (target 5%) as a drift check. |
| L32 | Language | Simulator core in **Numba** (compiled per-molecule loops, parallel over molecules on CPU). GPU (`numba.cuda`) only if needed later. |
| L33 | Running | **Same entry point locally and on the cluster.** Cluster: **SLURM**, with job arrays over seeds and sweep points; each job writes its own results file, merged afterwards. Local: same command, small molecule counts, for testing. GPU availability not assumed. |
| L34 | Stepping toggle | Time stepping is a **single config switch**: `time_stepping: adaptive` or `fixed`. In `fixed` mode every molecule uses one `dt` everywhere: either set explicitly or, by default, the strictest near-feature limit for that sweep point (resolve 10 nm, k·dt ≤ 0.05 for k_layer). Both modes share the same physics code and write the same output format, so runs can be compared directly; output records which mode and settings were used. |
| L35 | Architecture | Code plan in [architecture.md](architecture.md), approved. Results stored as **Parquet**, one row per molecule. v0.1 draft code to be **removed** (kept in git history). |
| L36 | Receptor layouts | Receptor rotations drawn **once per job** from its seed and shared by its molecules; layouts differ across jobs, so the ensemble averages over layouts. |
| L37 | Exact event timing; accuracy default | (1) Unbinding and binding times are **exact**: a bound dwell is drawn from Exp(k_off) when a molecule binds; a free trip binds when accumulated in-layer exposure (k_layer × time in layer) reaches a draw from Exp(1). Replaces per-step coin flips, which lengthened every dwell by ~k·dt/2 (2.5% at k·dt = 0.05). (2) Default **distance_fraction = 0.125** (was 0.25): at 0.25 the 5 ms point's free fraction was 3% low (4.85 ± 0.04%); at 0.125 and 0.0625 all three sweep points give 5% within error (5 ms: 4.96/4.96%; 71 ms: 5.00/4.96%; 1000 ms: 5.08/4.99%), and cell-change fractions agree across settings. The L31 "halved" check is therefore 0.0625. Full check run (`scripts/check_kernel.py`, then named check_kernel_2d.py; `results/figures/kernel_lattice2d.png`): at 0.125 the free fraction is 4.96 / 5.03 / 5.08% (5 / 71 / 1000 ms) and at 0.0625 4.89 / 4.99 / 4.99%, i.e. within ~2% of 5%, with run-to-run scatter somewhat larger than the quoted errors. Adaptive vs fixed agree within error, but the fixed runs were short (±10–20%). |
| L38 | Free-fraction sweep | Second sweep (in addition to L27): hold the **mean free-trip length 1/k_on = 15 ms** (k_on = 66.7 s⁻¹, from experimental data) and **vary the free fraction f**, with bound dwell 1/k_off = 15 ms × (1 − f)/f (e.g. f = 5% → 285 ms, 10% → 135 ms). With k_on fixed, k_layer (5280 s⁻¹ in 2D) and hence the validated step accuracy are the same at every point. Initial split (L16) uses the sweep point's f. **f = 1, 5, 10, 20, 40%** (bound dwell 1485, 285, 135, 60, 22.5 ms). Requires generalizing sweep points to (free fraction, timescale) pairs. |
| L39 | Linked sweeps | Add **285 ms** to the 5%-free dwell sweep (L27 becomes 5, 71, 285, 1000 ms) so it shares a point with the L38 sweep (f = 5%, 1/k_on = 15 ms); the two runs of that point should agree. |
| L40 | Pilot results (2D) | `scripts/pilot.py` (then pilot_2d.py), 200 molecules per point, t_max 5000 s, 5% free: **all captured, none escaped** at 5 / 71 / 1000 ms. Mean search **1081 / 993 / 1012 s (±50)**; 90th percentile 1.8–2.2 ks; longest 4.8 ks. Captures while bound: 96 / 92 / 73%. Time free 4.92–5.04%. Cost **~3.1–3.6 core-ms per molecule-second ≈ 3.3 core-s per molecule**, so 10⁴ molecules ≈ **9–10 core-hours per sweep point**. Capture-check (small geometry): adaptive, halved and fixed agree within ~2 SE (resolves differences ≳ 8–10%). |
| L41 | Production run settings; sweeps in config | **t_max = 20,000 s** (pilot's longest search 4.8 ks; molecules still searching at t_max are censored, L14). Config now holds **named sweeps**, each fixing two of free_fraction, bound_dwell (1/k_off), free_dwell (1/k_on): `dwell` (5% free; 5, 71, 285, 1000 ms) and `free_fraction` (1/k_on = 15 ms; f = 1, 5, 10, 20, 40%), numbered 0–8. The free fraction moved from `membrane` into the sweeps, and each point sets its own initial split (L16). Shared point check (L39), 200 molecules each: point 2 mean search 1107 ± 59 s, point 5 1115 ± 71 s; captured while bound 80 / 83%; time free 5.02 / 4.99%. Pilot files were renumbered to the new indices (its 1000 ms point is now index 3). |
| L42 | Job workflow | `scripts/run.py` runs one chunk (or all, locally); SLURM array id → (sweep point, chunk) with points varying slowest. Chunk c simulates molecule ids c·n … c·n+n−1 with receptor layout c (L36). A run directory refuses chunks from a different config (hash check) and skips chunks already done, so resubmitting fills gaps. `scripts/merge.py` concatenates chunks, checks config hash / duplicates / missing chunks / code version, and writes `merged.parquet` + `summary.csv`. Default production size: **40 chunks × 250 molecules = 10⁴ per point, 360 tasks, 4 cores each** (`slurm/submit_sweep.sh`, `slurm/README.md`). Tested locally end to end on `mini2d` (9 points × 2 chunks), including one task through the SLURM script. |
| L43 | 3D validation and pilot | **Kernel check** (`check_kernel.py --config configs/lattice3d.yaml`): time free 4.86–5.10% across sweep points and accuracy settings (5 ms: 4.95 / 4.86 / 4.93% at 0.25 / 0.125 / 0.0625; 71 ms: 4.90 / 5.01 / 5.01%; 1000 ms: 5.07 / 5.10 / 5.03%), no trend with step size; cell-change fraction identical across settings. Adaptive vs fixed agree within (wide) error. **Capture check** (small 3D geometry, 50 receptors/cell, 200 molecules): 5 ms captured 192 / 193 / 191 (adaptive / halved / fixed); 1000 ms captured 179 / 182 / 187 and escaped 21 / 18 / 12, a ~1.7 SE difference in escapes to re-check with larger fixed runs. **Pilot** (200 molecules per point, t_max 20 ks): all captured, none escaped; mean search **335 / 364 / 335 s (±16)** at 5 / 71 / 1000 ms, about 3× faster than 2D; 90th percentile 0.6–0.66 ks, longest 1.9 ks; captures while bound 99 / 98 / 93%; cost ~2.7–3.1 core-ms per molecule-second ≈ **1 core-s per molecule** (10⁴ molecules ≈ 3 core-hours per point). |
| L44 | 2D free-fraction sweep (local) | `lattice2d`, L38 points (1/k_on = 15 ms), **500 molecules per point** (2 chunks × 250, `run.py` → `merge.py` → `plot_capture_curves.py`; figure `results/figures/capture_curves_lattice2d_free_fraction.png`). Mean capture time **1699 / 1073 / 769 / 461 / 312 s (±56 / 40 / 25 / 14 / 11)** at f = 1 / 5 / 10 / 20 / 40%; medians 1326 / 851 / 625 / 386 / 250 s. Share captured while bound **96 / 85 / 76 / 65 / 43%**. All captured except 1 escape (1% point). Time free within 0.2% of target at every point. The f = 5% point (1073 ± 40 s) agrees with the earlier shared-point runs (1107 ± 59, 1115 ± 71 s). Wall time on 8 cores: 70 s (1%) to 500 s (40%) per 500 molecules; cost rises with f because free steps near membranes are the expensive ones. |
| L45 | 10%-free switching sweep | Third sweep `dwell_10pct`: **free fraction 10%**, bound dwell **5, 30, 135, 500, 1000 ms** (free trips 0.56, 3.3, 15, 56, 111 ms; k_on ≤ 1800 s⁻¹, inside the validated range). The 135 ms point equals the f = 10% point of the L38 sweep (1/k_on = 15 ms). Sweep points are now numbered 0–13; adding a sweep changes the config hash, so this run uses its own directory `results/runs/lattice2d_dwell10/`. 2D, 500 molecules per point. |
| L46 | 10%-free switching sweep results (2D) | 500 molecules per point (`results/figures/capture_curves_lattice2d_dwell_10pct.png`). Mean capture time **733 / 795 / 807 / 801 / 771 s (±26–29)** at bound dwell 5 / 30 / 135 / 500 / 1000 ms; medians 563–644 s. **Capture speed barely depends on switching rate** over a 200× range (5 ms is ~2 SE faster than the rest), while the share captured while bound falls **88 → 83 → 72 → 65 → 57%**. Same pattern as the 5%-free pilot (L40). Shared point (135 ms, 10% free) 807 ± 28 s vs 769 ± 25 s in the free-fraction sweep (L44), ~1 SE apart. 2 escapes in 2500. Time free within 0.4% of target. 26 min on 8 cores. |

**Research question (2026-10-19):** is switching between dimensions meaningfully different
from plain diffusion with the same average speed? Interpretation of L40/L44/L46: capture
time tracks the free fraction but not the switching rate, likely because transport to ring 3
dominates over the local search for a 10 nm receptor.

| # | Topic | Decision |
|---|-------|----------|
| L47 | Reference walker | "Just diffusing in 2D" = a **free-only walker** (never binds) diffusing in the gaps with **D = the measured D_eff** of the switching point it is compared to; same geometry, receptors and capture rule. Primary comparison: T_switch / T_reference. |
| L48 | Smaller target | Capture radius **2 nm** (was 10 nm), keeping **receptor count fixed** (3 per cell in 2D, 4000 in 3D), so coverage drops 25× (2D 0.0038%, 3D 0.004%). With the existing 10 nm results this gives a two-point capture-time vs target-size slope, an empirical effective dimension (1D ≈ independent of size, 2D ~ log(1/a), 3D ~ 1/a). |
| L49 | Search decomposition | New per-molecule output: **time of first arrival at any target cell** (enters its binding layer or binds to it), splitting each search into transport and local search; plus the number of separate visits to target cells before capture. |
| L50 | D_eff calibration | **Measure** D_eff at each switching point from receptor-free runs (long-time spreading per unit time), instead of the naive f·D_free + (1−f)·D_bound; kernel checks suggest it is about half the naive value in 2D. 3D HCP may be anisotropic (in-layer vs across layers), so report both. |
| L51 | First dimension experiment (2D) | **5% free**; switching points at bound dwell **5, 71, 285, 1000 ms**; each paired with a free-only reference whose **D_free = D_eff / φ**, where φ = (measured D_eff of a free-only walker) / D, since a free walker's long-range spreading is slowed by the gap geometry; reference D_eff is re-measured to confirm the match. Capture runs at **2 nm and 10 nm** targets, **~100 molecules per run**, **6 threads** (2 cores left free). Calibration: 100 molecules × 3000 s, D_eff from time-averaged MSD at lag 500 s (also reported at 50 s). Script: `scripts/dimension_experiment.py`. |
| L52 | First dimension experiment: results (2D) | `results/figures/dimension_experiment_lattice2d.png`, 100 molecules per run. **D_eff** at 5% free is flat in switching rate: 0.52 / 0.51 / 0.49 / 0.50 µm²/s (5 / 71 / 285 / 1000 ms bound; naive average 0.97); φ = 0.409 ± 0.015; references measured 0.47 / 0.52 / 0.46 / 0.47 (~6% low on average, i.e. φ likely slightly overestimated). **Total capture time** switching / reference ≈ 0.91 / 1.04 / 0.82 / 0.86 (±0.1 SE; 2 nm), so no clear difference overall, because **transport to a target cell is ~75–90% of the search** and is the same for both (transport ratio 0.90–1.15). **Local search** (first arrival at a target cell → capture) is **faster for switching**: ratio 0.58 ± 0.18 / 0.67 ± 0.19 / 0.38 ± 0.09 / 0.68 ± 0.19. **Target-size sensitivity of the local search** (paired 2 nm / 10 nm): reference 1.06 / 1.11 / 1.23 / 1.19 (2D-like, grows as the target shrinks); switching 1.06 / 1.05 / 0.89 / 0.94 (≈ 1, 1D-like). Suggestive of a dimension effect in the final approach, but small samples; in this geometry it is diluted in the total by transport. 60 min on 6 threads. |
| L53 | Second dimension experiment (2D) | To make the local search a larger share of the total: **target ring 2** (12 target cells; cluster 4 rings, 61 cells) and **2 receptors per target cell** (was ring 3, 3 receptors). Same 5% free, four switching points, matched free-only references, **2 nm** targets only, ~100 molecules, 6 threads. D_eff calibration and reference D_free reused from L52 (calibration runs have no receptors and their own 12-ring cluster, so they don't depend on targets or receptor count). |
| L54 | Second dimension experiment: results (2D) | `results/figures/dimension_experiment_lattice2d_ring2_rec2.png`, 100 molecules per run, 5 min on 6 threads. Searches are shorter (mean 375–517 s) and the **local search is now 40–59% of the total** (was 10–25%). Switching / free-only: **total** 0.94 / 1.12 / 0.82 / 0.96, **local search** 0.77 / 1.48 / 0.60 / 0.79, transport 0.89–1.20 (5 / 71 / 285 / 1000 ms). The four free-only references are near-replicates (D_free 1.21–1.28) yet their local-search means span 149–307 s, so **run-to-run noise at n = 100 is about ±35%**; no switching/free-only difference is resolved at this size. Across L52 and L54, switching had the faster local search in 7 of 8 comparisons, which is suggestive only. 5–12% of molecules are captured before 100 s (not shown on the 100 s+ curves). Next: ~1000 molecules per run to resolve ~10% differences. |

Derived geometry (from L1–L2):

| Quantity | 2D (disks) | 3D (spheres) |
|---|---|---|
| Fraction occupied by cells | 86.3% | 68.8% |
| Free space for 3D/2D diffusion | 13.7% | 31.2% |
| Largest pore (inscribed radius) | 1.84 µm (triangular pore) | 4.50 µm (octahedral hole); 2.55 µm (tetrahedral) |
| Narrowest passage between pores | 0.5 µm slit between two cells | window of radius 1.84 µm between three cells |

Note: in 2D, pores connect only through 0.5 µm slits; in 3D, pores connect
through much wider windows. The two versions are not just the same geometry in
different dimensions; their bulk transport differs qualitatively.

Animation choices, on hold until the geometry is built: notebook + saved video;
~8 highlighted walkers with fading trails; others as faint dots; absorbed walkers
freeze with a marker; live survival curve S(t).

## Open decisions / next steps

- [ ] Ring distance ≠ Euclidean distance: in ring k, center-to-center distance ranges
      from ≈ k·20.5·√3/2 µm (edge midpoints of the ring) to k·20.5 µm (corners,
      along a row). Report both?
- [ ] Decoy receptor layout (later).
- [ ] Make local search a larger share of total capture time (closer targets, fewer receptors) and/or more molecules, so the L52 local-search effect can show in total first-passage time; refine φ.
- [ ] Sharper adaptive-vs-fixed comparison (L31) with larger fixed-step runs on the cluster.
- [ ] Cross-check δ = 10 nm against δ = 1 nm in the full lattice at one sweep point (the flat-gap demo already agrees).
- [ ] What we compare across conditions (MFPT, capture probability, FPT distribution).
