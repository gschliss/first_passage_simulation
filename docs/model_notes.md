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
- [ ] Cross-check δ = 10 nm against δ = 1 nm in the full lattice at one sweep point (the flat-gap demo already agrees).
- [ ] What we compare across conditions (MFPT, capture probability, FPT distribution).
