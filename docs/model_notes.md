# Model notes

## Question

How does the dimensionality of diffusive search (1D, 2D, 3D, and mixtures —
cf. Adam & Delbrück's "reduction of dimensionality") change the first-passage
time to a target in a biologically inspired geometry?

## Current model (v0.1)

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

## Open decisions / next steps

- [ ] Choose the bio-inspired geometry (e.g. nucleus + DNA, membrane + receptor,
      cytoskeletal network, synapse, organelle).
- [ ] How to compare dimensions fairly (same volume? same D? same target size?).
- [ ] Mixed-dimension search: switching between 3D bulk, 2D surface, 1D track.
- [ ] Interaction with the target (partial absorption / reaction rate).
