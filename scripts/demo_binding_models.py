"""Compare membrane-binding models on one free trip (decision aid for L24/L27).

Geometry: a straight gap of width W between two flat membranes (y = 0 and y = W),
unbounded along the membrane (x). A molecule unbinds from y = 0, diffuses freely
(D_FREE, reflecting walls), and rebinds. We record each trip's duration and its
displacement along the membrane, which is what moves molecules toward receptors.

All models use one rule: binding is only possible inside a layer of thickness
``delta`` next to each membrane, at rate ``k_layer``. They differ only in delta:

  rate-based (option 1)   delta = W/2   (layer fills the gap: bind anywhere)
  thin layer (option 3)   delta = 10 nm
  ~contact binding        delta = 1 nm  (approaches binding on touch)

``k_layer = k_on * W / (2 * delta)`` makes the mean trip last 1/k_on in every model,
i.e. 5% free at equilibrium (L23). That holds exactly when unbinding molecules land
at a uniformly random depth inside the layer, which is what this demo does.

Usage:
    python scripts/demo_binding_models.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

D_FREE = 10.0  # µm²/s (L20)
W = 3.7  # µm, roughly a 2D pocket diameter (2 x 1.84 µm)
DWELLS_MS = [5, 71, 1000]  # mean bound dwell 1/k_off (L27)
MODELS = [  # (label, delta in µm, color)
    ("rate-based (layer fills gap)", W / 2, "#7f7f7f"),
    ("thin layer, 10 nm (option 3)", 0.010, "#d9480f"),
    ("~contact, 1 nm", 0.001, "#1f4e9c"),
]


def simulate_trips(delta, k_on, n, rng, n_trace=0):
    """Return trip durations, displacements along the membrane, and a few traces."""
    k_layer = k_on * W / (2 * delta)
    dt_layer = min((delta / 5) ** 2 / (2 * D_FREE), 0.05 / k_layer)

    y = rng.uniform(0, delta, n)  # unbind at a random depth in the layer
    x = np.zeros(n)
    t = np.zeros(n)
    alive = np.ones(n, bool)
    traces = [[(0.0, y[i])] for i in range(n_trace)]

    while alive.any():
        idx = np.flatnonzero(alive)
        yi = y[idx]
        in_layer = (yi <= delta) | (yi >= W - delta)
        # Outside the layer, take steps small enough (~1/4 of the distance) not to jump over it.
        dist = np.minimum(yi - delta, W - delta - yi)
        dt = np.where(in_layer, dt_layer, np.maximum(dt_layer, (np.maximum(dist, 0) / 4) ** 2 / (2 * D_FREE)))

        bind = in_layer & (rng.random(idx.size) < 1 - np.exp(-k_layer * dt))
        sigma = np.sqrt(2 * D_FREE * dt)
        step = ~bind
        x[idx[step]] += sigma[step] * rng.standard_normal(step.sum())
        y_new = yi[step] + sigma[step] * rng.standard_normal(step.sum())
        y_new = np.abs(y_new)  # reflect at y = 0
        y_new = W - np.abs(W - y_new)  # reflect at y = W
        y[idx[step]] = y_new
        t[idx[step]] += dt[step]
        alive[idx[bind]] = False

        for i in range(n_trace):
            if alive[i] or traces[i][-1] != (x[i], y[i]):
                traces[i].append((x[i], y[i]))
    return t, x, [np.array(tr) for tr in traces]


def main():
    rng = np.random.default_rng(0)
    n = 20000
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1.1]})

    results = {}
    for dwell in DWELLS_MS:
        k_on = 19 / (dwell / 1000)
        for label, delta, color in MODELS:
            t, x, traces = simulate_trips(delta, k_on, n, rng, n_trace=6 if dwell == 71 else 0)
            results[(dwell, label)] = (t, x, traces, k_on)
            print(f"dwell {dwell:>4} ms | {label:<30} mean trip {t.mean()*1e3:7.3f} ms "
                  f"(target {1e3/k_on:7.3f}) | median |dx| {np.median(np.abs(x)):.3f} µm | "
                  f"99th pct |dx| {np.percentile(np.abs(x), 99):.3f} µm")

    # Top row: example trips at the middle dwell time
    for ax, (label, delta, color) in zip(axes[0], MODELS):
        _, _, traces, k_on = results[(71, label)]
        ax.axhspan(0, delta, color=color, alpha=0.15, lw=0)
        ax.axhspan(W - delta, W, color=color, alpha=0.15, lw=0)
        for tr in traces:
            ax.plot(tr[:, 0], tr[:, 1], lw=0.6, color=color, alpha=0.8)
            ax.plot(tr[-1, 0], tr[-1, 1], "o", ms=4, color="k")
        ax.axhline(0, color="k", lw=2)
        ax.axhline(W, color="k", lw=2)
        ax.set_ylim(-0.2, W + 0.2)
        ax.set_title(f"{label}\n6 example trips, dwell 71 ms (dot = rebinding)", fontsize=10)
        ax.set_xlabel("x along membrane (µm)")
    axes[0][0].set_ylabel("y across gap (µm)")
    xmax = max(np.abs(tr[:, 0]).max() for lab, *_ in MODELS for tr in results[(71, lab)][2])
    for ax in axes[0]:
        ax.set_xlim(-xmax * 1.1, xmax * 1.1)

    # Bottom row: distribution of displacement along the membrane per trip
    for ax, dwell in zip(axes[1], DWELLS_MS):
        for label, delta, color in MODELS:
            t, x, _, k_on = results[(dwell, label)]
            dx = np.sort(np.abs(x))
            ccdf = 1 - np.arange(dx.size) / dx.size
            ax.loglog(dx, ccdf, color=color, lw=1.8,
                      label=f"{label}\ntrip mean {t.mean()*1e3:.2f} ms, median {np.median(t)*1e3:.2g} ms")
        ax.set_title(f"bound dwell {dwell} ms  (k_on = {19/(dwell/1000):.0f} s⁻¹, target mean trip {dwell/19:.2f} ms)",
                     fontsize=10)
        ax.set_xlabel("|displacement along membrane| per free trip (µm)")
        ax.set_ylim(1 / n, 1.05)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7.5, loc="lower left")
    axes[1][0].set_ylabel("fraction of trips longer than this")

    fig.suptitle(f"One free trip between membranes {W} µm apart (D_free = {D_FREE} µm²/s): "
                 "same average trip, different shape", fontsize=12)
    fig.tight_layout()
    out = PROJECT_ROOT / "results" / "sketches" / "binding_models_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
