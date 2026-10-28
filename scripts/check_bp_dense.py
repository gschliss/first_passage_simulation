"""Berg–Purcell check for free capture at a given receptor count (L72).

Free walkers around one isolated 10 µm cell start 50 nm off the membrane and are lost
at 2 µm; compares the simulated capture probability with Berg–Purcell, using the
current default step rules (including the L71 near-receptor rule).

Usage:
    NUMBA_NUM_THREADS=8 python scripts/check_bp_dense.py --receptors 100000 --n 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from fpsim.config import Stepping
from fpsim.receptors import fibonacci_sphere, make_grid
from fpsim.validation import bp_capture_probability, free_isolated, perfect_capture_probability


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--receptors", type=int, default=100000)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--a", type=float, default=0.002, help="capture radius (µm)")
    args = p.parse_args()
    R, D, st = 10.0, 1.1, Stepping()
    base = fibonacci_sphere(args.receptors)
    g = make_grid(base)
    t0 = time.time()
    c, t, ns = free_isolated(np.arange(args.n, dtype=np.int64) + 3 * 10**6, R, R + 2.0, R + 0.05, D, 0.01, args.a, base,
                             g.start, g.index, g.n, g.box, False, 0.0, st.distance_fraction, st.dt_cap, 0.0,
                             st.receptor_fraction)
    u = c.mean()
    se = np.sqrt(u * (1 - u) / args.n)
    bp = bp_capture_probability(R, R + 2.0, R + 0.05, args.receptors, args.a)
    print(f"N = {args.receptors} receptors, a = {args.a * 1e3:g} nm: simulated {u:.4f} ± {se:.4f}, Berg–Purcell {bp:.4f}, "
          f"fully absorbing {perfect_capture_probability(R, R + 2.0, R + 0.05):.4f} "
          f"(grid box {g.box * R * 1e3:.0f} nm; {ns.mean():.3g} steps/molecule; {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
