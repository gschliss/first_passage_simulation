"""Check that this machine can run the simulations, before you spend an allocation (L98).

    python scripts/check_env.py

Verifies the Python version, the dependencies, that Numba actually compiles and runs a
parallel kernel, how many cores the process may really use, and that the results filesystem
supports the hard link `jobs.write_once` needs. Exits non-zero if anything is a FAIL.

Written to run on a bare interpreter: it imports nothing from fpsim until it has reported on
the dependencies, so a broken install produces a readable report rather than a traceback.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 9)
# Package import name -> the minimum pyproject asks for (None = any).
DEPS = {"numpy": "1.23", "numba": "0.59", "scipy": "1.9", "yaml": "6.0",
        "matplotlib": "3.6", "pandas": "1.5", "pyarrow": "14"}

results: list[tuple[str, str, str]] = []


def record(status, name, detail):
    results.append((status, name, detail))


def version_tuple(v):
    out = []
    for part in str(v).split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def check_python():
    v = sys.version_info[:3]
    ok = v >= MIN_PYTHON
    record("PASS" if ok else "FAIL", "python",
           f"{'.'.join(map(str, v))} at {sys.executable}"
           + ("" if ok else f" (need >= {'.'.join(map(str, MIN_PYTHON))})"))


def check_deps():
    for mod, minimum in DEPS.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            record("FAIL", mod, f"not importable: {type(e).__name__}: {e}")
            continue
        have = getattr(m, "__version__", "?")
        if minimum and have != "?" and version_tuple(have) < version_tuple(minimum):
            record("WARN", mod, f"{have} (pyproject asks for >= {minimum})")
        else:
            record("PASS", mod, str(have))


def check_fpsim():
    try:
        import fpsim
        from fpsim import load_config
        cfg = load_config(PROJECT_ROOT / "configs" / "lattice3d.yaml")
        record("PASS", "fpsim", f"imports from {Path(fpsim.__file__).parent}; "
                                f"lattice3d config loads ({cfg.n_sweep_points} sweep points)")
    except Exception as e:
        record("FAIL", "fpsim", f"{type(e).__name__}: {e}  (did you run: pip install -e \".[dev]\"?)")


def check_cores():
    total = os.cpu_count() or 0
    usable = total
    if hasattr(os, "sched_getaffinity"):  # Linux: cgroups/cpuset can cut this well below cpu_count
        usable = len(os.sched_getaffinity(0))
    want = os.environ.get("NUMBA_NUM_THREADS")
    detail = f"{usable} usable of {total} on this host; NUMBA_NUM_THREADS={want or 'unset'}"
    if want and want.isdigit() and int(want) > usable:
        record("WARN", "cores", detail + " -- asking for more threads than cores oversubscribes")
    else:
        record("PASS", "cores", detail)


def check_numba():
    """Compile and run a real parallel kernel. Catches a broken llvmlite or threading layer."""
    try:
        import numba
        import numpy as np
        from numba import njit, prange

        @njit(parallel=True, cache=False)
        def _sum(a):
            t = 0.0
            for i in prange(a.size):
                t += a[i]
            return t

        a = np.arange(1000, dtype=np.float64)
        got = _sum(a)
        if abs(got - a.sum()) > 1e-6:
            record("FAIL", "numba parallel", f"wrong result: {got} != {a.sum()}")
            return
        record("PASS", "numba parallel",
               f"compiled and ran on {numba.get_num_threads()} threads, "
               f"layer {numba.threading_layer()}")
    except Exception as e:
        record("FAIL", "numba parallel", f"{type(e).__name__}: {e}")


def check_results_fs():
    """jobs.write_once uses os.link for atomic config.json creation; some mounts refuse it."""
    d = PROJECT_ROOT / "results"
    try:
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=d) as tmp:
            src, dst = Path(tmp, "a"), Path(tmp, "b")
            src.write_text("x")
            os.link(src, dst)
        target = os.path.realpath(d)
        free = shutil.disk_usage(d).free / 2**30
        note = f"hard links OK, {free:.0f} GiB free at {target}"
        record("WARN" if free < 5 else "PASS", "results filesystem",
               note + (" -- under 5 GiB" if free < 5 else ""))
    except OSError as e:
        record("FAIL", "results filesystem",
               f"os.link failed at {d}: {e}. jobs.write_once needs it; put results on a "
               f"different filesystem (symlink results/ to scratch)")


def check_tmpdir():
    t = os.environ.get("TMPDIR", "/tmp")
    ok = os.path.isdir(t) and os.access(t, os.W_OK)
    record("PASS" if ok else "WARN", "TMPDIR",
           f"{t}" + ("" if ok else " is not writable; NUMBA_CACHE_DIR needs somewhere local"))


def check_git():
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        record("PASS" if out.returncode == 0 else "WARN", "git commit",
               out.stdout.strip() or "not a git checkout (chunk metadata will say 'unknown')")
    except Exception as e:
        record("WARN", "git commit", f"{type(e).__name__}: {e}")


def main():
    print(f"fpsim environment check\n{platform.platform()}\n{PROJECT_ROOT}\n")
    for fn in (check_python, check_deps, check_fpsim, check_cores, check_numba,
               check_results_fs, check_tmpdir, check_git):
        fn()
    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        print(f"  {status:4}  {name:{width}}  {detail}")
    fails = sum(s == "FAIL" for s, _, _ in results)
    warns = sum(s == "WARN" for s, _, _ in results)
    print(f"\n{len(results) - fails - warns} pass, {warns} warn, {fails} fail")
    if fails:
        print("Fix the failures before submitting: the next step is "
              "`python -m pytest -q`, then the smoke test in docs/cluster.md.")
    else:
        print("Good to go. Next: python -m pytest -q")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
