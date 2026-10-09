"""Study 4: wall-clock cost of each test as a function of ``n`` and ``d``.

The project statement names latency alongside sample complexity as an axis of
comparison, and it is the axis on which the classical tests win most decisively
-- Hotelling's ``T^2`` is a closed-form expression evaluated once, while a
permutation-calibrated kernel test evaluates its statistic 501 times on a
quadratic kernel matrix.  Reporting power without reporting this cost would
flatter the expensive methods.

Measurement hygiene
-------------------
This machine has a hybrid CPU (performance and efficiency cores).  A process
migrated from a P-core to an E-core mid-measurement reports a two- to threefold
slowdown that has nothing to do with the code, so the benchmark pins itself to
a single performance core and takes the *minimum* over repeated passes rather
than the mean.  The minimum is the right statistic for a latency benchmark:
interference can only ever make a measurement slower, so the smallest
observation is the closest estimate of the true cost.

Output: ``results/latency.csv``.

Usage::

    python scripts/04_latency.py
    python scripts/04_latency.py --passes 5 --sizes 50 200 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shiftbench.synthetic import sample_probe  # noqa: E402
from shiftbench.tests import build  # noqa: E402

DEFAULT_TESTS = [
    "ks_bonferroni", "t_bonferroni", "z_bonferroni", "ad_bonferroni",
    "hotelling", "diag_hotelling", "chen_qin",
    "glrt_gauss", "glrt_gmm2",
    "mmd", "energy",
    "c2st",
]


def pin_to_core(core: int = 2) -> bool:
    """Pin this process to one CPU, returning whether it succeeded.

    Core 2 is chosen rather than 0 because the OS scheduler and interrupt
    handling concentrate on the first cores; on this machine cores 0-11 are the
    performance cores.
    """
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})
            return True
        import ctypes

        # Windows: SetProcessAffinityMask with a one-bit mask.
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        return bool(ctypes.windll.kernel32.SetProcessAffinityMask(handle, 1 << core))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests", nargs="+", default=DEFAULT_TESTS)
    ap.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 200, 500, 1000])
    ap.add_argument("--dims", nargs="+", type=int, default=[32, 784])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--core", type=int, default=2)
    args = ap.parse_args()

    pinned = pin_to_core(args.core)
    print(f"CPU affinity pinned to core {args.core}: {pinned}")
    if not pinned:
        print("  (unpinned - treat absolute timings as indicative, "
              "relative ordering is still meaningful)")

    rng = np.random.default_rng(0)
    rows = []
    for d in args.dims:
        for n in args.sizes:
            x, y = sample_probe("mean_sparse", n, n, d, 0.5, rng)
            for key in args.tests:
                best = float("inf")
                for _ in range(args.passes):
                    t0 = time.perf_counter()
                    build(key)(x, y, np.random.default_rng(0))
                    best = min(best, time.perf_counter() - t0)
                rows.append(dict(test=key, n=n, d=d, seconds=best))
                print(f"  d={d:5d} n={n:5d} {key:20s} {best * 1000:9.2f} ms", flush=True)

    df = pd.DataFrame(rows)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "latency.csv")
    df.to_csv(path, index=False)

    print("\nMilliseconds per test (rows = test, columns = n), d = "
          f"{args.dims[0]}:")
    print("-" * 78)
    pivot = (df[df["d"] == args.dims[0]]
             .pivot_table(index="test", columns="n", values="seconds") * 1000)
    with pd.option_context("display.width", 200):
        print(pivot.round(2).to_string())

    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
