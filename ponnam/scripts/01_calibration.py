"""Study 1: empirical Type-I error of every test under a true null.

This runs before anything else that produces a number, because a power
comparison between tests that do not hold the same level is meaningless.  A
test that rejects 18% of the time when nothing has changed will look
impressively powerful on every alternative, and the comparison will be an
artefact.

Protocol: draw both samples from the same distribution, run every test, and
record the rejection rate at each nominal level.  Repeat across three null
distributions -- Gaussian, heavy-tailed and bimodal -- because calibration is a
joint property of the test and the data, not of the test alone.

Output: ``results/calibration.parquet`` (raw, one row per replicate per alpha)
and ``results/calibration_summary.csv`` (rejection rates with binomial
standard errors and a pass/fail verdict).

Usage::

    python scripts/01_calibration.py --replicates 500
    python scripts/01_calibration.py --replicates 100 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shiftbench.experiment import run_calibration  # noqa: E402

FAST_TESTS = [
    "ks_bonferroni", "ks_holm", "ks_simes", "ks_fisher",
    "t_bonferroni", "t_simes", "z_bonferroni",
    "ad_bonferroni", "mwu_bonferroni",
    "hotelling", "hotelling_perm", "hotelling_ridge", "diag_hotelling",
    "chen_qin", "glrt_gauss", "glrt_gauss_asym",
    "mmd", "mmd_median_sq", "energy",
    "c2st", "c2st_knn",
]

# EM inside every permutation makes these two orders of magnitude slower than
# the rest; they are opt-in via --include-slow.
SLOW_TESTS = ["glrt_gmm2", "glrt_gmm3"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicates into rejection rates with a calibration verdict.

    The verdict uses a three-standard-error band around the nominal level.
    With ``R`` replicates the binomial standard error at level ``a`` is
    ``sqrt(a (1 - a) / R)``; a correctly calibrated test falls inside three of
    those roughly 99.7% of the time, so a test outside the band is flagged.
    Conservative (under-rejecting) and anti-conservative (over-rejecting)
    failures are distinguished, because they have opposite consequences: the
    first costs power, the second invalidates the stated false-alarm guarantee.
    """
    g = (
        df.groupby(["test", "probe", "d", "n", "alpha"])["reject"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "type_I_rate", "count": "replicates"})
    )
    a = g["alpha"]
    g["se"] = np.sqrt(a * (1 - a) / g["replicates"])
    g["z"] = (g["type_I_rate"] - a) / g["se"]
    g["verdict"] = np.where(
        g["z"] > 3, "ANTI-CONSERVATIVE",
        np.where(g["z"] < -3, "conservative", "ok"),
    )
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=500)
    ap.add_argument("--include-slow", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="One null distribution, one dimension, two sample sizes.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Worker processes; -1 uses every core, 1 runs serially.")
    args = ap.parse_args()

    tests = FAST_TESTS + (SLOW_TESTS if args.include_slow else [])
    if args.quick:
        probes, dims, sizes = ("mean_sparse",), (32,), (20, 200)
    else:
        probes = ("mean_sparse", "heavy_tail", "mixture")
        dims = (8, 32)
        sizes = (20, 50, 200)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"calibration: {len(tests)} tests x {len(probes)} nulls x {len(dims)} dims "
          f"x {len(sizes)} sizes x {args.replicates} replicates")
    t0 = time.perf_counter()
    df = run_calibration(
        tests, n_replicates=args.replicates, sample_sizes=sizes,
        dims=dims, probes=probes, seed=args.seed, n_jobs=args.jobs,
    )
    print(f"done in {time.perf_counter() - t0:.0f}s")

    raw_path = os.path.join(out_dir, "calibration.parquet")
    df.to_parquet(raw_path, index=False)

    summary = summarise(df)
    sum_path = os.path.join(out_dir, "calibration_summary.csv")
    summary.to_csv(sum_path, index=False)

    # Headline view: the primary alpha, Gaussian null, aggregated over the grid.
    primary = summary[(summary["alpha"] == 0.05)]
    print("\nType-I error at alpha = 0.05, by test (averaged over the grid):")
    print("-" * 72)
    agg = (
        primary.groupby("test")
        .agg(rate=("type_I_rate", "mean"),
             worst=("type_I_rate", "max"),
             n_bad=("verdict", lambda s: int((s == "ANTI-CONSERVATIVE").sum())))
        .sort_values("rate", ascending=False)
    )
    for name, row in agg.iterrows():
        flag = "  <-- ANTI-CONSERVATIVE" if row["n_bad"] else ""
        print(f"  {name:24s} mean={row['rate']:.3f}  worst={row['worst']:.3f}{flag}")

    print(f"\nwrote {raw_path}")
    print(f"wrote {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
