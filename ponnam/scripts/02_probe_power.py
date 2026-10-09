"""Study 2: power curves on the synthetic probes.

Each probe breaks exactly one modelling assumption while holding everything
else fixed, so a test's power curve on a probe is a direct measurement of how
much that assumption was worth.  Three questions are answered here, and they
are the three the project brief asks:

* *Where does each classical method break?*  A test that is structurally blind
  to a probe produces a power curve pinned at the nominal level regardless of
  sample size or effect size.  That flat line is the answer, and it is a
  stronger statement than "performed poorly": it cannot be fixed with more
  data.
* *Do classical methods beat C2ST at small n?*  The ``mean_sparse`` and
  ``mean_dense`` probes are the fair ground for this -- both are detectable by
  every method, so any difference is power rather than blindness.
* *What does the Bonferroni correction cost?*  ``mean_sparse`` versus
  ``mean_dense`` holds the true effect size constant and moves it from one
  coordinate to all of them.  The gap between a test's two curves is the cost
  of relying on a single extreme marginal.

Output: ``results/probe_power.parquet`` and a printed crossover table.

Usage::

    python scripts/02_probe_power.py
    python scripts/02_probe_power.py --probes correlation mixture --replicates 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shiftbench.experiment import run_probe_power  # noqa: E402

DEFAULT_TESTS = [
    "ks_bonferroni", "ks_simes",
    "t_bonferroni", "z_bonferroni", "ad_bonferroni", "mwu_bonferroni",
    "hotelling", "diag_hotelling", "chen_qin",
    "glrt_gauss",
    "mmd", "mmd_median_sq", "energy",
    "c2st", "c2st_knn",
]


def crossover_table(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Smallest ``n`` at which each test first reaches 80% power.

    This is the single number the comparison turns on.  "Test A is more
    powerful than test B" is vague; "test A needs 50 samples and test B needs
    500 to reach the same detection rate on the same alternative" is not, and
    it is directly the sample-complexity axis the project statement names.
    ``NaN`` means 80% power was never reached on the grid -- which for a
    structurally blind test is the correct and expected entry.
    """
    g = (
        df[df["alpha"] == alpha]
        .groupby(["probe", "eps", "d", "test", "n"])["reject"]
        .mean()
        .reset_index()
        .rename(columns={"reject": "power"})
    )
    rows = []
    for (probe, eps, d, test), sub in g.groupby(["probe", "eps", "d", "test"]):
        sub = sub.sort_values("n")
        hit = sub[sub["power"] >= 0.8]
        rows.append(
            dict(
                probe=probe, eps=eps, d=d, test=test,
                n_for_80pct=int(hit["n"].iloc[0]) if len(hit) else np.nan,
                max_power=float(sub["power"].max()),
            )
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests", nargs="+", default=DEFAULT_TESTS)
    ap.add_argument(
        "--probes", nargs="+",
        default=["mean_sparse", "mean_dense", "variance", "correlation",
                 "mixture", "heavy_tail", "skew"],
    )
    ap.add_argument("--eps", nargs="+", type=float, default=[0.25, 0.5, 0.8])
    ap.add_argument("--dims", nargs="+", type=int, default=[32])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[10, 20, 50, 100, 200, 500])
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Worker processes; -1 uses every core, 1 runs serially.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    cells = (len(args.tests) * len(args.probes) * len(args.eps)
             * len(args.dims) * len(args.sizes))
    print(f"probe power: {cells} cells x {args.replicates} replicates")
    t0 = time.perf_counter()
    df = run_probe_power(
        args.tests, args.probes, args.eps,
        sample_sizes=args.sizes, dims=args.dims,
        n_replicates=args.replicates, seed=args.seed, n_jobs=args.jobs,
    )
    print(f"done in {time.perf_counter() - t0:.0f}s")

    raw = os.path.join(out_dir, "probe_power.parquet")
    df.to_parquet(raw, index=False)

    cross = crossover_table(df)
    cross_path = os.path.join(out_dir, "probe_crossover.csv")
    cross.to_csv(cross_path, index=False)

    # Headline: for the largest effect size, which tests are structurally blind.
    eps_max = max(args.eps)
    print(f"\nMaximum power reached at eps = {eps_max}, d = {args.dims[0]} "
          f"(alpha = 0.05, largest n = {max(args.sizes)}):")
    print("-" * 78)
    view = cross[(cross["eps"] == eps_max) & (cross["d"] == args.dims[0])]
    pivot = view.pivot_table(index="test", columns="probe", values="max_power")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pivot.round(2).to_string())

    print(f"\nwrote {raw}")
    print(f"wrote {cross_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
