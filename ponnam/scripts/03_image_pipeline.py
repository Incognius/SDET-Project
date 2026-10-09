"""Study 3: the *Failing Loudly* reduce-then-test pipeline on real image data.

Runs the full shift catalogue against the full test inventory on MNIST,
Fashion-MNIST and CIFAR-10, following the paper's protocol: fit the reduction
on the training split, use the validation split as the source sample, and
apply a shift to the test split to obtain the target sample.

Two deviations from the reference implementation are deliberate and are
recorded in the output rather than hidden:

* Source and target samples are matched in size.  The reference code passes the
  full 10 000-sample source to the univariate tests but subsamples the source
  to the target's size for the kernel test, which advantages the univariate
  family at exactly the small ``n`` this project studies.
* The adversarial (FGSM) shift is omitted, since generating it requires a
  trained neural network.  Every other shift in the catalogue is reproduced.

Output: ``results/image_pipeline_<dataset>.parquet``.

Usage::

    python scripts/03_image_pipeline.py --dataset mnist
    python scripts/03_image_pipeline.py --dataset cifar10 --seeds 0 1 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shiftbench.experiment import run_image_pipeline  # noqa: E402
from shiftbench.shifts import SHIFTS  # noqa: E402

DEFAULT_TESTS = [
    "ks_bonferroni", "ks_simes",
    "t_bonferroni", "z_bonferroni",
    "hotelling", "diag_hotelling", "chen_qin",
    "glrt_gauss",
    "mmd", "energy",
    "c2st",
]

# A representative subset spanning every shift family and intensity, for runs
# where the full 28-shift catalogue is too expensive.
QUICK_SHIFTS = [
    "rand",
    "small_gn_shift_1.0", "medium_gn_shift_1.0", "large_gn_shift_1.0",
    "small_img_shift_1.0", "medium_img_shift_1.0", "large_img_shift_1.0",
    "medium_img_shift_0.5", "medium_img_shift_0.1",
    "ko_shift_0.5", "ko_shift_1.0",
    "only_zero_shift+medium_img_shift_1.0",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--tests", nargs="+", default=DEFAULT_TESTS)
    ap.add_argument("--reductors", nargs="+",
                    default=["NoRed", "PCA", "SRP", "BBSDs"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument("--quick", action="store_true",
                    help="Use the representative shift subset, not all 28.")
    ap.add_argument("--shifts", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Worker processes; -1 uses every core, 1 runs serially.")
    args = ap.parse_args()

    if args.shifts:
        shifts = args.shifts
    elif args.quick:
        shifts = QUICK_SHIFTS
    else:
        shifts = list(SHIFTS)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{args.dataset}: {len(shifts)} shifts x {len(args.reductors)} reductors "
          f"x {len(args.tests)} tests x {len(args.sizes)} sizes x {len(args.seeds)} seeds")
    t0 = time.perf_counter()
    df = run_image_pipeline(
        args.dataset, args.tests,
        reductor_names=args.reductors, shift_names=shifts,
        sample_sizes=args.sizes, seeds=args.seeds, n_jobs=args.jobs,
    )
    print(f"done in {time.perf_counter() - t0:.0f}s")

    path = os.path.join(out_dir, f"image_pipeline_{args.dataset}.parquet")
    df.to_parquet(path, index=False)

    # Headline: detection accuracy per (test, reductor), averaged over all
    # genuinely shifted cells.  The paper's Table 1a is the same quantity.
    shifted = df[(df["alpha"] == 0.05) & (df["is_null"] == 0)]
    print("\nDetection rate on shifted data, alpha = 0.05 "
          "(rows = test, columns = reduction):")
    print("-" * 78)
    pivot = shifted.pivot_table(index="test", columns="reductor", values="reject")
    with pd.option_context("display.width", 200):
        print(pivot.round(3).to_string())

    nulls = df[(df["alpha"] == 0.05) & (df["is_null"] == 1)]
    if len(nulls):
        print("\nFalse-alarm rate on the no-shift control "
              "(should be near 0.05):")
        print("-" * 78)
        pn = nulls.pivot_table(index="test", columns="reductor", values="reject")
        with pd.option_context("display.width", 200):
            print(pn.round(3).to_string())

    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
