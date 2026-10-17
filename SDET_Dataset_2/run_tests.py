"""Run three classical two-sample tests on SDET_Dataset_2 (synthetic, d = 10).

Tests (alpha = 0.05, no multiple-testing correction):

    t     per-feature Welch t-test; reject if ANY of the d features has p < alpha
    ks    per-feature two-sample KS test; reject if ANY feature has p < alpha
    glrt  Gaussian GLRT, H0: same mean and variance for every feature;
          -2 log Lambda ~ chi2(2d)

Rejection rate at level k = 0 is the false-alarm rate (Type-I error); at k > 0 it is the power.

Run from this folder:  python run_tests.py   (writes results/rejection_rates.csv, results/glrt_null_stats.npz)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "synthetic"
OUT = HERE / "results"
ALPHA = 0.05
D = 10
NS = [10, 20, 50, 100]
FAMILIES = ["mean_dense", "variance", "heavytail_mean"]
TESTS = ["t", "ks", "glrt"]


def glrt(X, Y):
    """-2 log Lambda for H0: N(mu_j, s_j^2) shared by X and Y, independently per feature."""
    n, m = X.shape[-2], Y.shape[-2]
    Z = np.concatenate([X, Y], axis=-2)
    vx, vy, v0 = X.var(axis=-2), Y.var(axis=-2), Z.var(axis=-2)  # MLE variances (ddof=0)
    return ((n + m) * np.log(v0) - n * np.log(vx) - m * np.log(vy)).sum(axis=-1)


def pvalues(X, Y):
    """X, Y: (R, n, d). Returns dict test -> (R,) p-values, plus the raw GLRT statistic."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    s = glrt(X, Y)
    return {
        "t": stats.ttest_ind(X, Y, axis=1, equal_var=False).pvalue.min(axis=1),
        "ks": stats.ks_2samp(X, Y, axis=1).pvalue.min(axis=1),
        "glrt": stats.chi2.sf(s, 2 * X.shape[-1]),
    }, s


def main():
    OUT.mkdir(exist_ok=True)
    rows, nulls = [], {}
    for fam in FAMILIES:
        z = np.load(DATA / f"{fam}_d{D}.npz")
        for n in NS:
            X = z[f"X__n{n}"]
            for k, lev in enumerate(z["S__levels"]):
                p, s = pvalues(X, z[f"Y__n{n}__m{k}"])
                if k == 0:
                    nulls[f"{fam}_n{n}"] = s
                for t in TESTS:
                    rows.append(dict(family=fam, n=n, k=k, level=float(lev), test=t,
                                     reject_rate=float((p[t] < ALPHA).mean())))
        print(f"  {fam}", flush=True)
    with open(OUT / "rejection_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(OUT / "glrt_null_stats.npz", **nulls)
    print(f"wrote results to {OUT}")


if __name__ == "__main__":
    main()
