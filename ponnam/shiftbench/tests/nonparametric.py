"""Kernel and distance based multivariate two-sample tests.

These are the non-parametric comparators: they assume nothing about the form of
``p`` and ``q``, and with a characteristic kernel they are consistent against
*every* alternative.  The price is that consistency is an asymptotic statement
and says nothing about power at ``n = 20``, which is where the classical tests
are expected to win.

A note on the bandwidth, because it changes the answer
------------------------------------------------------
The *Failing Loudly* code calls ``torch_two_sample`` with
``alphas = [1 / median_dist]`` where ``median_dist`` is the median *Euclidean*
distance, and that library's kernel is ``exp(-alpha ||x - y||^2)``.  The
effective kernel is therefore ``exp(-||x - y||^2 / median_dist)``, which matches
the paper's own equation but is *not* the standard median heuristic.  The usual
form is ``exp(-||x - y||^2 / (2 sigma^2))`` with ``sigma^2`` the median of the
*squared* distances.  Since the median squared distance is roughly the square
of the median distance, the paper's bandwidth is off by a factor of order
``median_dist`` -- which on normalised CIFAR-10 is around 10, so the kernel is
roughly ten times wider than the standard heuristic would make it.

Both are implemented and selectable, because the difference is large enough to
move power curves and is therefore a finding rather than a detail.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from .base import TestResult, TwoSampleTest, permutation_pvalue_from_matrix

__all__ = ["MMDTest", "EnergyTest"]


class MMDTest(TwoSampleTest):
    """Maximum Mean Discrepancy with a Gaussian kernel and a permutation null.

    Uses the unbiased estimator of the squared MMD, which is the one given as
    equation (3) of *Failing Loudly* and equation (3) of Gretton et al. (2012)::

        MMD^2 = 1/(m^2 - m) sum_{i != j} k(x_i, x_j)
              + 1/(n^2 - n) sum_{i != j} k(y_i, y_j)
              - 2/(mn) sum_{i, j} k(x_i, y_j)

    Being unbiased, it can and does take negative values under ``H0``; that is
    correct behaviour and not a bug.  The null is obtained by permuting the
    index sets of the precomputed kernel matrix, which is exact under ``H0`` and
    costs ``O(B (n + m)^2)`` with no further kernel evaluations.

    Parameters
    ----------
    bandwidth
        ``"median_dist"`` reproduces the *Failing Loudly* setting,
        ``k(x, y) = exp(-||x - y||^2 / median(||x - y||))``.
        ``"median_sq"`` is the standard heuristic,
        ``k(x, y) = exp(-||x - y||^2 / (2 median(||x - y||^2)))``.
    """

    def __init__(
        self,
        bandwidth: str = "median_dist",
        n_permutations: int = 500,
        max_n: int = 2000,
    ):
        if bandwidth not in ("median_dist", "median_sq"):
            raise ValueError("bandwidth must be 'median_dist' or 'median_sq'")
        self.bandwidth = bandwidth
        self.n_permutations = n_permutations
        self.max_n = max_n
        self.name = f"MMD[{bandwidth}]"

    def _run(self, x, y, rng) -> TestResult:
        n, m = len(x), len(y)
        # Kernel methods are quadratic in the pooled sample size; the paper caps
        # them at 1000 target samples for the same reason.
        if n + m > self.max_n:
            keep_x = rng.choice(n, min(n, self.max_n // 2), replace=False)
            keep_y = rng.choice(m, min(m, self.max_n // 2), replace=False)
            x, y = x[keep_x], y[keep_y]
            n, m = len(x), len(y)

        pooled = np.concatenate([x, y], axis=0)
        sq = squareform(pdist(pooled, "sqeuclidean"))

        if self.bandwidth == "median_dist":
            # Median of the cross-sample Euclidean distances, matching the
            # reference implementation's use of cdist(X_tr, X_te).
            med = np.median(cdist(x, y, "euclidean"))
            denom = med if med > 0 else 1.0
        else:
            off = sq[~np.eye(len(sq), dtype=bool)]
            med_sq = np.median(off)
            denom = 2.0 * med_sq if med_sq > 0 else 1.0

        k = np.exp(-sq / denom)

        def stat(ix, iy):
            kxx = k[np.ix_(ix, ix)]
            kyy = k[np.ix_(iy, iy)]
            kxy = k[np.ix_(ix, iy)]
            a, b = len(ix), len(iy)
            term_x = (kxx.sum() - np.trace(kxx)) / (a * a - a)
            term_y = (kyy.sum() - np.trace(kyy)) / (b * b - b)
            return term_x + term_y - 2.0 * kxy.mean()

        p, obs, _ = permutation_pvalue_from_matrix(
            k, n, stat, self.n_permutations, rng
        )
        return TestResult(p, obs, self.name, meta={"bandwidth_denom": float(denom)})


class EnergyTest(TwoSampleTest):
    """Energy-distance two-sample test with a permutation null.

    The energy statistic is::

        E = 2/(nm) sum ||x_i - y_j|| - 1/n^2 sum ||x_i - x_j|| - 1/m^2 sum ||y_i - y_j||

    It is the MMD induced by the (conditionally negative definite) distance
    kernel, so it belongs to the same family but has no bandwidth to choose.
    That makes it a useful control: any gap between MMD and the energy test on
    this benchmark is attributable to bandwidth selection rather than to the
    kernel-versus-distance distinction, since the two are otherwise equivalent.
    """

    def __init__(self, n_permutations: int = 500, max_n: int = 2000):
        self.n_permutations = n_permutations
        self.max_n = max_n
        self.name = "Energy"

    def _run(self, x, y, rng) -> TestResult:
        n, m = len(x), len(y)
        if n + m > self.max_n:
            keep_x = rng.choice(n, min(n, self.max_n // 2), replace=False)
            keep_y = rng.choice(m, min(m, self.max_n // 2), replace=False)
            x, y = x[keep_x], y[keep_y]
            n, m = len(x), len(y)

        pooled = np.concatenate([x, y], axis=0)
        dist = squareform(pdist(pooled, "euclidean"))

        def stat(ix, iy):
            dxy = dist[np.ix_(ix, iy)].mean()
            dxx = dist[np.ix_(ix, ix)].mean()
            dyy = dist[np.ix_(iy, iy)].mean()
            return 2.0 * dxy - dxx - dyy

        p, obs, _ = permutation_pvalue_from_matrix(
            dist, n, stat, self.n_permutations, rng
        )
        return TestResult(p, obs, self.name)
