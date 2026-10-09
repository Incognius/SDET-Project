"""Per-dimension univariate tests combined across dimensions.

This is the family the paper calls "multiple univariate testing": run a
one-dimensional test on each of the ``K`` coordinates independently, then
combine the ``K`` p-values into a single decision.  Two choices are orthogonal
and are exposed separately here, because conflating them is the most common way
to misattribute a result:

* the **marginal test** -- what evidence a single coordinate provides;
* the **combiner** -- how evidence is pooled across coordinates while
  controlling the family-wise error rate.

The paper fixes these to KS and Bonferroni respectively.  Separating them lets
us answer which of the two is responsible for an observed loss of power.

A structural limitation worth stating plainly, since it bounds everything in
this module: no combination of marginal tests can detect a shift that leaves
every marginal unchanged.  A rotation of a multivariate Gaussian that preserves
each coordinate's variance is invisible to this entire family while being
trivially detectable by a test that looks at the covariance.  The synthetic
probe suite includes exactly that case.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats

from .base import COMBINERS, TestResult, TwoSampleTest

__all__ = [
    "MarginalTest",
    "KSTest",
    "WelchTTest",
    "ZTest",
    "AndersonDarlingTest",
    "MannWhitneyTest",
]


class MarginalTest(TwoSampleTest):
    """Run ``marginal`` on every column, then pool with ``combiner``.

    Parameters
    ----------
    marginal
        Callable ``(a, b) -> p_value`` for one dimension.
    combiner
        Key into :data:`shiftbench.tests.base.COMBINERS`.
    """

    marginal_name = "marginal"

    def __init__(self, combiner: str = "bonferroni"):
        if combiner not in COMBINERS:
            raise ValueError(f"unknown combiner {combiner!r}")
        self.combiner = combiner
        self._combine = COMBINERS[combiner]
        self.name = f"{self.marginal_name}+{combiner}"

    def _marginal(self, a: np.ndarray, b: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def _run(self, x, y, rng) -> TestResult:
        k = x.shape[1]
        p_vals = np.empty(k)
        for j in range(k):
            p_vals[j] = self._marginal(x[:, j], y[:, j])
        # Degenerate columns (zero variance in both samples) yield NaN from some
        # scipy routines; treat them as carrying no evidence rather than
        # propagating NaN into the combiner.
        p_vals = np.where(np.isfinite(p_vals), p_vals, 1.0)
        p = self._combine(p_vals)
        return TestResult(
            p_value=p,
            statistic=float(np.min(p_vals)),
            name=self.name,
            meta={"k": float(k), "min_p": float(np.min(p_vals))},
        )


class KSTest(MarginalTest):
    """Two-sample Kolmogorov-Smirnov test per dimension.

    The statistic is the supremum distance between the two empirical CDFs,
    ``sup_z |F_p(z) - F_q(z)|``.  It is distribution-free under ``H0`` for
    continuous data, which is its main appeal: the null distribution does not
    depend on the unknown common distribution.

    The assumption that bites in practice is continuity.  Pixel data that has
    been clipped to ``[0, 1]`` -- which is exactly what the large Gaussian-noise
    shift produces -- has atoms at the boundaries, and in the presence of ties
    scipy's asymptotic p-value is conservative.  This is a real effect on this
    benchmark, not a hypothetical one.
    """

    marginal_name = "KS"

    def _marginal(self, a, b):
        return float(stats.ks_2samp(a, b).pvalue)


class WelchTTest(MarginalTest):
    """Welch's unequal-variance t-test per dimension.

    Tests only the first moment, so it is blind to any pure variance or shape
    change; against a mean shift it is close to optimal under normality and
    remains well-calibrated far from normality thanks to the central limit
    theorem.  The interesting comparison against KS is that it concentrates all
    its power on one alternative instead of spreading it over all of them --
    which should win when the shift really is a mean shift and lose otherwise.
    """

    marginal_name = "Welch-t"

    def _marginal(self, a, b):
        if np.std(a) == 0 and np.std(b) == 0:
            return 1.0
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


class ZTest(MarginalTest):
    """Two-sample z-test per dimension, variance treated as known.

    Differs from :class:`WelchTTest` only in using the normal rather than the
    t reference distribution.  Included because it is the textbook starting
    point and because the gap between the two quantifies, empirically, how much
    the "variance is known" assumption costs at the small sample sizes
    (``n = 10, 20``) where this benchmark is most interesting.
    """

    marginal_name = "z"

    def _marginal(self, a, b):
        na, nb = len(a), len(b)
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(va / na + vb / nb)
        if se == 0:
            return 1.0
        z = (np.mean(a) - np.mean(b)) / se
        return float(2.0 * stats.norm.sf(abs(z)))


class AndersonDarlingTest(MarginalTest):
    """Anderson-Darling k-sample test per dimension.

    Weights the CDF discrepancy by ``1 / (F(1 - F))``, giving far more weight to
    the tails than KS does.  Against the clipped Gaussian-noise shift, where
    the entire signal is mass piling up at 0 and 1, this should outperform KS
    noticeably -- a concrete instance of "which test wins depends on where in
    the distribution the shift lives".
    """

    marginal_name = "AD"

    def _marginal(self, a, b):
        try:
            # scipy caps the reported p-value at the edges of its interpolation
            # table and warns; the clipping is harmless for a combiner that only
            # needs ordering near zero.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(stats.anderson_ksamp([a, b]).pvalue)
        except (ValueError, IndexError):
            # Raised when a sample is constant or too small; no evidence.
            return 1.0


class MannWhitneyTest(MarginalTest):
    """Mann-Whitney U (Wilcoxon rank-sum) per dimension.

    A pure location-shift test on ranks: invariant to any monotone
    transformation of the data, and therefore unaffected by the ``/255``
    scaling or by clipping that preserves order.  It is the natural
    non-parametric counterpart to the t-test and isolates how much of KS's
    behaviour comes from being sensitive to the whole distribution rather than
    just location.
    """

    marginal_name = "MWU"

    def _marginal(self, a, b):
        if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
            return 1.0
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
