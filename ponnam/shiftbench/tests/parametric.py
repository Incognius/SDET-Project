"""Classical parametric multivariate two-sample tests.

These are the methods the project brief points at: tests that buy power by
assuming a model for the data, and that are expected to win at small sample
sizes precisely because the model supplies the information the data cannot.
Each class documents the assumption it makes and the construction that breaks
it, because locating those break points is the actual deliverable.

All tests here can be calibrated either by their classical asymptotic null or
by permutation.  Permutation is the default.  The reason is not stylistic: at
the sample sizes this benchmark cares about (``n`` as low as 10) and the
dimensions it runs at, the asymptotic nulls are visibly wrong, and a test whose
nominal level is not its actual level cannot be compared against anything on a
power curve.  The calibration harness measures both so the claim is evidence
rather than assertion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .base import TestResult, TwoSampleTest, permutation_pvalue

__all__ = [
    "HotellingT2",
    "GaussianGLRT",
    "GMMGLRT",
    "ChenQin",
    "DiagonalHotelling",
]


def _pooled_covariance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unbiased pooled covariance assuming a common population covariance."""
    n1, n2 = len(x), len(y)
    sx = np.cov(x, rowvar=False, ddof=1)
    sy = np.cov(y, rowvar=False, ddof=1)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    return ((n1 - 1) * sx + (n2 - 1) * sy) / (n1 + n2 - 2)


class HotellingT2(TwoSampleTest):
    """Hotelling's two-sample ``T^2`` test for equality of mean vectors.

    Statistic::

        T^2 = (n1 n2 / (n1 + n2)) (xbar - ybar)' S_pooled^{-1} (xbar - ybar)

    Under ``H0``, and assuming both samples are multivariate normal with a
    *common* covariance, ``T^2 (n1 + n2 - d - 1) / (d (n1 + n2 - 2))`` follows
    an ``F(d, n1 + n2 - d - 1)`` distribution.

    Assumptions, and how each one fails on this benchmark
    -----------------------------------------------------
    * **Multivariate normality.**  Pixel intensities clipped to ``[0, 1]`` are
      not remotely normal.  In practice this is the mildest of the three
      problems because the statistic depends on the data only through sample
      means, which are asymptotically normal regardless.
    * **Common covariance.**  The image shifts change the covariance structure
      as well as the mean, so the pooled estimate is a mixture of two different
      covariances and the ``F`` calibration is wrong.
    * **``n1 + n2 - 2 > d``.**  This is the one that is fatal rather than
      merely inaccurate.  ``S_pooled`` is a sum of ``n1 + n2 - 2`` rank-one
      terms, so it is singular the moment the dimension reaches that number,
      and the statistic is undefined.  On raw CIFAR-10 (``d = 3072``) this
      means Hotelling's test cannot be computed at *any* sample size the
      benchmark uses; on a ``K = 32`` representation it is fine from ``n = 20``
      upwards.  This single inequality is the strongest argument in the paper
      for dimensionality reduction, and it is a purely algebraic fact rather
      than an empirical finding.

    Parameters
    ----------
    regularization
        ``None`` for the textbook test, which reports ``p = 1`` and flags
        ``singular`` when the covariance cannot be inverted.  ``"ridge"`` adds
        ``shrinkage * trace(S) / d`` to the diagonal, which keeps the statistic
        defined in the ``d >= n`` regime at the cost of invalidating the ``F``
        null -- so ridge mode forces permutation calibration.
    """

    def __init__(
        self,
        regularization: Optional[str] = None,
        shrinkage: float = 0.1,
        calibration: str = "asymptotic",
        n_permutations: int = 500,
    ):
        if regularization not in (None, "ridge"):
            raise ValueError("regularization must be None or 'ridge'")
        if regularization == "ridge" and calibration == "asymptotic":
            # The F distribution is derived for the unregularised statistic.
            calibration = "permutation"
        self.regularization = regularization
        self.shrinkage = shrinkage
        self.calibration = calibration
        self.n_permutations = n_permutations
        suffix = "-ridge" if regularization else ""
        self.name = f"HotellingT2{suffix}[{calibration}]"

    def _t2(self, x: np.ndarray, y: np.ndarray) -> float:
        n1, n2 = len(x), len(y)
        d = x.shape[1]
        diff = x.mean(axis=0) - y.mean(axis=0)
        s = _pooled_covariance(x, y)
        if self.regularization == "ridge":
            s = s + self.shrinkage * (np.trace(s) / d) * np.eye(d)
        try:
            sol = np.linalg.solve(s, diff)
        except np.linalg.LinAlgError:
            return float("nan")
        return float((n1 * n2) / (n1 + n2) * diff @ sol)

    def _run(self, x, y, rng) -> TestResult:
        n1, n2 = len(x), len(y)
        d = x.shape[1]
        dof_resid = n1 + n2 - d - 1

        if self.regularization is None and dof_resid <= 0:
            # Report this loudly rather than silently returning a number: the
            # test genuinely does not exist in this regime.
            return TestResult(
                p_value=1.0,
                statistic=float("nan"),
                name=self.name,
                meta={"singular": 1.0, "d": float(d), "n": float(n1 + n2)},
            )

        t2 = self._t2(x, y)
        if not np.isfinite(t2):
            return TestResult(
                p_value=1.0, statistic=float("nan"), name=self.name,
                meta={"singular": 1.0, "d": float(d)},
            )

        if self.calibration == "asymptotic":
            f_stat = t2 * dof_resid / (d * (n1 + n2 - 2))
            p = float(stats.f.sf(f_stat, d, dof_resid))
            return TestResult(p, t2, self.name, meta={"F": f_stat, "singular": 0.0})

        p, obs, _ = permutation_pvalue(
            x, y, self._t2, self.n_permutations, rng
        )
        return TestResult(p, obs, self.name, meta={"singular": 0.0})


class DiagonalHotelling(TwoSampleTest):
    """Hotelling's ``T^2`` with the covariance constrained to be diagonal.

    Replaces ``S_pooled^{-1}`` with the inverse of its diagonal, which costs
    ``O(d)`` instead of ``O(d^3)`` and -- more importantly -- remains defined
    when ``d >= n``, since it needs only each coordinate's variance to be
    non-zero.  This is the natural bridge between the per-dimension t-tests and
    full Hotelling: it keeps the multivariate aggregation but throws away the
    correlation structure.  Comparing the three isolates how much of Hotelling's
    power comes from aggregation and how much from whitening.

    Calibrated by permutation, since the diagonal statistic has no simple
    exact null when the true covariance is not diagonal.
    """

    def __init__(self, n_permutations: int = 500):
        self.n_permutations = n_permutations
        self.name = "DiagHotelling[permutation]"

    @staticmethod
    def _stat(x: np.ndarray, y: np.ndarray) -> float:
        n1, n2 = len(x), len(y)
        diff = x.mean(axis=0) - y.mean(axis=0)
        var = ((n1 - 1) * x.var(axis=0, ddof=1) + (n2 - 1) * y.var(axis=0, ddof=1)) / (
            n1 + n2 - 2
        )
        var = np.where(var > 0, var, np.inf)  # zero-variance dims contribute nothing
        return float((n1 * n2) / (n1 + n2) * np.sum(diff**2 / var))

    def _run(self, x, y, rng) -> TestResult:
        p, obs, _ = permutation_pvalue(x, y, self._stat, self.n_permutations, rng)
        return TestResult(p, obs, self.name)


class GaussianGLRT(TwoSampleTest):
    """Generalised likelihood ratio test for two multivariate normals.

    Tests ``H0: (mu1, Sigma1) = (mu2, Sigma2)`` against the unrestricted
    alternative, so unlike Hotelling it is sensitive to covariance change as
    well as mean change.  With Gaussian MLEs the log likelihood ratio reduces
    to a function of three log-determinants::

        -2 log(lambda) = (n1 + n2) log|Sigma_pooled| - n1 log|Sigma_1| - n2 log|Sigma_2|

    where ``Sigma_pooled`` is the MLE under ``H0`` (computed about the grand
    mean, so it absorbs any mean difference as extra variance) and
    ``Sigma_1, Sigma_2`` are the per-sample MLEs.

    This is the test the project brief singles out as tying directly back to
    course content, and it is the one whose assumption is easiest to break on
    purpose: it models the data as unimodal and elliptical, so a shift between
    two distributions with identical mean and covariance but different higher
    moments -- a Gaussian versus a two-component mixture matched to the same
    first two moments -- is completely invisible to it.  The synthetic probe
    suite contains that construction.

    Parameters
    ----------
    shrinkage
        Ledoit-Wolf-style shrinkage toward a scaled identity, applied to every
        covariance estimate.  Needed whenever ``d`` approaches ``n``, which is
        the usual case here: the per-sample MLE has rank at most ``n - 1``, so
        its log-determinant is ``-inf`` without it.  The default of 0.1 is
        enough to keep every determinant finite at ``n = 10, d = 32``.
    """

    def __init__(
        self,
        shrinkage: float = 0.1,
        calibration: str = "permutation",
        n_permutations: int = 500,
    ):
        self.shrinkage = shrinkage
        self.calibration = calibration
        self.n_permutations = n_permutations
        self.name = f"GaussianGLRT[{calibration}]"

    def _logdet(self, a: np.ndarray) -> float:
        d = a.shape[0]
        a = (1 - self.shrinkage) * a + self.shrinkage * (np.trace(a) / d) * np.eye(d)
        sign, logdet = np.linalg.slogdet(a)
        return logdet if sign > 0 else -np.inf

    def _stat(self, x: np.ndarray, y: np.ndarray) -> float:
        n1, n2 = len(x), len(y)
        d = x.shape[1]
        pooled = np.concatenate([x, y], axis=0)

        def mle_cov(a):
            c = a - a.mean(axis=0)
            return (c.T @ c) / len(a)

        s0 = mle_cov(pooled)
        s1 = mle_cov(x)
        s2 = mle_cov(y)
        val = (n1 + n2) * self._logdet(s0) - n1 * self._logdet(s1) - n2 * self._logdet(s2)
        return float(val) if np.isfinite(val) else 0.0

    def _run(self, x, y, rng) -> TestResult:
        d = x.shape[1]
        if self.calibration == "asymptotic":
            stat = self._stat(x, y)
            # Free parameters gained under H1: one extra mean vector (d) plus
            # one extra covariance (d (d + 1) / 2).
            df = d + d * (d + 1) / 2
            p = float(stats.chi2.sf(stat, df))
            return TestResult(p, stat, self.name, meta={"df": float(df)})

        p, obs, _ = permutation_pvalue(x, y, self._stat, self.n_permutations, rng)
        return TestResult(p, obs, self.name)


class GMMGLRT(TwoSampleTest):
    """GLRT with a Gaussian-mixture model in place of a single Gaussian.

    Fits a ``n_components``-component GMM to the pooled sample under ``H0`` and
    one GMM to each sample under ``H1``, and uses the log-likelihood difference
    as the statistic.  This is the direct relaxation of :class:`GaussianGLRT`
    suggested by the project brief, and it is the right instrument for showing
    that "parametric" is not synonymous with "Gaussian": a mixture model can
    represent multi-modal class structure that a single Gaussian cannot, and
    should recover the power that the single-Gaussian GLRT loses on exactly the
    multi-modal probe that defeats it.

    The cost is substantial -- three EM fits per permutation -- so the default
    permutation count is lower than for the closed-form statistics, and the
    latency benchmark reports this honestly rather than hiding it.

    Calibration is by permutation only.  The asymptotic chi-squared null for a
    mixture likelihood ratio is not valid: the parameter under ``H0`` lies on
    the boundary of the alternative's parameter space and the regularity
    conditions for Wilks' theorem fail.  This is a textbook trap and quoting a
    chi-squared p-value here would be straightforwardly wrong.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_permutations: int = 100,
        covariance_type: str = "diag",
        reg_covar: float = 1e-4,
        max_iter: int = 50,
    ):
        self.n_components = n_components
        self.n_permutations = n_permutations
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.name = f"GMMGLRT(k={n_components},{covariance_type})[permutation]"

    def _fit_ll(self, a: np.ndarray, seed: int) -> float:
        from sklearn.mixture import GaussianMixture

        k = min(self.n_components, max(1, len(a) // 2))
        gm = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=seed,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.fit(a)
            return float(gm.score(a) * len(a))

    def _stat(self, x: np.ndarray, y: np.ndarray) -> float:
        pooled = np.concatenate([x, y], axis=0)
        ll0 = self._fit_ll(pooled, 0)
        ll1 = self._fit_ll(x, 0) + self._fit_ll(y, 0)
        return 2.0 * (ll1 - ll0)

    def _run(self, x, y, rng) -> TestResult:
        p, obs, _ = permutation_pvalue(x, y, self._stat, self.n_permutations, rng)
        return TestResult(p, obs, self.name)


class ChenQin(TwoSampleTest):
    """Chen and Qin (2010) high-dimensional two-sample mean test.

    Chen, S. X. and Qin, Y.-L., *A two-sample test for high-dimensional data
    with applications to gene-set testing*, Annals of Statistics 38(2), 2010.

    The statistic drops the ``i = j`` terms from the naive squared distance
    between sample means::

        T = sum_{i != j} X_i' X_j / (n1 (n1 - 1))
          + sum_{i != j} Y_i' Y_j / (n2 (n2 - 1))
          - 2 sum_{i, j} X_i' Y_j / (n1 n2)

    Removing the diagonal removes the ``tr(Sigma) / n`` bias that otherwise
    dominates ``||xbar - ybar||^2`` when ``d`` is large, and -- the point of the
    method -- it never requires inverting or even forming a covariance matrix.
    It is therefore defined for any ``d`` relative to ``n``, exactly where
    Hotelling's ``T^2`` ceases to exist.

    What it still assumes: it tests means only, so it inherits Hotelling's
    blindness to pure covariance shifts.  It trades the normality and
    invertibility assumptions for a dimension-free guarantee, not for
    omniscience.

    Calibration is by permutation.  The paper's asymptotic normal calibration
    needs consistent estimates of ``tr(Sigma_i Sigma_j)``, and the estimators
    that make it unbiased are themselves fragile at ``n = 10``; permutation
    sidesteps the issue entirely and gives exact finite-sample level, which is
    what this project is measuring.
    """

    def __init__(self, n_permutations: int = 500):
        self.n_permutations = n_permutations
        self.name = "ChenQin[permutation]"

    @staticmethod
    def _stat(x: np.ndarray, y: np.ndarray) -> float:
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            return 0.0
        sx = x.sum(axis=0)
        sy = y.sum(axis=0)
        # sum_{i != j} X_i'X_j = ||sum_i X_i||^2 - sum_i ||X_i||^2
        cross_x = float(sx @ sx - np.sum(x * x))
        cross_y = float(sy @ sy - np.sum(y * y))
        cross_xy = float(sx @ sy)
        return (
            cross_x / (n1 * (n1 - 1))
            + cross_y / (n2 * (n2 - 1))
            - 2.0 * cross_xy / (n1 * n2)
        )

    def _run(self, x, y, rng) -> TestResult:
        p, obs, _ = permutation_pvalue(x, y, self._stat, self.n_permutations, rng)
        return TestResult(p, obs, self.name)
