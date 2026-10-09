"""Generalised likelihood ratio tests for the two-sample problem.

The generalised likelihood ratio test is the general recipe for turning a
modelling assumption into a hypothesis test.  Given a parametric family
``{f(.; theta) : theta in Theta}``, a null restriction ``Theta_0 subset Theta``,
and data ``Z``, the statistic is

.. math::

    \\Lambda = \\frac{\\sup_{\\theta \\in \\Theta_0} L(\\theta; Z)}
                     {\\sup_{\\theta \\in \\Theta} L(\\theta; Z)},
    \\qquad
    G = -2 \\log \\Lambda .

For the two-sample problem the null is "both samples share one parameter" and
the alternative is "each sample has its own", so

.. math::

    G = 2\\left[\\ell(\\hat\\theta_X; X) + \\ell(\\hat\\theta_Y; Y)
                - \\ell(\\hat\\theta_0; X \\cup Y)\\right],

where ``\\ell`` is the log likelihood and ``\\hat\\theta_0`` is the pooled MLE.
Wilks' theorem gives ``G \\to \\chi^2_r`` under ``H_0``, with ``r`` the
difference in the number of free parameters between the two models -- provided
the regularity conditions hold.

Five models are implemented, in increasing order of generality.  Each is a
different answer to "what do we assume the data looks like", and the point of
the set is that moving along it trades assumption strength against the number
of parameters that must be estimated:

====  =====================================  ==========================  ==========
Case  Model                                  Free parameters under H1    Wilks ``r``
====  =====================================  ==========================  ==========
A     ``N(mu, Sigma)``, ``Sigma`` known      ``mu`` per sample           ``d``
B     ``N(mu, Sigma)``, ``Sigma`` common     ``mu`` per sample           ``d``
C     ``N(mu, diag(sigma^2))``               ``mu, sigma^2`` per sample  ``2d``
D     ``N(mu, Sigma)`` both free             ``mu, Sigma`` per sample    ``d + d(d+1)/2``
E     Gaussian mixture, ``k`` components     all mixture parameters      not valid
====  =====================================  ==========================  ==========

Two facts from this table drive everything downstream.

**Case B is Hotelling's** ``T^2``.  The GLRT for a mean shift under a common
unknown covariance satisfies

.. math::

    \\Lambda^{2/(n+m)} = \\left(1 + \\frac{T^2}{n+m-2}\\right)^{-1},

so ``G = (n+m)\\log(1 + T^2/(n+m-2))`` is a strictly increasing function of
``T^2``.  The two tests are therefore *identical* -- same rejection regions,
same power -- and differ only in which reference distribution is used to set
the threshold.  Hotelling's exact ``F`` calibration is available and is
preferable to the asymptotic ``\\chi^2_d``; this is verified in
:mod:`shiftbench.experiment`'s calibration study rather than assumed.

**Case E has no valid asymptotic null.**  Under ``H_0`` the extra mixture
components are unidentifiable and the null parameter sits on the boundary of
the alternative's parameter space, so Wilks' regularity conditions fail and
``G`` is *not* asymptotically ``\\chi^2``.  Quoting a chi-squared p-value for a
mixture likelihood ratio is a standard and serious error.  Case E is therefore
permutation-calibrated only, and the class refuses to produce an asymptotic
p-value at all.

Calibration
-----------
Every case supports two calibrations:

``"asymptotic"``
    The Wilks ``\\chi^2_r`` reference.  Fast, closed form, and an
    *approximation* whose quality at finite ``n`` is exactly what the project
    brief asks to be checked before it is trusted.

``"permutation"``
    Under ``H_0`` the pooled sample is exchangeable, so the permutation
    distribution of ``G`` is an exact finite-sample null regardless of ``n``,
    ``d``, or whether the model is correctly specified.  This is the default.

The difference between them is not cosmetic.  Measured over a 378-cell grid at
500 replicates, the asymptotic calibration of case D at ``d = 32`` rejects
19.8% of the time at a nominal 5% when ``n = 20``, and 0.0% of the time when
``n = 200``; the permutation calibration of the *same statistic on the same
data* holds 4.5-5.3% throughout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .base import TestResult, TwoSampleTest, permutation_pvalue

__all__ = [
    "GLRT",
    "GaussianMeanGLRT",
    "GaussianMeanCovGLRT",
    "DiagonalGaussianGLRT",
    "GMMGLRT",
    "KnownCovarianceGLRT",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mle_cov(a: np.ndarray) -> np.ndarray:
    """Maximum-likelihood covariance: divisor ``n``, not ``n - 1``.

    The GLRT is built from likelihoods, so every covariance entering it must be
    the MLE.  Substituting the unbiased estimator changes the statistic by a
    factor that does not cancel between numerator and denominator and quietly
    breaks the Wilks calibration.
    """
    c = a - a.mean(axis=0)
    return (c.T @ c) / len(a)


def _shrink(s: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink a covariance toward a scaled identity.

    A sample covariance from ``n`` points has rank at most ``n - 1``, so for
    ``d >= n`` its determinant is exactly zero and the log likelihood is
    ``-inf``.  Shrinking toward ``(tr(S)/d) I`` keeps the estimate positive
    definite while preserving its overall scale.

    This is a deliberate departure from the exact MLE and it has a cost: the
    statistic is no longer the exact likelihood ratio, so the Wilks reference
    distribution is no longer even asymptotically correct.  Under permutation
    calibration that does not matter, because the permutation null is computed
    for whatever statistic is actually used.  Under asymptotic calibration it
    matters a great deal, and is one of the reasons case D's chi-squared
    p-values fail so badly at small ``n``.
    """
    if shrinkage <= 0:
        return s
    d = s.shape[0]
    return (1.0 - shrinkage) * s + shrinkage * (np.trace(s) / d) * np.eye(d)


def _logdet(s: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(s)
    return float(val) if sign > 0 else float("-inf")


class GLRT(TwoSampleTest):
    """Base class for the two-sample generalised likelihood ratio tests.

    Subclasses supply :meth:`_statistic` (the value of ``G``) and
    :attr:`wilks_df` (the Wilks degrees of freedom ``r``, or ``None`` when no
    valid asymptotic null exists).
    """

    wilks_df_valid = True

    def __init__(
        self,
        calibration: str = "permutation",
        n_permutations: int = 500,
        max_dim: Optional[int] = None,
    ):
        if calibration not in ("permutation", "asymptotic"):
            raise ValueError("calibration must be 'permutation' or 'asymptotic'")
        if calibration == "asymptotic" and not self.wilks_df_valid:
            raise ValueError(
                f"{type(self).__name__} has no valid asymptotic null; "
                "Wilks' regularity conditions do not hold for this model. "
                "Use calibration='permutation'."
            )
        self.calibration = calibration
        self.n_permutations = n_permutations
        self.max_dim = max_dim

    def _statistic(self, x: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def _wilks_df(self, d: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def _cost_ok(self, n: int, d: int) -> Optional[str]:
        """Return a reason string if this cell is too expensive, else ``None``."""
        if self.max_dim is not None and d > self.max_dim:
            return f"d={d} exceeds max_dim={self.max_dim}"
        return None

    def _run(self, x, y, rng) -> TestResult:
        d = x.shape[1]
        reason = self._cost_ok(len(x) + len(y), d)
        if reason is not None:
            return TestResult.skip(self.name, reason)

        if self.calibration == "asymptotic":
            g = self._statistic(x, y)
            df = self._wilks_df(d)
            p = float(stats.chi2.sf(g, df))
            return TestResult(p, g, self.name, meta={"wilks_df": float(df)})

        p, g, _ = permutation_pvalue(
            x, y, self._statistic, self.n_permutations, rng
        )
        return TestResult(p, g, self.name, meta={"wilks_df": self._wilks_df(d)})


# ---------------------------------------------------------------------------
# Case A: mean shift, covariance known
# ---------------------------------------------------------------------------


class KnownCovarianceGLRT(GLRT):
    """Case A. ``N(mu, Sigma)`` with ``Sigma`` known and common.

    With ``Sigma`` known the log likelihood is quadratic in ``mu`` and the GLRT
    collapses to a closed form:

    .. math::

        G = \\frac{nm}{n+m}(\\bar x - \\bar y)' \\Sigma^{-1} (\\bar x - \\bar y)
        \\;\\sim\\; \\chi^2_d .

    This is the multivariate z-test, and it is the only case in this module
    whose asymptotic null is also *exact* -- the statistic is an exact
    chi-square under normality for every ``n``, because no parameter is
    estimated other than the two means whose difference is already normal.

    Its weakness is the assumption itself.  ``Sigma`` is essentially never
    known, and substituting an estimate without accounting for the estimation
    error is precisely the error that makes the univariate z-test reject at
    9.3% against a nominal 5% at ``n = 20``.  Case B is the honest version.

    Parameters
    ----------
    sigma
        The known covariance.  If ``None``, it is estimated from the pooled
        sample, which makes the test an approximation rather than exact and is
        provided only to quantify how much that approximation costs.
    """

    def __init__(self, sigma: Optional[np.ndarray] = None, **kw):
        self.sigma = sigma
        super().__init__(**kw)
        self.name = f"GLRT-A-knowncov[{self.calibration}]"

    def _wilks_df(self, d: int) -> float:
        return float(d)

    def _statistic(self, x, y) -> float:
        n, m = len(x), len(y)
        d = x.shape[1]
        diff = x.mean(axis=0) - y.mean(axis=0)
        sigma = self.sigma
        if sigma is None:
            sigma = _shrink(_mle_cov(np.concatenate([x, y], axis=0)), 1e-3)
        try:
            sol = np.linalg.solve(sigma, diff)
        except np.linalg.LinAlgError:
            return 0.0
        return float(n * m / (n + m) * diff @ sol)


# ---------------------------------------------------------------------------
# Case B: mean shift, covariance unknown but common  (== Hotelling's T^2)
# ---------------------------------------------------------------------------


class GaussianMeanGLRT(GLRT):
    """Case B. ``N(mu, Sigma)``, ``Sigma`` unknown but shared by both samples.

    The null restricts the two means to be equal; the covariance is a nuisance
    parameter estimated under both hypotheses.  The likelihood ratio works out
    to a strictly increasing function of Hotelling's ``T^2``:

    .. math::

        G = -2\\log\\Lambda
          = (n+m)\\,\\log\\!\\left(1 + \\frac{T^2}{n+m-2}\\right),
        \\qquad
        T^2 = \\frac{nm}{n+m}(\\bar x - \\bar y)' S_p^{-1} (\\bar x - \\bar y).

    **The GLRT and Hotelling's test are the same test.**  They induce identical
    rejection regions and therefore identical power; only the reference
    distribution used to pick the threshold differs.  Hotelling's is exact
    under normality,

    .. math::

        \\frac{n+m-d-1}{d\\,(n+m-2)}\\,T^2 \\;\\sim\\; F_{d,\\;n+m-d-1},

    whereas Wilks gives only ``G \\to \\chi^2_d``.  When both are available the
    ``F`` calibration should be preferred, and ``calibration="exact_f"``
    selects it.

    The hard constraint is ``n + m - 2 > d``.  ``S_p`` is a sum of ``n+m-2``
    rank-one terms, so below that threshold it is singular and the statistic
    does not exist at any sample size -- not a weak test, an undefined one.
    """

    def __init__(self, calibration: str = "permutation", n_permutations: int = 500,
                 max_dim: Optional[int] = None):
        if calibration == "exact_f":
            # Handled in _run; bypass the base class's validation.
            self.calibration = "exact_f"
            self.n_permutations = n_permutations
            self.max_dim = max_dim
        else:
            super().__init__(calibration, n_permutations, max_dim)
        self.name = f"GLRT-B-hotelling[{self.calibration}]"

    def _wilks_df(self, d: int) -> float:
        return float(d)

    @staticmethod
    def _t2(x: np.ndarray, y: np.ndarray) -> float:
        n, m = len(x), len(y)
        diff = x.mean(axis=0) - y.mean(axis=0)
        sx = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
        sy = np.atleast_2d(np.cov(y, rowvar=False, ddof=1))
        sp = ((n - 1) * sx + (m - 1) * sy) / (n + m - 2)
        try:
            sol = np.linalg.solve(sp, diff)
        except np.linalg.LinAlgError:
            return float("nan")
        return float(n * m / (n + m) * diff @ sol)

    def _statistic(self, x, y) -> float:
        n, m = len(x), len(y)
        t2 = self._t2(x, y)
        if not np.isfinite(t2):
            return 0.0
        return float((n + m) * np.log1p(t2 / (n + m - 2)))

    def _run(self, x, y, rng) -> TestResult:
        n, m = len(x), len(y)
        d = x.shape[1]
        dof_resid = n + m - d - 1

        if dof_resid <= 0:
            # The pooled covariance is singular: the test is undefined here,
            # which is zero power rather than missing data. Recorded as a
            # non-rejection with the reason flagged.
            return TestResult(
                1.0, float("nan"), self.name,
                meta={"singular": 1.0, "d": float(d), "n_total": float(n + m)},
            )

        if self.calibration == "exact_f":
            t2 = self._t2(x, y)
            f = t2 * dof_resid / (d * (n + m - 2))
            p = float(stats.f.sf(f, d, dof_resid))
            return TestResult(p, t2, self.name, meta={"F": f, "singular": 0.0})

        return super()._run(x, y, rng)


# ---------------------------------------------------------------------------
# Case C: diagonal covariance
# ---------------------------------------------------------------------------


class DiagonalGaussianGLRT(GLRT):
    """Case C. ``N(mu, diag(sigma^2))`` with both mean and variances free.

    Modelling the covariance as diagonal discards all cross-coordinate
    dependence but buys two things that matter on this benchmark.  First, the
    statistic separates into a sum over coordinates,

    .. math::

        G = \\sum_{j=1}^{d}\\Big[(n+m)\\log\\hat\\sigma_{0j}^2
            - n\\log\\hat\\sigma_{1j}^2 - m\\log\\hat\\sigma_{2j}^2\\Big],

    costing ``O(nd)`` rather than ``O(d^3)``.  Second, it needs only each
    coordinate's variance to be non-zero, so unlike cases B and D it remains
    **defined for ``d`` far larger than ``n``** -- it runs on raw CIFAR-10
    pixels, where Hotelling's ``T^2`` does not exist.

    It is sensitive to both mean and variance changes, so ``r = 2d``.

    The assumption it breaks on is dependence.  A shift that alters only the
    correlation structure while leaving every marginal distribution unchanged
    is invisible to it, exactly as it is to the per-coordinate univariate
    tests.  The ``correlation`` probe is built to exhibit this.
    """

    def __init__(self, calibration: str = "permutation", n_permutations: int = 500,
                 max_dim: Optional[int] = None, floor: float = 1e-12):
        self.floor = floor
        super().__init__(calibration, n_permutations, max_dim)
        self.name = f"GLRT-C-diag[{self.calibration}]"

    def _wilks_df(self, d: int) -> float:
        return float(2 * d)

    def _statistic(self, x, y) -> float:
        n, m = len(x), len(y)
        pooled = np.concatenate([x, y], axis=0)
        v0 = np.maximum(pooled.var(axis=0), self.floor)
        v1 = np.maximum(x.var(axis=0), self.floor)
        v2 = np.maximum(y.var(axis=0), self.floor)
        g = (n + m) * np.log(v0) - n * np.log(v1) - m * np.log(v2)
        return float(np.sum(g))


# ---------------------------------------------------------------------------
# Case D: mean and covariance both unrestricted
# ---------------------------------------------------------------------------


class GaussianMeanCovGLRT(GLRT):
    """Case D. ``N(mu, Sigma)`` with mean and full covariance both free.

    The most general Gaussian case:

    .. math::

        G = (n+m)\\log|\\hat\\Sigma_0| - n\\log|\\hat\\Sigma_1|
            - m\\log|\\hat\\Sigma_2|,

    with ``\\hat\\Sigma_0`` the pooled MLE about the grand mean -- which absorbs
    any mean difference as additional spread, so the single statistic responds
    to both mean and covariance shifts.  Wilks gives
    ``r = d + d(d+1)/2``.

    Two practical facts dominate its behaviour.

    **It needs shrinkage, and shrinkage costs the asymptotic null.**  Each
    per-sample covariance MLE has rank at most ``n - 1``, so for ``d`` anywhere
    near ``n`` the log-determinants diverge.  The ``shrinkage`` parameter keeps
    them finite, but the shrunk statistic is no longer the exact likelihood
    ratio and Wilks no longer applies even asymptotically.  Measured at
    ``d = 32``: the chi-squared calibration rejects 19.8% of the time at a
    nominal 5% when ``n = 20``, and 0.0% when ``n = 200``.  It does not merely
    drift -- it crosses the nominal level from above to below as ``n`` grows,
    so no single correction factor repairs it.  Permutation calibration of the
    identical statistic holds 4.2-6.2% across the whole grid.

    **It is expensive.**  Each permutation needs three ``d x d``
    log-determinants, so the permutation test costs ``O(B d^3)``; at
    ``d = 784`` with ``B = 500`` that is of order ``10^12`` floating-point
    operations.  ``max_dim`` declines such cells explicitly rather than
    returning a number nobody waited for.
    """

    def __init__(self, shrinkage: float = 0.1, calibration: str = "permutation",
                 n_permutations: int = 500, max_dim: Optional[int] = 256):
        self.shrinkage = shrinkage
        super().__init__(calibration, n_permutations, max_dim)
        self.name = f"GLRT-D-meancov[{self.calibration}]"

    def _wilks_df(self, d: int) -> float:
        return float(d + d * (d + 1) / 2)

    def _statistic(self, x, y) -> float:
        n, m = len(x), len(y)
        pooled = np.concatenate([x, y], axis=0)
        ld0 = _logdet(_shrink(_mle_cov(pooled), self.shrinkage))
        ld1 = _logdet(_shrink(_mle_cov(x), self.shrinkage))
        ld2 = _logdet(_shrink(_mle_cov(y), self.shrinkage))
        g = (n + m) * ld0 - n * ld1 - m * ld2
        return float(g) if np.isfinite(g) else 0.0


# ---------------------------------------------------------------------------
# Case E: Gaussian mixture
# ---------------------------------------------------------------------------


class GMMGLRT(GLRT):
    """Case E. Gaussian mixture with ``k`` components.

    The natural relaxation of case D when the data is multi-modal, which image
    data with several classes certainly is.  ``G = 2[\\ell_1 + \\ell_2 -
    \\ell_0]`` with each log likelihood maximised by EM.

    **There is no valid asymptotic null here.**  Under ``H_0`` the two samples
    share a mixture, so the alternative's extra components are unidentifiable
    and the null parameter lies on the boundary of the alternative's parameter
    space.  Wilks' regularity conditions fail and ``G`` is not asymptotically
    ``\\chi^2_r`` for any ``r``.  Attempting to construct this class with
    ``calibration="asymptotic"`` raises, rather than silently producing a
    wrong p-value.

    What it buys: case D models each sample as one ellipsoid, so two
    distributions agreeing in mean and covariance are indistinguishable to it
    however different their shapes.  A two-component mixture matched to
    ``N(0, I)`` in both moments is exactly such a pair, and recovering it is
    the specific claim this case exists to test.

    What it costs: three EM fits per permutation.  The defaults
    (``B = 199``, ``max_iter = 20``, loose tolerance) keep a single call to a
    few seconds; it is run on targeted studies rather than the full grid.
    """

    wilks_df_valid = False

    def __init__(self, n_components: int = 2, n_permutations: int = 199,
                 covariance_type: str = "diag", reg_covar: float = 1e-4,
                 max_iter: int = 20, tol: float = 1e-2,
                 max_dim: Optional[int] = 256):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        super().__init__("permutation", n_permutations, max_dim)
        self.name = f"GLRT-E-gmm(k={n_components},{covariance_type})[permutation]"

    def _wilks_df(self, d: int) -> float:
        return float("nan")  # not defined: see the class docstring

    def _fit_ll(self, a: np.ndarray, seed: int) -> float:
        import warnings

        from sklearn.mixture import GaussianMixture

        k = min(self.n_components, max(1, len(a) // 2))
        gm = GaussianMixture(
            n_components=k, covariance_type=self.covariance_type,
            reg_covar=self.reg_covar, max_iter=self.max_iter, tol=self.tol,
            n_init=1, init_params="k-means++", random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.fit(a)
            return float(gm.score(a) * len(a))

    def _statistic(self, x, y) -> float:
        pooled = np.concatenate([x, y], axis=0)
        return 2.0 * (self._fit_ll(x, 0) + self._fit_ll(y, 0) - self._fit_ll(pooled, 0))
