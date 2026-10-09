"""Common interface and permutation machinery for every two-sample test.

Design rationale
----------------
The four of us are implementing different tests (KS, z/t, Bonferroni-family
corrections, GLRT) and the only way the resulting power curves can be laid side
by side is if every test is a function with the same signature, calibrated the
same way, and evaluated on identical bytes.  That is what this module fixes:

* :class:`TwoSampleTest` is the single interface.  A test consumes two arrays
  and returns a p-value; nothing else about it is visible to the experiment
  runner.
* :func:`permutation_pvalue` provides a null distribution for *any* statistic,
  including ones with no tractable asymptotic null.  This is the project's
  answer to the requirement for rigorous false-alarm guarantees: a permutation
  p-value is exact in finite samples under the exchangeability of the pooled
  sample under ``H0``, with no distributional assumption whatsoever.

The ``+1`` convention
---------------------
The permutation p-value is computed as ``(1 + #{T_perm >= T_obs}) / (1 + B)``
rather than ``#{T_perm >= T_obs} / B``.  The naive version can return exactly
zero, which is not a valid p-value and makes the test anti-conservative; the
corrected version is provably valid at any level for any ``B``.  This detail is
the difference between a test that holds its nominal level and one that does
not, and it is checked empirically by the calibration harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

__all__ = [
    "TestResult",
    "TwoSampleTest",
    "permutation_pvalue",
    "bonferroni",
    "holm",
    "simes",
    "fisher",
    "COMBINERS",
]


@dataclass
class TestResult:
    """Outcome of a single two-sample test.

    ``skipped`` marks a cell the test declined to evaluate, with ``reason``
    saying why.  This is kept distinct from "ran and did not reject": a test
    that cannot be computed at a given ``(n, d)`` has no power there, and
    recording that as a non-rejection would silently average a structural
    impossibility together with genuine failures to detect.  Skipped cells
    carry ``p_value = nan`` and are excluded from rejection rates rather than
    counted as zeros.
    """

    p_value: float
    statistic: float
    name: str
    elapsed_s: float = 0.0
    skipped: bool = False
    reason: str = ""
    meta: Dict[str, float] = field(default_factory=dict)

    def reject(self, alpha: float = 0.05) -> bool:
        if self.skipped or not np.isfinite(self.p_value):
            return False
        return bool(self.p_value < alpha)

    @classmethod
    def skip(cls, name: str, reason: str) -> "TestResult":
        return cls(float("nan"), float("nan"), name, skipped=True, reason=reason)


class TwoSampleTest:
    """Base class: a callable that maps two samples to a :class:`TestResult`.

    Subclasses implement :meth:`_run`.  The base class handles timing, which the
    latency comparison depends on, and basic shape validation.
    """

    name = "base"

    def __call__(
        self, x: np.ndarray, y: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> TestResult:
        if rng is None:
            rng = np.random.default_rng(0)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if y.ndim == 1:
            y = y[:, None]
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"{self.name}: dimension mismatch {x.shape[1]} vs {y.shape[1]}"
            )
        t0 = time.perf_counter()
        res = self._run(x, y, rng)
        res.elapsed_s = time.perf_counter() - t0
        return res

    def _run(
        self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> TestResult:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 500,
    rng: Optional[np.random.Generator] = None,
    larger_is_more_extreme: bool = True,
) -> tuple:
    """Exact-in-finite-samples p-value by permuting the pooled sample.

    Under ``H0: p = q`` the pooled sample is exchangeable, so every assignment
    of ``n`` of the ``n + m`` rows to the first group is equally likely.  Drawing
    ``B`` such assignments gives a Monte-Carlo estimate of the null distribution
    of any statistic at all.

    Returns ``(p_value, observed_statistic, null_draws)``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, m = len(x), len(y)
    pooled = np.concatenate([x, y], axis=0)
    observed = float(statistic(x, y))

    null = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(n + m)
        null[b] = statistic(pooled[perm[:n]], pooled[perm[n:]])

    if larger_is_more_extreme:
        count = int(np.sum(null >= observed))
    else:
        count = int(np.sum(null <= observed))
    p = (1.0 + count) / (1.0 + n_permutations)
    return p, observed, null


def permutation_pvalue_from_matrix(
    kernel_or_distance: np.ndarray,
    n: int,
    statistic_from_matrix: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Permutation p-value that reuses a precomputed pairwise matrix.

    For kernel and distance statistics the pairwise matrix is the entire cost of
    the test: recomputing it inside every permutation would make the test
    ``O(B * (n + m)^2 * d)`` instead of ``O((n + m)^2 * d + B * (n + m)^2)``.
    Permuting row/column indices of the fixed matrix is algebraically identical
    to permuting the data and is what makes a 500-permutation MMD test
    affordable at ``n = 1000``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    total = kernel_or_distance.shape[0]
    idx = np.arange(total)
    observed = float(statistic_from_matrix(idx[:n], idx[n:]))

    null = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(total)
        null[b] = statistic_from_matrix(perm[:n], perm[n:])

    count = int(np.sum(null >= observed))
    p = (1.0 + count) / (1.0 + n_permutations)
    return p, observed, null


# ---------------------------------------------------------------------------
# Multiple-testing combiners
# ---------------------------------------------------------------------------


def bonferroni(p_vals: np.ndarray) -> float:
    """Bonferroni: ``K * min(p)``, clipped to 1.

    Controls the family-wise error rate under *any* dependence structure, which
    is why the paper chooses it: the K dimensions of a PCA or autoencoder
    representation are not independent and no sharper correction is valid
    without assuming something about that dependence.  The price is severe
    conservatism as K grows -- at ``K = 3072`` raw CIFAR-10 pixels the per-test
    level is ``alpha / 3072``, and this conservatism is a large part of why
    dimensionality reduction helps at all.
    """
    p_vals = np.asarray(p_vals, dtype=float)
    return float(min(len(p_vals) * np.min(p_vals), 1.0))


def holm(p_vals: np.ndarray) -> float:
    """Holm-Bonferroni step-down, returned as a global p-value.

    Uniformly more powerful than Bonferroni and valid under the same (absent)
    dependence assumptions, so it strictly dominates it.  For the purpose of a
    single global decision the two differ only through the smallest p-value, so
    this is reported mainly to make that equivalence explicit in the results.
    """
    p = np.sort(np.asarray(p_vals, dtype=float))
    k = len(p)
    adjusted = np.maximum.accumulate((k - np.arange(k)) * p)
    return float(min(adjusted[0] if k else 1.0, 1.0))


def simes(p_vals: np.ndarray) -> float:
    """Simes' global test: ``min_i K * p_(i) / i``.

    Valid under independence and under positive regression dependence (PRDS).
    Pixel intensities of neighbouring pixels are positively correlated, so Simes
    is plausibly valid here and is markedly less conservative than Bonferroni --
    it is included to quantify how much power the conservative choice costs.
    """
    p = np.sort(np.asarray(p_vals, dtype=float))
    k = len(p)
    if k == 0:
        return 1.0
    return float(min(np.min(k * p / np.arange(1, k + 1)), 1.0))


def fisher(p_vals: np.ndarray) -> float:
    """Fisher's combination, ``-2 sum log p ~ chi2_{2K}`` under independence.

    Included as the opposite extreme to Bonferroni: it pools evidence across all
    dimensions rather than relying on the single most extreme one, so it is
    powerful against diffuse shifts and badly mis-calibrated when the
    dimensions are strongly dependent.  The calibration harness measures that
    mis-calibration rather than assuming it.
    """
    from scipy import stats

    p = np.clip(np.asarray(p_vals, dtype=float), 1e-300, 1.0)
    stat = -2.0 * np.sum(np.log(p))
    return float(stats.chi2.sf(stat, df=2 * len(p)))


COMBINERS: Dict[str, Callable[[np.ndarray], float]] = {
    "bonferroni": bonferroni,
    "holm": holm,
    "simes": simes,
    "fisher": fisher,
}
