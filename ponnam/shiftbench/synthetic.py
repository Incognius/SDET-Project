"""Synthetic probe suite: controlled alternatives with known ground truth.

The image shifts of *Failing Loudly* are realistic but opaque -- when a test
fails on ``large_img_shift`` it is not obvious which of its assumptions was
violated.  The probes in this module are the opposite: each one is a pair of
distributions constructed so that exactly one assumption is broken, with
everything else held fixed.  They are the instrument for the comparison the
project is actually about.

Every probe is parameterised by an *effect size* ``eps`` and a dimension ``d``,
so that each one yields a power curve rather than a single number.

The table below is the claim being tested.  "Blind" means the test cannot
detect the alternative at any sample size, because the quantity it looks at is
identical under ``p`` and ``q`` by construction -- a structural statement, not
an empirical one.

=========================  ======================================================
Probe                      Expected to be blind
=========================  ======================================================
``mean_sparse``            nothing (favourable to everything)
``mean_dense``             nothing, but Bonferroni-corrected marginals lose badly
``variance``               every mean-only test: t, z, Hotelling, Chen-Qin
``correlation``            every *marginal* test (KS, t, z, AD, MWU) and every
                           mean-only test; marginals are identical by
                           construction
``mixture``                Gaussian GLRT, and every mean-only test; first two
                           moments match exactly
``heavy_tail``             mean-only tests; normality-dependent calibration
                           degrades
``skew``                   mean-only and variance-only tests
=========================  ======================================================

``correlation`` and ``mixture`` are the two that matter most.  The first is the
case no amount of multiple-testing correction can rescue, because the evidence
simply is not present in any one-dimensional marginal.  The second is the case
that separates "parametric" from "Gaussian": a mixture GLRT should recover what
a single-Gaussian GLRT cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

__all__ = ["Probe", "PROBES", "build_probe", "sample_probe"]


@dataclass(frozen=True)
class Probe:
    """A parameterised pair of distributions ``(p, q)``.

    ``sampler`` maps ``(n_p, n_q, d, eps, rng)`` to ``(X, Y)``.  When ``eps``
    is zero the two samples are drawn from the same distribution, which makes
    every probe double as a null-calibration generator.
    """

    name: str
    sampler: Callable[[int, int, int, float, np.random.Generator], Tuple[np.ndarray, np.ndarray]]
    description: str
    blind_to: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Probe samplers
# ---------------------------------------------------------------------------


def _mean_sparse(n_p, n_q, d, eps, rng):
    """Mean shift concentrated in a single coordinate.

    The best case for a Bonferroni-corrected marginal test: all the evidence
    sits in one dimension, so taking the minimum p-value loses nothing except
    the factor-of-``K`` correction.
    """
    x = rng.standard_normal((n_p, d))
    y = rng.standard_normal((n_q, d))
    y[:, 0] += eps
    return x, y


def _mean_dense(n_p, n_q, d, eps, rng):
    """Mean shift of the same total norm spread evenly over all coordinates.

    Each coordinate moves by ``eps / sqrt(d)``, so the Mahalanobis distance
    between the two means is identical to :func:`_mean_sparse`.  A test that
    aggregates across dimensions (Hotelling, Chen-Qin) should be unaffected by
    the change from sparse to dense; a test that takes the minimum marginal
    p-value should collapse, since no single coordinate carries detectable
    evidence any more.  This pair isolates the cost of the Bonferroni
    correction from everything else.
    """
    x = rng.standard_normal((n_p, d))
    y = rng.standard_normal((n_q, d))
    y += eps / np.sqrt(d)
    return x, y


def _variance(n_p, n_q, d, eps, rng):
    """Equal means, scaled covariance: ``N(0, I)`` versus ``N(0, (1 + eps)^2 I)``.

    Invisible to anything that compares only means.  Visible to KS, which
    compares whole marginal distributions, and to the Gaussian GLRT, which
    models the covariance explicitly.
    """
    x = rng.standard_normal((n_p, d))
    y = rng.standard_normal((n_q, d)) * (1.0 + eps)
    return x, y


def _correlation(n_p, n_q, d, eps, rng):
    """Identical marginals, different correlation structure.

    ``p = N(0, I)``; ``q = N(0, Sigma)`` with ``Sigma = (1 - eps) I + eps 11'``
    rescaled so every diagonal entry is exactly 1.  Each coordinate of ``q`` is
    therefore exactly ``N(0, 1)``, the same as under ``p``, and *no* univariate
    marginal test can distinguish them no matter how many samples it is given
    or how the per-dimension p-values are combined.  The means are equal too,
    so Hotelling and Chen-Qin are equally blind.

    What can see it: the Gaussian GLRT (it compares covariances), MMD and the
    energy test (they see the joint), and C2ST (the classifier can learn the
    dependence).  This is the cleanest demonstration in the suite that the
    multiple-univariate family has a ceiling that is structural rather than a
    matter of sample size.
    """
    x = rng.standard_normal((n_p, d))
    rho = float(np.clip(eps, 0.0, 0.95))
    # Equicorrelation with unit diagonal: Sigma = (1 - rho) I + rho 11'.
    # Sample via z = sqrt(1 - rho) * e + sqrt(rho) * c with a shared scalar c,
    # which gives exactly this covariance and unit marginal variance.
    e = rng.standard_normal((n_q, d))
    c = rng.standard_normal((n_q, 1))
    y = np.sqrt(1.0 - rho) * e + np.sqrt(rho) * c
    return x, y


def _mixture(n_p, n_q, d, eps, rng):
    """Gaussian versus a two-component mixture matched in mean *and* covariance.

    ``p = N(0, I)``.  ``q = 0.5 N(mu, I - mu mu') + 0.5 N(-mu, I - mu mu')``
    with ``mu = eps * e_1`` and ``eps < 1``.  The mixture has mean ``0`` and
    covariance ``(I - mu mu') + mu mu' = I``, so it agrees with ``p`` on the
    first two moments exactly and differs only from the fourth moment onward
    (the first coordinate becomes platykurtic, then bimodal as ``eps`` grows).

    Consequence: every test whose model is "multivariate normal with some mean
    and covariance" -- t, z, Hotelling, Chen-Qin, and the single-Gaussian GLRT
    -- is blind to this by construction, however much data it is given.  A
    two-component GMM GLRT has exactly the right model class and should
    detect it; KS and MMD should too, through the shape of the first marginal
    and the joint respectively.
    """
    x = rng.standard_normal((n_p, d))
    e = float(np.clip(eps, 0.0, 0.95))
    mu = np.zeros(d)
    mu[0] = e
    # Covariance of each component: I - mu mu'.  Since mu is axis-aligned this
    # is diagonal, with 1 - e^2 in the first coordinate and 1 elsewhere.
    scale = np.ones(d)
    scale[0] = np.sqrt(max(1.0 - e * e, 1e-12))
    sign = rng.choice([-1.0, 1.0], size=(n_q, 1))
    y = rng.standard_normal((n_q, d)) * scale + sign * mu
    return x, y


def _heavy_tail(n_p, n_q, d, eps, rng):
    """Normal versus Student-t rescaled to unit variance.

    Degrees of freedom interpolate from very heavy (``nu = 3``) at ``eps = 1``
    to effectively normal at ``eps = 0``.  Means and variances agree, so this
    is again invisible to mean-only tests; its purpose is to stress the
    *calibration* of tests that assume normality, and to show where the
    tail-sensitive Anderson-Darling statistic beats KS.
    """
    x = rng.standard_normal((n_p, d))
    if eps <= 0:
        return x, rng.standard_normal((n_q, d))
    nu = 3.0 + (1.0 - min(eps, 1.0)) * 97.0  # eps=1 -> nu=3; eps->0 -> nu->100
    t = rng.standard_t(nu, size=(n_q, d))
    y = t / np.sqrt(nu / (nu - 2.0))  # rescale to unit variance
    return x, y


def _skew(n_p, n_q, d, eps, rng):
    """Symmetric versus skewed, with mean and variance held fixed.

    Builds ``q`` from a centred, rescaled exponential mixed with a normal in
    proportion ``eps``.  Defeats mean-only and variance-only tests, and is the
    probe on which the rank-based Mann-Whitney test is expected to behave
    differently from KS, since skewing without moving the median changes the
    distribution but barely moves the ranks.
    """
    x = rng.standard_normal((n_p, d))
    e = float(np.clip(eps, 0.0, 1.0))
    if e == 0:
        return x, rng.standard_normal((n_q, d))
    expo = rng.exponential(1.0, size=(n_q, d)) - 1.0  # mean 0, variance 1
    norm = rng.standard_normal((n_q, d))
    y = np.sqrt(e) * expo + np.sqrt(1.0 - e) * norm  # mean 0, variance 1
    return x, y


PROBES: Dict[str, Probe] = {
    p.name: p
    for p in [
        Probe("mean_sparse", _mean_sparse,
              "Mean shift in one coordinate.",
              ()),
        Probe("mean_dense", _mean_dense,
              "Mean shift of equal norm spread over all coordinates.",
              ()),
        Probe("variance", _variance,
              "Equal means, scaled covariance.",
              ("t", "z", "hotelling", "chen_qin")),
        Probe("correlation", _correlation,
              "Identical marginals, equicorrelated joint.",
              ("ks", "t", "z", "ad", "mwu", "hotelling", "chen_qin")),
        Probe("mixture", _mixture,
              "Two-component mixture matched to N(0, I) in mean and covariance.",
              ("t", "z", "hotelling", "chen_qin", "glrt_gauss")),
        Probe("heavy_tail", _heavy_tail,
              "Student-t rescaled to unit variance.",
              ("t", "z", "hotelling", "chen_qin")),
        Probe("skew", _skew,
              "Skewed with mean and variance held fixed.",
              ("t", "z", "hotelling", "chen_qin")),
    ]
}


def build_probe(name: str) -> Probe:
    if name not in PROBES:
        raise KeyError(f"unknown probe {name!r}; known: {sorted(PROBES)}")
    return PROBES[name]


def sample_probe(
    name: str, n_p: int, n_q: int, d: int, eps: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one ``(X, Y)`` pair from a probe.

    ``eps = 0`` draws both samples from ``p``, which is the null used by the
    calibration harness.
    """
    return build_probe(name).sampler(n_p, n_q, d, eps, rng)
