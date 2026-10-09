"""Two-sample tests, all sharing the :class:`TwoSampleTest` interface.

The project's subject is the generalised likelihood ratio test: how much a
modelling assumption is worth, when it pays for itself, and how far it can be
wrong before it stops doing so.  The registry below is organised to make that
reading explicit.

* **GLRT variants** (:mod:`shiftbench.tests.glrt`) are the object of study.
  Five models, each a different assumption about the data, each available under
  both an asymptotic Wilks calibration and an exact permutation calibration.
* **Classical comparators** (:mod:`shiftbench.tests.parametric`,
  :mod:`shiftbench.tests.univariate`) are the other members of the classical
  family the GLRT must be placed against -- Hotelling's ``T^2``, Chen-Qin, and
  the per-coordinate univariate tests with their multiplicity corrections.
* **Non-parametric baselines** (:mod:`shiftbench.tests.nonparametric`,
  :mod:`shiftbench.tests.classifier`) are reference points, not the focus.  MMD
  and C2ST are included so that "the classical test needs fewer samples" is a
  measured ratio rather than an assertion.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import (
    COMBINERS,
    TestResult,
    TwoSampleTest,
    bonferroni,
    fisher,
    holm,
    permutation_pvalue,
    simes,
)
from .classifier import C2ST
from .glrt import (
    GLRT,
    DiagonalGaussianGLRT,
    GaussianMeanCovGLRT,
    GaussianMeanGLRT,
    GMMGLRT,
    KnownCovarianceGLRT,
)
from .nonparametric import EnergyTest, MMDTest
from .parametric import ChenQin, DiagonalHotelling, HotellingT2
from .univariate import (
    AndersonDarlingTest,
    KSTest,
    MannWhitneyTest,
    WelchTTest,
    ZTest,
)

__all__ = [
    "TwoSampleTest",
    "TestResult",
    "REGISTRY",
    "GLRT_KEYS",
    "build",
    "permutation_pvalue",
    "bonferroni",
    "holm",
    "simes",
    "fisher",
    "COMBINERS",
    "GLRT",
    "KnownCovarianceGLRT",
    "GaussianMeanGLRT",
    "DiagonalGaussianGLRT",
    "GaussianMeanCovGLRT",
    "GMMGLRT",
    "KSTest",
    "WelchTTest",
    "ZTest",
    "AndersonDarlingTest",
    "MannWhitneyTest",
    "HotellingT2",
    "DiagonalHotelling",
    "ChenQin",
    "MMDTest",
    "EnergyTest",
    "C2ST",
]


#: The GLRT variants, by model case.  Each ``*_asym`` entry is the same
#: statistic as its permutation counterpart, calibrated by Wilks' theorem
#: instead; the pairs exist so the two calibrations can be compared directly.
GLRT_KEYS: Dict[str, Callable[[], TwoSampleTest]] = {
    # Case A: mean shift, covariance known (the multivariate z-test).
    "glrt_a_knowncov": lambda: KnownCovarianceGLRT(),
    "glrt_a_knowncov_asym": lambda: KnownCovarianceGLRT(calibration="asymptotic"),
    # Case B: mean shift, covariance unknown but common (== Hotelling's T^2).
    "glrt_b_mean": lambda: GaussianMeanGLRT(),
    "glrt_b_mean_asym": lambda: GaussianMeanGLRT(calibration="asymptotic"),
    "glrt_b_mean_exactf": lambda: GaussianMeanGLRT(calibration="exact_f"),
    # Case C: diagonal covariance; defined for d >> n.
    "glrt_c_diag": lambda: DiagonalGaussianGLRT(),
    "glrt_c_diag_asym": lambda: DiagonalGaussianGLRT(calibration="asymptotic"),
    # Case D: mean and full covariance both free.
    "glrt_d_meancov": lambda: GaussianMeanCovGLRT(),
    "glrt_d_meancov_asym": lambda: GaussianMeanCovGLRT(calibration="asymptotic"),
    # Case E: Gaussian mixture. Permutation only -- Wilks does not apply.
    "glrt_e_gmm2": lambda: GMMGLRT(n_components=2),
    "glrt_e_gmm3": lambda: GMMGLRT(n_components=3),
}


REGISTRY: Dict[str, Callable[[], TwoSampleTest]] = {
    **GLRT_KEYS,
    # --- classical comparators: per-coordinate tests x multiplicity rule ---
    "ks_bonferroni": lambda: KSTest("bonferroni"),
    "ks_holm": lambda: KSTest("holm"),
    "ks_simes": lambda: KSTest("simes"),
    "ks_fisher": lambda: KSTest("fisher"),
    "t_bonferroni": lambda: WelchTTest("bonferroni"),
    "t_simes": lambda: WelchTTest("simes"),
    "z_bonferroni": lambda: ZTest("bonferroni"),
    "ad_bonferroni": lambda: AndersonDarlingTest("bonferroni"),
    "mwu_bonferroni": lambda: MannWhitneyTest("bonferroni"),
    # --- classical comparators: multivariate ------------------------------
    "hotelling": lambda: HotellingT2(),
    "hotelling_ridge": lambda: HotellingT2(regularization="ridge"),
    "diag_hotelling": lambda: DiagonalHotelling(),
    "chen_qin": lambda: ChenQin(),
    # --- non-parametric baselines (reference points, not the focus) -------
    "mmd": lambda: MMDTest("median_dist"),
    "mmd_median_sq": lambda: MMDTest("median_sq"),
    "energy": lambda: EnergyTest(),
    "c2st": lambda: C2ST("logistic"),
    "c2st_knn": lambda: C2ST("knn"),
}


def build(key: str) -> TwoSampleTest:
    """Instantiate a registered test by name."""
    if key not in REGISTRY:
        raise KeyError(f"unknown test {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]()
