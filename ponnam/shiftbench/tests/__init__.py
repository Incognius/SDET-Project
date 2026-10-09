"""Two-sample tests, all sharing the :class:`TwoSampleTest` interface.

The registry below is the single place where the team's division of labour is
made concrete.  Each entry is a zero-argument factory, so the experiment runner
constructs a fresh test object per cell and nothing leaks between runs.

Ownership (for cross-referencing against each member's own implementation):

================================  ==============================================
Owner                             Registry keys
================================  ==============================================
KS                                ``ks_bonferroni``, ``ks_holm``, ``ks_simes``,
                                  ``ks_fisher``
z / t                             ``z_bonferroni``, ``t_bonferroni``,
                                  ``t_simes``
Bonferroni / corrections          every ``*_holm``, ``*_simes``, ``*_fisher``
                                  pairing, plus ``ad_bonferroni``,
                                  ``mwu_bonferroni``
GLRT                              ``glrt_gauss``, ``glrt_gauss_asym``,
                                  ``glrt_gmm2``, ``glrt_gmm3``
Shared comparators (this branch)  ``hotelling``, ``hotelling_ridge``,
                                  ``diag_hotelling``, ``chen_qin``, ``mmd``,
                                  ``mmd_median_sq``, ``energy``, ``c2st``,
                                  ``c2st_knn``
================================  ==============================================
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
from .nonparametric import EnergyTest, MMDTest
from .parametric import ChenQin, DiagonalHotelling, GaussianGLRT, GMMGLRT, HotellingT2
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
    "build",
    "permutation_pvalue",
    "bonferroni",
    "holm",
    "simes",
    "fisher",
    "COMBINERS",
    "KSTest",
    "WelchTTest",
    "ZTest",
    "AndersonDarlingTest",
    "MannWhitneyTest",
    "HotellingT2",
    "DiagonalHotelling",
    "GaussianGLRT",
    "GMMGLRT",
    "ChenQin",
    "MMDTest",
    "EnergyTest",
    "C2ST",
]


REGISTRY: Dict[str, Callable[[], TwoSampleTest]] = {
    # --- multiple univariate: marginal test x combiner -------------------
    "ks_bonferroni": lambda: KSTest("bonferroni"),
    "ks_holm": lambda: KSTest("holm"),
    "ks_simes": lambda: KSTest("simes"),
    "ks_fisher": lambda: KSTest("fisher"),
    "t_bonferroni": lambda: WelchTTest("bonferroni"),
    "t_simes": lambda: WelchTTest("simes"),
    "z_bonferroni": lambda: ZTest("bonferroni"),
    "ad_bonferroni": lambda: AndersonDarlingTest("bonferroni"),
    "mwu_bonferroni": lambda: MannWhitneyTest("bonferroni"),
    # --- classical multivariate parametric --------------------------------
    "hotelling": lambda: HotellingT2(),
    "hotelling_perm": lambda: HotellingT2(calibration="permutation"),
    "hotelling_ridge": lambda: HotellingT2(regularization="ridge"),
    "diag_hotelling": lambda: DiagonalHotelling(),
    "chen_qin": lambda: ChenQin(),
    "glrt_gauss": lambda: GaussianGLRT(),
    "glrt_gauss_asym": lambda: GaussianGLRT(calibration="asymptotic"),
    "glrt_gmm2": lambda: GMMGLRT(n_components=2),
    "glrt_gmm3": lambda: GMMGLRT(n_components=3),
    # --- kernel / distance -------------------------------------------------
    "mmd": lambda: MMDTest("median_dist"),
    "mmd_median_sq": lambda: MMDTest("median_sq"),
    "energy": lambda: EnergyTest(),
    # --- classifier --------------------------------------------------------
    "c2st": lambda: C2ST("logistic"),
    "c2st_knn": lambda: C2ST("knn"),
}


def build(key: str) -> TwoSampleTest:
    """Instantiate a registered test by name."""
    if key not in REGISTRY:
        raise KeyError(f"unknown test {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]()
