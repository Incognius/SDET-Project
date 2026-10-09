"""Dimensionality reduction stages of the detection pipeline.

The *Failing Loudly* pipeline is a composition: reduce, then test.  The
reduction is fitted on the training split only and then applied to the source
and target samples, so it never sees the data it will be used to compare.

The paper's seven reductions are ``NoRed``, ``PCA``, ``SRP``, ``UAE``, ``TAE``,
``BBSDs`` and ``BBSDh``.  Four are implemented here.  The two autoencoders are
omitted deliberately -- they are neural networks whose only role is to produce a
32-dimensional embedding, the project brief directs effort away from neural
methods, and PCA already provides a linear 32-dimensional embedding to compare
against.  ``BBSDs`` and ``BBSDh`` are implemented on top of a multinomial
logistic regression rather than the paper's ResNet-18, for the same reason: the
mechanism being tested is "project onto a label classifier's outputs", and a
convex classifier exercises that mechanism without a training budget.  Where
this weakens the comparison it is said so in the results rather than glossed.

A property of the whole family that is easy to lose sight of: every one of these
maps is surjective from ``D`` down to ``K``, so each is capable of mapping a
genuinely shifted input distribution onto an unshifted latent distribution.
Detection after reduction is therefore not a test of ``p(x) = q(x)`` but of
``p(f(x)) = q(f(x))``, which is a strictly weaker claim.  The paper argues this
is empirically benign in non-adversarial settings; it remains an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Reductor", "NoReduction", "PCAReduction", "SRPReduction", "BBSD", "build_reductor", "REDUCTORS"]

#: The paper reduces to 32 latent dimensions for PCA, SRP and both autoencoders,
#: a setting it justifies by noting that 32 components explain roughly 80% of
#: the variance of CIFAR-10.
DEFAULT_K = 32


class Reductor:
    """Base class: fit on training data, then transform source and target."""

    name = "base"

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "Reductor":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def out_dim(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError


class NoReduction(Reductor):
    """Identity: test on the raw pixels.

    The paper's baseline, and the setting in which the dimension-related
    failures are most visible -- at ``D = 3072`` the Bonferroni correction
    divides the level by 3072 and Hotelling's ``T^2`` does not exist at all.
    """

    name = "NoRed"

    def __init__(self):
        self._d = None

    def fit(self, x, y=None):
        self._d = x.shape[1]
        return self

    def transform(self, x):
        return x

    @property
    def out_dim(self):
        return self._d


class PCAReduction(Reductor):
    """Principal component analysis onto ``k`` components.

    Chosen to maximise retained variance, which is not the same thing as
    retaining the directions a shift acts along.  A shift confined to a
    low-variance direction is attenuated by exactly this step, and that is a
    predictable failure mode rather than bad luck.
    """

    name = "PCA"

    def __init__(self, k: int = DEFAULT_K):
        self.k = k
        self._model = None

    def fit(self, x, y=None):
        from sklearn.decomposition import PCA

        self._model = PCA(n_components=min(self.k, x.shape[1], len(x)), random_state=0)
        self._model.fit(x)
        return self

    def transform(self, x):
        return self._model.transform(x)

    @property
    def out_dim(self):
        return int(self._model.n_components_)


class SRPReduction(Reductor):
    """Sparse random projection onto ``k`` dimensions.

    Data-independent, so unlike PCA it cannot preferentially discard the shift
    direction -- but equally it cannot preferentially keep it.  By
    Johnson-Lindenstrauss it approximately preserves pairwise distances, which
    is exactly the structure the kernel and distance tests rely on, so it is
    the most natural reduction to pair with MMD.
    """

    name = "SRP"

    def __init__(self, k: int = DEFAULT_K, seed: int = 0):
        self.k = k
        self.seed = seed
        self._model = None

    def fit(self, x, y=None):
        from sklearn.random_projection import SparseRandomProjection

        self._model = SparseRandomProjection(
            n_components=min(self.k, x.shape[1]), random_state=self.seed
        )
        self._model.fit(x)
        return self

    def transform(self, x):
        return np.asarray(self._model.transform(x))

    @property
    def out_dim(self):
        return int(self._model.n_components_)


class BBSD(Reductor):
    """Black-box shift detection: project onto a label classifier's outputs.

    Lipton, Wang and Smola (2018) show that if a fixed classifier ``f`` has an
    invertible confusion matrix then ``p(x) != q(x)`` can be detected by testing
    ``p(f(x)) != q(f(x))``, under a label-shift assumption.  *Failing Loudly*
    finds empirically that this works well far outside that assumption, and
    reports it as the best reduction overall.

    ``soft=True`` (``BBSDs``) tests on the full ``C``-dimensional softmax
    vector; ``soft=False`` (``BBSDh``) tests on the single hard predicted label,
    which is categorical and therefore needs a chi-squared homogeneity test
    rather than any of the continuous tests in this package.

    The classifier here is multinomial logistic regression, not the paper's
    ResNet-18.  The reduction is consequently weaker -- a linear model's softmax
    on CIFAR-10 carries much less class information than a trained ResNet's --
    so BBSD results on this branch should be read as a lower bound on what the
    paper's version achieves, not as a reproduction of it.
    """

    def __init__(
        self,
        soft: bool = True,
        max_iter: int = 100,
        seed: int = 0,
        max_fit_samples: int = 10_000,
    ):
        self.soft = soft
        self.max_iter = max_iter
        self.seed = seed
        self.max_fit_samples = max_fit_samples
        self.name = "BBSDs" if soft else "BBSDh"
        self._model = None
        self._n_classes = None

    def fit(self, x, y=None):
        if y is None:
            raise ValueError("BBSD requires labels to fit the label classifier")
        from sklearn.linear_model import LogisticRegression
        import warnings

        self._n_classes = int(len(np.unique(y)))

        # Fitting on the full 50 000-example training split costs minutes per
        # worker process and buys very little: a linear softmax saturates well
        # before that, and the reduction's job is to be a fixed, informative
        # projection rather than a state-of-the-art classifier.  The subsample
        # is deterministic so the reduction stays reproducible.
        if self.max_fit_samples and len(x) > self.max_fit_samples:
            idx = np.random.default_rng(self.seed).choice(
                len(x), self.max_fit_samples, replace=False
            )
            x, y = x[idx], y[idx]

        self._model = LogisticRegression(
            max_iter=self.max_iter, random_state=self.seed, n_jobs=-1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(x, y)
        return self

    def transform(self, x):
        if self.soft:
            return self._model.predict_proba(x)
        return self._model.predict(x).reshape(-1, 1).astype(np.float64)

    @property
    def out_dim(self):
        return self._n_classes if self.soft else 1


REDUCTORS = {
    "NoRed": lambda: NoReduction(),
    "PCA": lambda: PCAReduction(),
    "SRP": lambda: SRPReduction(),
    "BBSDs": lambda: BBSD(soft=True),
    "BBSDh": lambda: BBSD(soft=False),
}


def build_reductor(name: str) -> Reductor:
    if name not in REDUCTORS:
        raise KeyError(f"unknown reductor {name!r}; known: {sorted(REDUCTORS)}")
    return REDUCTORS[name]()
