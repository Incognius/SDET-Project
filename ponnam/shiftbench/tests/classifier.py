"""Classifier two-sample test (C2ST).

Lopez-Paz, D. and Oquab, M., *Revisiting Classifier Two-Sample Tests*,
ICLR 2017, arXiv:1610.06545.

The construction is the one the project brief recommends implementing, and it
is worth stating precisely because its validity rests entirely on one detail.
Label the source sample ``0`` and the target sample ``1``, split the pooled,
labelled data into a training half and a held-out half, fit any binary
classifier on the training half, and use its accuracy on the *held-out* half as
the test statistic.

Under ``H0: p = q`` the labels are independent of the features, so no
classifier can do better than chance on data it has not seen, and the number of
correct held-out predictions is ``Binomial(n_held, 1/2)``.  That gives a
closed-form null with no permutation required.

The detail that matters: the split is not an efficiency measure, it is what
makes the test valid.  Training accuracy is not distributed as
``Binomial(n, 1/2)`` under ``H0`` -- a sufficiently flexible classifier drives
it to 1 by memorising the labels, and a test built on it would reject always.
Using held-out accuracy is the whole construction.

Two further practical points, both of which show up on this benchmark:

* The normal approximation to the binomial is poor at ``n_held = 5``, which is
  what a 50/50 split of ``n = 10`` gives.  An exact binomial test is used
  instead of the normal approximation of the original paper, so the reported
  level is honest in the small-sample regime that this project is about.
* Half the data is spent on training, so C2ST sees only ``n / 2`` examples'
  worth of evidence in its statistic.  This is a concrete, quantifiable reason
  to expect classical tests to win at small ``n``, and it is the hypothesis the
  power curves are designed to settle.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .base import TestResult, TwoSampleTest, permutation_pvalue

__all__ = ["C2ST"]


class C2ST(TwoSampleTest):
    """Classifier two-sample test.

    Parameters
    ----------
    classifier
        ``"logistic"`` (default) is a regularised logistic regression: cheap,
        convex, no tuning, and it cannot overfit badly enough to invalidate the
        held-out construction.  ``"knn"`` is the ``k``-nearest-neighbour variant
        used in the original paper.  ``"mlp"`` is a small neural network and is
        provided only for completeness -- the project brief deprioritises
        neural methods and it is not part of the headline comparison.
    calibration
        ``"binomial"`` uses the exact closed-form null.  ``"permutation"``
        refits the classifier on each permutation, which is far more expensive
        but is the only option that stays valid if the held-out split is
        compromised.  Used in the calibration study to confirm the binomial
        null is behaving.
    test_size
        Fraction of the pooled sample held out for the statistic.
    """

    def __init__(
        self,
        classifier: str = "logistic",
        calibration: str = "binomial",
        test_size: float = 0.5,
        n_permutations: int = 100,
    ):
        if classifier not in ("logistic", "knn", "mlp"):
            raise ValueError("classifier must be 'logistic', 'knn' or 'mlp'")
        self.classifier = classifier
        self.calibration = calibration
        self.test_size = test_size
        self.n_permutations = n_permutations
        self.name = f"C2ST({classifier})[{calibration}]"

    def _make_clf(self, seed: int, n_train: int):
        if self.classifier == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=200, random_state=seed)
        if self.classifier == "knn":
            from sklearn.neighbors import KNeighborsClassifier

            # The original paper uses k = n_train ** 0.5, floored at 1 and kept
            # below the training set size.
            k = max(1, min(int(np.sqrt(n_train)), max(1, n_train - 1)))
            return KNeighborsClassifier(n_neighbors=k)
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=300, random_state=seed
        )

    def _accuracy(self, x: np.ndarray, y: np.ndarray, seed: int) -> tuple:
        """Return ``(n_correct, n_held)`` from one train/held-out split."""
        import warnings
        from sklearn.model_selection import train_test_split

        feats = np.concatenate([x, y], axis=0)
        labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])

        # Stratify so that both classes are present on both sides of the split;
        # without it a small sample can produce a single-class training set and
        # the classifier is undefined.
        n_held = int(round(self.test_size * len(feats)))
        if n_held < 2 or len(feats) - n_held < 2:
            return 0, 0
        try:
            f_tr, f_te, l_tr, l_te = train_test_split(
                feats, labels, test_size=self.test_size,
                random_state=seed, stratify=labels,
            )
        except ValueError:
            return 0, 0
        if len(np.unique(l_tr)) < 2:
            return 0, 0

        clf = self._make_clf(seed, len(f_tr))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(f_tr, l_tr)
            pred = clf.predict(f_te)
        return int(np.sum(pred == l_te)), int(len(l_te))

    def _run(self, x, y, rng) -> TestResult:
        seed = int(rng.integers(0, 2**31 - 1))

        if self.calibration == "binomial":
            correct, n_held = self._accuracy(x, y, seed)
            if n_held == 0:
                return TestResult(1.0, 0.5, self.name, meta={"degenerate": 1.0})
            acc = correct / n_held
            # One-sided: only above-chance accuracy is evidence against H0.
            # A classifier reliably *below* chance would also indicate a
            # difference, but the test as defined in the paper is one-sided and
            # a two-sided version would halve the power for no practical gain.
            p = float(stats.binomtest(correct, n_held, 0.5, alternative="greater").pvalue)
            return TestResult(
                p, acc, self.name,
                meta={"accuracy": acc, "n_held": float(n_held)},
            )

        def stat(a, b):
            c, nh = self._accuracy(a, b, seed)
            return c / nh if nh else 0.5

        p, obs, _ = permutation_pvalue(x, y, stat, self.n_permutations, rng)
        return TestResult(p, obs, self.name, meta={"accuracy": obs})
