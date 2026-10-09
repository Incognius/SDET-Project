"""Canonical dataset loading for the distribution-shift benchmark.

This module reproduces the dataset inventory of Rabanser, Guennemann and Lipton,
*Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*
(NeurIPS 2019), arXiv:1810.11953, together with the reference implementation at
https://github.com/steverab/failing-loudly.

Every loader returns images already flattened to ``(N, D)`` and scaled to
``[0, 1]`` by dividing by 255, which is the normalisation the reference
implementation applies before any shift is simulated.  Preserving that order of
operations matters: the Gaussian-noise shifts in :mod:`shiftbench.shifts` are
specified as a standard deviation *in raw 0-255 units* that is divided by 255 at
application time, so applying them to data on a different scale silently changes
the shift intensity.

Splits
------
The paper splits each dataset into a training portion used only to fit the
dimensionality-reduction mappings, a validation portion that plays the role of
the source sample ``p``, and a test portion to which a shift is applied to
produce the target sample ``q``.  The split sizes follow the paper:

====================  =========  =========  =========  ==========  =========
Dataset               ``N_tr``   ``N_val``  ``N_te``   ``D``       ``C``
====================  =========  =========  =========  ==========  =========
MNIST                 50 000     10 000     10 000     28*28*1     10
CIFAR-10              40 000     10 000     10 000     32*32*3     10
Fashion-MNIST         50 000     10 000     10 000     28*28*1     10
SVHN                  busy       10 000     10 000     32*32*3     10
MNIST->USPS            1 000      1 000      1 000     16*16*1     10
====================  =========  =========  =========  ==========  =========

``busy`` denotes "whatever remains after the validation and test portions are
carved out", since SVHN's native training split is larger than CIFAR-10's.

Determinism
-----------
All shuffling is driven by an explicit :class:`numpy.random.Generator` seeded
from the ``seed`` argument.  Two machines that call ``load_split(name, seed)``
with the same arguments obtain bit-identical arrays; this is what makes the
benchmark shareable across the four team members without exchanging large data
files.  :func:`split_digest` exposes a checksum so that equality can be
verified rather than assumed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "DatasetSpec",
    "DATASETS",
    "load_raw",
    "load_split",
    "split_digest",
    "data_root",
]

# The reference implementation normalises by 255 before shifting.  Keep this a
# module constant so the shift module can refer to the same number.
PIXEL_SCALE = 255.0


def data_root() -> str:
    """Directory used to cache downloaded datasets.

    Overridable with the ``SHIFTBENCH_DATA`` environment variable so that the
    cache can be pointed at a scratch disk rather than a synchronised folder.
    """
    root = os.environ.get("SHIFTBENCH_DATA")
    if root is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(here, "data", "raw")
    os.makedirs(root, exist_ok=True)
    return root


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one dataset in the benchmark."""

    name: str
    image_shape: Tuple[int, int, int]  # (H, W, C) in the paper's convention
    n_classes: int
    n_train: int
    n_val: int
    n_test: int

    @property
    def dim(self) -> int:
        h, w, c = self.image_shape
        return h * w * c


DATASETS = {
    "mnist": DatasetSpec("mnist", (28, 28, 1), 10, 50_000, 10_000, 10_000),
    "fashion_mnist": DatasetSpec("fashion_mnist", (28, 28, 1), 10, 50_000, 10_000, 10_000),
    "cifar10": DatasetSpec("cifar10", (32, 32, 3), 10, 40_000, 10_000, 10_000),
    "svhn": DatasetSpec("svhn", (32, 32, 3), 10, 53_257, 10_000, 10_000),
    "mnist_usps": DatasetSpec("mnist_usps", (16, 16, 1), 10, 1_000, 1_000, 1_000),
}


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


def _load_pooled(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the pooled (train + test) arrays for a single-domain dataset.

    The paper reshuffles train and test together before splitting, so the
    native train/test boundary is not meaningful here and we concatenate.

    MNIST, Fashion-MNIST and CIFAR-10 come through :mod:`shiftbench._download`,
    which uses fast mirrors; torchvision's defaults for these throttle to the
    point of being unusable (see that module's docstring).  SVHN still goes
    through torchvision, since it is not part of the headline grid.
    """
    root = data_root()
    if name in ("mnist", "fashion_mnist", "cifar10"):
        from ._download import fetch

        return fetch(name, root)
    if name == "svhn":
        import torchvision  # imported lazily: only needed when the cache is cold

        tr = torchvision.datasets.SVHN(root, split="train", download=True)
        te = torchvision.datasets.SVHN(root, split="test", download=True)
        # torchvision stores SVHN as (N, C, H, W); the paper works in (N, H, W, C).
        x = np.concatenate([tr.data, te.data], axis=0).transpose(0, 2, 3, 1)
        y = np.concatenate([tr.labels, te.labels], axis=0)
        return x.astype(np.uint8), y.astype(np.int64)
    raise ValueError(f"no loader for {name!r}")


def _load_mnist_usps() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MNIST -> USPS domain-adaptation pair, both resized to 16x16 greyscale.

    The reference implementation ships a pre-built ``MNIST_vs_USPS.mat`` through
    Git LFS.  Rather than depend on that pointer file resolving, we rebuild the
    same pair from the canonical sources: MNIST via torchvision and USPS via
    torchvision's USPS loader, downsampling MNIST from 28x28 to 16x16 so that
    the two domains share a feature space.  This reproduces the paper's stated
    ``D = 16 x 16 x 1`` and ``N_tr = N_val = N_te = 1000``.
    """
    import torch
    import torch.nn.functional as F
    import torchvision

    root = data_root()
    mx, my = _load_pooled("mnist")
    u_tr = torchvision.datasets.USPS(root, train=True, download=True)
    u_te = torchvision.datasets.USPS(root, train=False, download=True)

    mnist_x = torch.from_numpy(mx[..., 0]).float().unsqueeze(1)  # (N, 1, 28, 28)
    # Area interpolation is the right choice for downsampling: it averages the
    # pixels that fall into each output cell instead of point-sampling them.
    mnist_x = F.interpolate(mnist_x, size=(16, 16), mode="area")
    mnist_x = mnist_x.squeeze(1).numpy().astype(np.uint8)[..., None]
    mnist_y = my.astype(np.int64)

    usps_x = np.concatenate([np.asarray(u_tr.data), np.asarray(u_te.data)], axis=0)
    usps_y = np.concatenate(
        [np.asarray(u_tr.targets), np.asarray(u_te.targets)], axis=0
    ).astype(np.int64)
    if usps_x.ndim == 3:
        usps_x = usps_x[..., None]
    usps_x = usps_x.astype(np.uint8)

    return mnist_x, mnist_y, usps_x, usps_y


def load_raw(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for a single-domain dataset, as uint8 ``(N, H, W, C)``."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    if name == "mnist_usps":
        raise ValueError("mnist_usps is a domain pair; use load_split instead")
    return _load_pooled(name)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@dataclass
class Split:
    """One train / validation / test partition of a dataset.

    ``x_val`` plays the role of the source sample drawn from ``p``; ``x_test``
    is the sample to which a shift is applied to obtain the target sample from
    ``q``.  ``x_train`` is reserved for fitting dimensionality reductions and is
    never used as either side of a two-sample test.
    """

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    spec: DatasetSpec

    @property
    def dim(self) -> int:
        return self.x_val.shape[1]


def load_split(name: str, seed: int = 0) -> Split:
    """Build a deterministic train/val/test split, flattened and scaled to [0, 1].

    Parameters
    ----------
    name
        One of :data:`DATASETS`.
    seed
        Seed for the shuffle.  The paper averages detection performance over
        five random splits; we expose the seed so that the five runs are
        explicit and reproducible rather than implicit in global RNG state.
    """
    spec = DATASETS[name]
    rng = np.random.default_rng(seed)

    if name == "mnist_usps":
        mx, my, ux, uy = _load_mnist_usps()
        # Source domain: MNIST.  Target domain: USPS.  Each side is shuffled and
        # truncated to 1000 examples per the paper's N_tr = N_val = N_te = 1000.
        mi = rng.permutation(len(mx))
        ui = rng.permutation(len(ux))
        n = spec.n_val
        x_train, y_train = mx[mi[: spec.n_train]], my[mi[: spec.n_train]]
        x_val, y_val = mx[mi[spec.n_train : spec.n_train + n]], my[mi[spec.n_train : spec.n_train + n]]
        x_test, y_test = ux[ui[:n]], uy[ui[:n]]
    else:
        x, y = load_raw(name)
        idx = rng.permutation(len(x))
        x, y = x[idx], y[idx]
        n_tr, n_val, n_te = spec.n_train, spec.n_val, spec.n_test
        total = n_tr + n_val + n_te
        if total > len(x):
            raise ValueError(
                f"{name}: requested {total} examples but only {len(x)} available"
            )
        x_train, y_train = x[:n_tr], y[:n_tr]
        x_val, y_val = x[n_tr : n_tr + n_val], y[n_tr : n_tr + n_val]
        x_test, y_test = x[n_tr + n_val : total], y[n_tr + n_val : total]

    def prep(a: np.ndarray) -> np.ndarray:
        return (a.reshape(len(a), -1).astype(np.float32) / PIXEL_SCALE)

    return Split(
        name=name,
        x_train=prep(x_train),
        y_train=np.asarray(y_train).ravel(),
        x_val=prep(x_val),
        y_val=np.asarray(y_val).ravel(),
        x_test=prep(x_test),
        y_test=np.asarray(y_test).ravel(),
        spec=spec,
    )


def split_digest(split: Split) -> str:
    """SHA-256 over the split's arrays, for cross-machine equality checks.

    Two team members who obtain the same digest are provably testing on the
    same bytes, which is the point of the unified benchmark.  Floating-point
    arrays are hashed in their exact binary representation, so this catches
    library-version differences that silently perturb the data as well as
    outright seed mismatches.
    """
    h = hashlib.sha256()
    h.update(split.name.encode())
    for arr in (
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        split.x_test,
        split.y_test,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
