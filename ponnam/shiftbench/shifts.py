"""Faithful reimplementation of the *Failing Loudly* synthetic shift suite.

The shift catalogue, and the exact numeric parameters attached to each
intensity level, are taken from the reference implementation
(``shift_applicator.py`` and ``data_utils.py`` at
https://github.com/steverab/failing-loudly) rather than from the paper's prose,
because the prose is an incomplete description of what the code does.  Two
discrepancies are worth recording explicitly, since they change the shift and
therefore any power curve measured against it:

1. **The paper omits shear and flips from the image shifts.**  Section 4
   describes the image shift as "combinations of random rotations
   ``{10, 40, 90}``, (x, y)-axis-translation percentages ``{0.05, 0.2, 0.4}``,
   as well as zoom-in percentages ``{0.1, 0.2, 0.4}``".  The code additionally
   applies a shear of ``{0.1, 0.2, 0.3}`` and, at medium intensity, a random
   horizontal flip, and at large intensity both horizontal and vertical flips.
   A horizontal flip is a far stronger perturbation of MNIST than a 40-degree
   rotation, so a reimplementation that follows only the prose measures a
   materially weaker shift.

2. **The Gaussian-noise standard deviation is in raw 0-255 units.**  The
   quoted ``sigma`` in ``{1, 10, 100}`` is divided by 255 before being added to
   data that has already been scaled to ``[0, 1]``, and the result is clipped
   back to ``[0, 1]``.  The effective standard deviations are therefore
   approximately ``{0.0039, 0.0392, 0.3922}``, and the clipping makes the large
   setting strongly non-Gaussian -- a substantial fraction of pixels pile up
   exactly on the boundaries 0 and 1.  That boundary mass is the mechanism by
   which this shift is detectable at all by a test that only looks at means.

The image geometry is produced by replicating Keras's
``ImageDataGenerator.random_transform`` and ``apply_affine_transform``, which is
what the reference implementation calls.  Keras is not a dependency of this
package; the affine composition is reproduced directly on top of
:mod:`scipy.ndimage` so that results do not depend on a TensorFlow install.

Excluded shift
--------------
The adversarial (FGSM) shift of the paper is deliberately not implemented.  It
requires a trained neural network to generate the adversarial examples, and the
project brief directs us away from neural methods; every other shift in the
catalogue is reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from .datasets import PIXEL_SCALE

__all__ = [
    "ShiftSpec",
    "SHIFTS",
    "apply_shift",
    "gaussian_noise_subset",
    "knockout_shift",
    "only_one_shift",
    "image_shift",
]


# ---------------------------------------------------------------------------
# Keras-compatible affine transform
# ---------------------------------------------------------------------------


def _transform_matrix_offset_center(matrix: np.ndarray, h: int, w: int) -> np.ndarray:
    """Re-centre an affine matrix so it acts about the image centre.

    Keras uses ``h / 2 - 0.5`` rather than ``h / 2``; the half-pixel matters for
    small images such as MNIST's 28x28, so it is reproduced exactly.
    """
    o_x = float(h) / 2 - 0.5
    o_y = float(w) / 2 - 0.5
    offset = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    return offset @ matrix @ reset


def _apply_affine(
    x: np.ndarray,
    theta: float,
    tx: float,
    ty: float,
    shear: float,
    zx: float,
    zy: float,
) -> np.ndarray:
    """Apply one affine transform to a single ``(H, W, C)`` image.

    Mirrors ``keras_preprocessing.image.affine_transformations.apply_affine_transform``
    with ``fill_mode="nearest"`` and ``order=1`` (bilinear), which are the
    settings the reference implementation uses.
    """
    matrix = None

    if theta != 0:
        t = np.deg2rad(theta)
        rot = np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )
        matrix = rot

    if tx != 0 or ty != 0:
        shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        matrix = shift if matrix is None else matrix @ shift

    if shear != 0:
        s = np.deg2rad(shear)
        sh = np.array([[1, -np.sin(s), 0], [0, np.cos(s), 0], [0, 0, 1]])
        matrix = sh if matrix is None else matrix @ sh

    if zx != 1 or zy != 1:
        zoom = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        matrix = zoom if matrix is None else matrix @ zoom

    if matrix is None:
        return x

    h, w = x.shape[0], x.shape[1]
    matrix = _transform_matrix_offset_center(matrix, h, w)
    final_matrix = matrix[:2, :2]
    final_offset = matrix[:2, 2]

    out = np.empty_like(x)
    for c in range(x.shape[2]):
        out[..., c] = ndimage.affine_transform(
            x[..., c],
            final_matrix,
            final_offset,
            order=1,
            mode="nearest",
            cval=0.0,
        )
    return out


def _random_transform(
    x: np.ndarray,
    rng: np.random.Generator,
    rotation_range: float,
    width_shift_range: float,
    height_shift_range: float,
    shear_range: float,
    zoom_range: float,
    horizontal_flip: bool,
    vertical_flip: bool,
) -> np.ndarray:
    """Draw and apply one random transform, following Keras's parameterisation.

    Keras treats a shift range below 1 as a fraction of the corresponding image
    dimension, and draws the zoom factors for the two axes independently.
    """
    h, w = x.shape[0], x.shape[1]

    theta = rng.uniform(-rotation_range, rotation_range) if rotation_range else 0.0
    # Keras: tx is along the row (height) axis, ty along the column (width) axis.
    tx = rng.uniform(-height_shift_range, height_shift_range) if height_shift_range else 0.0
    ty = rng.uniform(-width_shift_range, width_shift_range) if width_shift_range else 0.0
    if abs(tx) < 1:
        tx *= h
    if abs(ty) < 1:
        ty *= w
    shear = rng.uniform(-shear_range, shear_range) if shear_range else 0.0
    if zoom_range:
        zx, zy = rng.uniform(1 - zoom_range, 1 + zoom_range, 2)
    else:
        zx = zy = 1.0

    out = _apply_affine(x, theta, tx, ty, shear, zx, zy)

    if horizontal_flip and rng.random() < 0.5:
        out = out[:, ::-1, :]
    if vertical_flip and rng.random() < 0.5:
        out = out[::-1, :, :]
    return out


# ---------------------------------------------------------------------------
# Individual shift primitives
# ---------------------------------------------------------------------------


def _subset_indices(n: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of the fraction ``delta`` of rows a shift is applied to.

    ``ceil`` matches the reference implementation, so ``delta = 0.1`` on 10
    samples perturbs 1 sample rather than 0.
    """
    return rng.choice(n, ceil(n * delta), replace=False)


def gaussian_noise_subset(
    x: np.ndarray,
    noise_amt: float,
    rng: np.random.Generator,
    normalization: float = PIXEL_SCALE,
    delta: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """Add ``N(0, (noise_amt / normalization)^2)`` noise to a fraction of rows.

    Note the clipping to ``[0, 1]``: at ``noise_amt = 100`` this is the dominant
    effect and makes the shifted marginal strongly bimodal at the boundaries.
    """
    x = x.copy()
    idx = _subset_indices(len(x), delta, rng)
    noise = rng.normal(0, noise_amt / normalization, (len(idx), x.shape[1]))
    mod = x[idx] + noise
    x[idx] = np.clip(mod, 0.0, 1.0) if clip else mod
    return x


def knockout_shift(
    x: np.ndarray, y: np.ndarray, cl: int, delta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Delete a fraction ``delta`` of the examples belonging to class ``cl``.

    This is label shift: the class marginal changes while ``p(x | y)`` does not.
    The reference implementation rounds the deletion count *up* to an even
    number, which is reproduced here so that sample counts match exactly.
    """
    del_idx = np.where(y == cl)[0]
    until = ceil(delta * len(del_idx))
    if until % 2 != 0:
        until += 1
    del_idx = del_idx[:until]
    return np.delete(x, del_idx, axis=0), np.delete(y, del_idx, axis=0)


def only_one_shift(
    x: np.ndarray, y: np.ndarray, cl: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Retain only class ``cl`` -- the most extreme label shift in the suite."""
    for other in np.unique(y):
        if other == cl:
            continue
        x, y = knockout_shift(x, y, int(other), 1.0)
    return x, y


def image_shift(
    x: np.ndarray,
    image_shape: Tuple[int, int, int],
    rng: np.random.Generator,
    rotation: float,
    width_shift: float,
    height_shift: float,
    shear: float,
    zoom: float,
    horizontal_flip: bool,
    vertical_flip: bool,
    delta: float = 1.0,
) -> np.ndarray:
    """Apply a random affine perturbation to a fraction ``delta`` of rows."""
    x = x.copy()
    idx = _subset_indices(len(x), delta, rng)
    for i in idx:
        img = x[i].reshape(image_shape)
        x[i] = _random_transform(
            img,
            rng,
            rotation,
            width_shift,
            height_shift,
            shear,
            zoom,
            horizontal_flip,
            vertical_flip,
        ).reshape(-1)
    return x


# ---------------------------------------------------------------------------
# Shift catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftSpec:
    """One entry of the shift catalogue.

    ``kind`` selects the primitive; ``params`` carries its arguments; ``delta``
    is the fraction of target samples affected.  ``family`` and ``intensity``
    are metadata used to group results in the analysis scripts.
    """

    name: str
    kind: str
    delta: float
    family: str
    intensity: str
    params: Dict[str, float] = field(default_factory=dict)
    shifts_labels: bool = False


# Parameters per intensity level, read off shift_applicator.py.  The tuple order
# is (rotation, width_shift, height_shift, shear, zoom, hflip, vflip).
_IMG_PARAMS = {
    "small": dict(rotation=10, width_shift=0.05, height_shift=0.05, shear=0.1,
                  zoom=0.1, horizontal_flip=False, vertical_flip=False),
    "medium": dict(rotation=40, width_shift=0.2, height_shift=0.2, shear=0.2,
                   zoom=0.2, horizontal_flip=True, vertical_flip=False),
    "large": dict(rotation=90, width_shift=0.4, height_shift=0.4, shear=0.3,
                  zoom=0.4, horizontal_flip=True, vertical_flip=True),
}

_GN_SIGMA = {"small": 1.0, "medium": 10.0, "large": 100.0}

_DELTAS = (0.1, 0.5, 1.0)


def _build_catalogue() -> Dict[str, ShiftSpec]:
    cat: Dict[str, ShiftSpec] = {}

    # The no-shift control.  Essential: it is the only cell of the grid where we
    # know the truth is H0, so it is what the Type-I error rate is measured on.
    cat["rand"] = ShiftSpec("rand", "none", 0.0, "control", "none")

    for level, sigma in _GN_SIGMA.items():
        for d in _DELTAS:
            n = f"{level}_gn_shift_{d}"
            cat[n] = ShiftSpec(n, "gaussian_noise", d, "covariate", level,
                               {"noise_amt": sigma})

    for level, p in _IMG_PARAMS.items():
        for d in _DELTAS:
            n = f"{level}_img_shift_{d}"
            cat[n] = ShiftSpec(n, "image", d, "covariate", level, dict(p))

    for d in _DELTAS:
        n = f"ko_shift_{d}"
        cat[n] = ShiftSpec(n, "knockout", d, "label", f"delta={d}",
                           {"cl": 0}, shifts_labels=True)

    # Combined covariate + label shifts.
    for d in _DELTAS:
        n = f"medium_img_shift_0.5+ko_shift_{d}"
        cat[n] = ShiftSpec(n, "image+knockout", d, "mixed", "medium",
                           dict(_IMG_PARAMS["medium"], img_delta=0.5, cl=0),
                           shifts_labels=True)

    for d in _DELTAS:
        n = f"only_zero_shift+medium_img_shift_{d}"
        cat[n] = ShiftSpec(n, "onlyzero+image", d, "mixed", "medium",
                           dict(_IMG_PARAMS["medium"], cl=0), shifts_labels=True)

    return cat


SHIFTS: Dict[str, ShiftSpec] = _build_catalogue()


def apply_shift(
    x: np.ndarray,
    y: np.ndarray,
    spec: ShiftSpec,
    image_shape: Tuple[int, int, int],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one catalogued shift to ``(x, y)`` and return the shifted pair.

    ``x`` is expected flattened and scaled to ``[0, 1]`` as produced by
    :func:`shiftbench.datasets.load_split`.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x, y = x.copy(), y.copy()
    p = spec.params
    kind = spec.kind

    if kind == "none":
        return x, y

    if kind == "gaussian_noise":
        return gaussian_noise_subset(x, p["noise_amt"], rng, delta=spec.delta), y

    if kind == "image":
        return image_shift(
            x, image_shape, rng,
            p["rotation"], p["width_shift"], p["height_shift"], p["shear"],
            p["zoom"], p["horizontal_flip"], p["vertical_flip"],
            delta=spec.delta,
        ), y

    if kind == "knockout":
        return knockout_shift(x, y, int(p["cl"]), spec.delta)

    if kind == "image+knockout":
        x = image_shift(
            x, image_shape, rng,
            p["rotation"], p["width_shift"], p["height_shift"], p["shear"],
            p["zoom"], p["horizontal_flip"], p["vertical_flip"],
            delta=p["img_delta"],
        )
        return knockout_shift(x, y, int(p["cl"]), spec.delta)

    if kind == "onlyzero+image":
        x, y = only_one_shift(x, y, int(p["cl"]))
        x = image_shift(
            x, image_shape, rng,
            p["rotation"], p["width_shift"], p["height_shift"], p["shear"],
            p["zoom"], p["horizontal_flip"], p["vertical_flip"],
            delta=spec.delta,
        )
        return x, y

    raise ValueError(f"unknown shift kind {kind!r}")
