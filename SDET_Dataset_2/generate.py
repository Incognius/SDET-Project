"""Generate SDET_Dataset_2: a deliberately small shift-detection benchmark.

Two parts, each answering one question.

Synthetic (``data/synthetic``) -- *which test is best for which kind of change?*
    Five families on a standard Gaussian reference P = N(0, I).  Each family
    changes exactly one thing, so a test's success or failure has one
    explanation.

    ===============  ======================================  ====================
    family           Q at level k                            level is
    ===============  ======================================  ====================
    mean_sparse      N(a e_1, I)                             a (Mahalanobis)
    mean_dense       N(a 1/sqrt(d), I)                       a (Mahalanobis)
    variance         N(0, s^2 I)                             s (std multiplier)
    correlation      N(0, (1-r) I + r 11')                   r (pairwise corr.)
    heavytail_mean   t3 noise (unit variance) + a 1/sqrt(d)  a (Mahalanobis)
    ===============  ======================================  ====================

    The correlation family keeps every marginal exactly N(0, 1), so no
    per-feature test can see it.  heavytail_mean is mean_dense with the
    Gaussian assumption broken (P is also t3).

MNIST (``data/mnist``) -- *does it carry over to real data?*
    Three of the Failing Loudly shifts, applied to the target pool with the
    paper's parameters (``ponnam/shiftbench/shifts.py``):

    * gn   Gaussian pixel noise, sigma in {1, 10, 100}/255, fraction delta
    * img  random affine transform (rotation, shift, shear, zoom, flips)
    * ko   knock-out: delete a fraction delta of the digit-0 images (label shift)

    with delta in {0.1, 0.5, 1.0}.  The adversarial and combined shifts of the
    paper are left out on purpose.

Layout of a synthetic file (float32, layout (replicates, n, d))::

    X__n{n}          reference sample from P
    Y__n{n}__m{k}    new sample from Q_k, k = 0..6; k = 0 is the exact null
    S__levels        the level values
    S__delta         (K, d) mean shift per level (zeros where not a mean family)
    S__Sigma_Q       (K, d, d) covariance of Q_k before any t-noise scaling

Within a replicate every Y_k is built from the *same* base noise draw, so
power curves are smooth in k.  Cells with different k are therefore not
independent replications.

Run from the repository root::

    python SDET_Dataset_2/generate.py            # everything (about 2 minutes)
    python SDET_Dataset_2/generate.py --only synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
sys.path.insert(0, str(HERE.parent / "ponnam"))

SEED = 20261017

SYNTH = dict(d=[10, 50], n=[10, 20, 50, 100], R=100)

MEAN_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]
FAMILIES = {
    "mean_sparse": dict(levels=MEAN_LEVELS, level_name="a (Mahalanobis distance)",
                        what="Mean of feature 1 moves by a; every other feature unchanged."),
    "mean_dense": dict(levels=MEAN_LEVELS, level_name="a (Mahalanobis distance)",
                       what="Every feature's mean moves by a/sqrt(d); same total distance as mean_sparse."),
    "variance": dict(levels=[1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0], level_name="s (std multiplier)",
                     what="Every feature's standard deviation is multiplied by s; means unchanged."),
    "correlation": dict(levels=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5], level_name="r (pairwise correlation)",
                        what="Features become equicorrelated at r; every marginal stays exactly N(0, 1)."),
    "heavytail_mean": dict(levels=MEAN_LEVELS, level_name="a (Mahalanobis distance)",
                           what="mean_dense, but P and Q use Student-t (3 df) noise scaled to unit variance."),
}

MNIST_DELTAS = [0.1, 0.5, 1.0]
MNIST_LEVELS = ["small", "medium", "large"]


def _rng(*tags) -> np.random.Generator:
    ent = [SEED] + [zlib.crc32(str(t).encode()) for t in tags]
    return np.random.default_rng(np.random.SeedSequence(ent))


def _t3(rng, shape) -> np.ndarray:
    # t with 3 df has variance 3; divide by sqrt(3) for unit variance.
    return rng.standard_t(3, shape) / np.sqrt(3.0)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def synth_scenario(family: str, d: int, outdir: Path) -> dict:
    spec = FAMILIES[family]
    levels = np.asarray(spec["levels"], float)
    K, R = len(levels), SYNTH["R"]

    deltas = np.zeros((K, d))
    sigmas = np.tile(np.eye(d), (K, 1, 1))
    if family == "mean_sparse":
        deltas[:, 0] = levels
    elif family in ("mean_dense", "heavytail_mean"):
        deltas[:] = levels[:, None] / np.sqrt(d)
    elif family == "variance":
        sigmas = levels[:, None, None] ** 2 * sigmas
    elif family == "correlation":
        sigmas = np.stack([(1 - r) * np.eye(d) + r * np.ones((d, d)) for r in levels])
    chols = np.linalg.cholesky(sigmas)

    arrays = {}
    for n in SYNTH["n"]:
        rng = _rng(family, d, n)
        noise = _t3 if family == "heavytail_mean" else (lambda g, s: g.standard_normal(s))
        X = noise(rng, (R, n, d))
        Z = noise(rng, (R, n, d))  # base noise shared by every level
        arrays[f"X__n{n}"] = X.astype(np.float32)
        for k in range(K):
            Y = Z @ chols[k].T + deltas[k]
            arrays[f"Y__n{n}__m{k}"] = Y.astype(np.float32)

    name = f"{family}_d{d}"
    arrays.update(S__levels=levels, S__delta=deltas, S__Sigma_Q=sigmas,
                  S__Sigma_P=np.eye(d), S__mags=levels)
    path = outdir / f"{name}.npz"
    np.savez(path, **arrays)
    return dict(name=name, part="synthetic", family=family, file=f"synthetic/{path.name}",
                d=d, n=SYNTH["n"], R=R, levels=levels.tolist(), level_name=spec["level_name"],
                description=spec["what"],
                noise="student_t3_unit_var" if family == "heavytail_mean" else "gaussian",
                sha256=_sha(path))


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def mnist_part(outdir: Path) -> list:
    from shiftbench.datasets import load_split
    from shiftbench.shifts import (_GN_SIGMA, _IMG_PARAMS, gaussian_noise_subset,
                                   image_shift, knockout_shift)

    sp = load_split("mnist", seed=0)
    shape = sp.spec.image_shape
    entries = []

    src = outdir / "source.npz"
    np.savez_compressed(src, x_train=_u8(sp.x_train), y_train=sp.y_train,
                        x_source=_u8(sp.x_val), y_source=sp.y_val)
    entries.append(dict(name="source", part="mnist", file=f"mnist/{src.name}",
                        description="x_train (50k) fits the reducers; x_source (10k) is the reference pool P.",
                        sha256=_sha(src)))

    def save(name, x, y, family, level, delta, what, as_float=False):
        path = outdir / f"target_{name}.npz"
        xs = x.astype(np.float16) if as_float else _u8(x)
        np.savez_compressed(path, x=xs, y=y)
        entries.append(dict(name=name, part="mnist", file=f"mnist/{path.name}", family=family,
                            intensity=level, delta=delta, n_pool=int(len(x)),
                            dtype=str(xs.dtype), description=what, sha256=_sha(path)))
        print(f"  mnist {name:<18} pool={len(x)}", flush=True)

    save("none", sp.x_test, sp.y_test, "null", "none", 0.0, "Unshifted target pool: the null.")
    for level in MNIST_LEVELS:
        for delta in MNIST_DELTAS:
            sigma = _GN_SIGMA[level]
            x = gaussian_noise_subset(sp.x_test, sigma, _rng("gn", level, delta), delta=delta)
            # float16: rounding to 0-255 would erase most of the sigma = 1/255 noise.
            save(f"gn_{level}_{delta}", x, sp.y_test, "gn", level, delta,
                 f"Gaussian noise sigma={sigma:g}/255 on {delta:.0%} of images, clipped to [0,1].",
                 as_float=True)
    for level in MNIST_LEVELS:
        for delta in MNIST_DELTAS:
            p = _IMG_PARAMS[level]
            x = image_shift(sp.x_test, shape, _rng("img", level, delta),
                            p["rotation"], p["width_shift"], p["height_shift"], p["shear"],
                            p["zoom"], p["horizontal_flip"], p["vertical_flip"], delta=delta)
            save(f"img_{level}_{delta}", x, sp.y_test, "img", level, delta,
                 f"Random affine ({level}: rotation up to {p['rotation']} deg, shift {p['width_shift']}, "
                 f"zoom {p['zoom']}, shear {p['shear']}) on {delta:.0%} of images.")
    for delta in MNIST_DELTAS:
        x, y = knockout_shift(sp.x_test, sp.y_test, cl=0, delta=delta)
        save(f"ko_{delta}", x, y, "ko", f"delta={delta}", delta,
             f"Delete {delta:.0%} of the digit-0 images (label shift).")
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["synthetic", "mnist"])
    args = ap.parse_args(argv)

    man_path = OUT / "manifest.json"
    old = json.loads(man_path.read_text()) if man_path.exists() else {}
    scen = [s for s in old.get("scenarios", []) if args.only and s["part"] != args.only]

    if args.only in (None, "synthetic"):
        sd = OUT / "synthetic"
        sd.mkdir(parents=True, exist_ok=True)
        for fam in FAMILIES:
            for d in SYNTH["d"]:
                e = synth_scenario(fam, d, sd)
                scen.append(e)
                print(f"  synthetic {e['name']}", flush=True)
    if args.only in (None, "mnist"):
        md = OUT / "mnist"
        md.mkdir(parents=True, exist_ok=True)
        scen += mnist_part(md)

    manifest = dict(dataset="SDET_Dataset_2", seed=SEED, alpha=0.05, synthetic_grid=SYNTH,
                    mnist=dict(deltas=MNIST_DELTAS, intensities=MNIST_LEVELS,
                               split="shiftbench.datasets.load_split('mnist', seed=0)"),
                    numpy=np.__version__, scenarios=scen)
    man_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(scen)} entries to {man_path}")


if __name__ == "__main__":
    main()
