"""Experiment drivers: calibration, power curves, and the real-data pipeline.

Three studies are defined here, in the order they must be run:

1. :func:`run_calibration` -- measure each test's *actual* Type-I error rate
   against its nominal level.  Nothing downstream is interpretable until this
   passes, because a power comparison between a test holding 5% and a test
   holding 18% is not a comparison of power.
2. :func:`run_probe_power` -- power curves on the synthetic probes, where the
   ground truth is known by construction and every assumption break is
   deliberate.
3. :func:`run_image_pipeline` -- the *Failing Loudly* pipeline on real image
   data: reduce, then test, across the full shift catalogue.

All three return tidy :class:`pandas.DataFrame` objects with a common schema so
they can be concatenated and filtered uniformly.

Parallelism and reproducibility
-------------------------------
The grids are large -- the calibration study alone is hundreds of cells of
hundreds of permutation-calibrated replicates -- so cells are evaluated in
parallel across processes.  Reproducibility is preserved by deriving each
cell's seed deterministically from the global seed and the cell's own
coordinates via :func:`_cell_seed`, rather than from a shared generator whose
consumption order would depend on scheduling.  A run with ``n_jobs=1`` and a
run with ``n_jobs=16`` therefore produce identical numbers.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .datasets import Split, load_split
from .reductors import build_reductor
from .shifts import SHIFTS, apply_shift
from .synthetic import sample_probe
from .tests import build as build_test

__all__ = [
    "run_calibration",
    "run_probe_power",
    "run_image_pipeline",
    "SAMPLE_SIZES",
    "ALPHAS",
]

#: Target sample sizes from the paper's experimental protocol.  The paper also
#: evaluates 10 000 but caps kernel methods at 1 000 for cost; the cap is
#: enforced inside the kernel tests themselves rather than here.
SAMPLE_SIZES: Sequence[int] = (10, 20, 50, 100, 200, 500, 1000)

#: Levels at which rejection rates are reported.  The paper fixes alpha = 0.05;
#: reporting several makes the conservatism of Bonferroni visible as a curve
#: rather than a single point.
ALPHAS: Sequence[float] = (0.01, 0.05, 0.10)


def _cell_seed(global_seed: int, *parts) -> int:
    """Derive a stable per-cell seed from the global seed and cell coordinates.

    Hashing the coordinates rather than incrementing a counter means the seed
    for a given cell does not change when unrelated cells are added to or
    removed from the grid, so partial re-runs stay consistent with earlier
    complete ones.
    """
    key = "|".join([str(global_seed)] + [str(p) for p in parts])
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _rows_from_result(res, **fields) -> List[dict]:
    """Expand one test result into one row per reported alpha.

    A skipped cell keeps ``p_value = nan`` and ``reject = 0`` but is flagged, so
    that summaries can drop it.  Treating "could not be computed" as "did not
    reject" would quietly average a structural impossibility into a power
    estimate.
    """
    return [
        dict(
            p_value=res.p_value,
            statistic=res.statistic,
            elapsed_s=res.elapsed_s,
            skipped=int(res.skipped),
            reason=res.reason,
            alpha=alpha,
            reject=int(res.reject(alpha)),
            **fields,
        )
        for alpha in ALPHAS
    ]


def _parallel(jobs, n_jobs: int, verbose: bool, label: str) -> List[List[dict]]:
    """Run ``jobs`` (zero-argument callables) with progress reporting."""
    total = len(jobs)
    if n_jobs == 1:
        out = []
        t0 = time.perf_counter()
        for i, job in enumerate(jobs, 1):
            out.append(job())
            if verbose and (i % 10 == 0 or i == total):
                print(f"  [{label} {i}/{total}] {time.perf_counter() - t0:6.0f}s",
                      flush=True)
        return out

    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0, batch_size=1)(
        delayed(job)() for job in jobs
    )
    if verbose:
        print(f"  [{label}] {total} cells in {time.perf_counter() - t0:.0f}s "
              f"across {n_jobs} processes", flush=True)
    return out


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------


def _calibration_cell(
    key: str, probe: str, d: int, n: int, n_replicates: int, seed: int
) -> List[dict]:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for r in range(n_replicates):
        # eps = 0 -> both samples come from p, so H0 is true by construction.
        x, y = sample_probe(probe, n, n, d, 0.0, rng)
        res = build_test(key)(x, y, rng)
        rows.extend(
            _rows_from_result(
                res, study="calibration", test=key, probe=probe,
                d=d, n=n, replicate=r, eps=0.0,
            )
        )
    return rows


def run_calibration(
    test_keys: Sequence[str],
    n_replicates: int = 200,
    sample_sizes: Sequence[int] = (20, 50, 200),
    dims: Sequence[int] = (8, 32),
    probes: Sequence[str] = ("mean_sparse", "heavy_tail", "mixture"),
    seed: int = 0,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Measure each test's empirical Type-I error under a true null.

    Both samples are drawn from the *same* distribution, so every rejection is
    a false alarm by construction.  A correctly calibrated test rejects at its
    nominal level; the binomial standard error on ``n_replicates`` draws at
    ``alpha = 0.05`` is ``sqrt(0.05 * 0.95 / n)``, which is 1.0 percentage
    points at ``n = 500``.  Read anything beyond roughly three standard errors
    as a genuine calibration failure.

    The null distribution is varied across ``probes`` because calibration is
    not a property of a test alone: a test that holds its level under Gaussian
    data may not hold it under heavy-tailed or bimodal data, and that
    difference is the point.
    """
    jobs = []
    for probe in probes:
        for d in dims:
            for n in sample_sizes:
                for key in test_keys:
                    s = _cell_seed(seed, "calib", key, probe, d, n)
                    jobs.append(
                        _Job(_calibration_cell, key, probe, d, n, n_replicates, s)
                    )
    out = _parallel(jobs, n_jobs, verbose, "calib")
    return pd.DataFrame([row for cell in out for row in cell])


# ---------------------------------------------------------------------------
# 2. Synthetic probe power
# ---------------------------------------------------------------------------


def _power_cell(
    key: str, probe: str, d: int, n: int, eps: float, n_replicates: int, seed: int
) -> List[dict]:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for r in range(n_replicates):
        x, y = sample_probe(probe, n, n, d, eps, rng)
        res = build_test(key)(x, y, rng)
        rows.extend(
            _rows_from_result(
                res, study="probe_power", test=key, probe=probe,
                d=d, n=n, eps=eps, replicate=r,
            )
        )
    return rows


def run_probe_power(
    test_keys: Sequence[str],
    probes: Sequence[str],
    eps_values: Sequence[float],
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
    dims: Sequence[int] = (32,),
    n_replicates: int = 100,
    seed: int = 0,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Power curves on the synthetic probes.

    Power is the fraction of replicates in which a test rejects when the
    alternative is true.  Because the probes are constructed so that specific
    tests are *structurally* blind, a flat power curve sitting at the nominal
    level is the expected and correct result for those tests -- it confirms the
    construction rather than indicating a bug.
    """
    jobs = []
    for probe in probes:
        for d in dims:
            for eps in eps_values:
                for n in sample_sizes:
                    for key in test_keys:
                        s = _cell_seed(seed, "power", key, probe, d, n, eps)
                        jobs.append(
                            _Job(_power_cell, key, probe, d, n, eps,
                                 n_replicates, s)
                        )
    out = _parallel(jobs, n_jobs, verbose, "power")
    return pd.DataFrame([row for cell in out for row in cell])


# ---------------------------------------------------------------------------
# 3. Real-data pipeline
# ---------------------------------------------------------------------------


def _fit_reductors(split: Split, reductor_names: Sequence[str]) -> Dict[str, object]:
    """Fit every reduction on the training split only."""
    out = {}
    for name in reductor_names:
        red = build_reductor(name)
        red.fit(split.x_train, split.y_train)
        out[name] = red
    return out


def _image_chunk(
    dataset: str,
    seed: int,
    shift_names: Sequence[str],
    test_keys: Sequence[str],
    reductor_names: Sequence[str],
    sample_sizes: Sequence[int],
) -> List[dict]:
    """One (dataset, seed) job covering several shifts.

    Loading a split and fitting the reductions are expensive and depend only on
    ``(dataset, seed)`` -- fitting BBSDs means a multinomial logistic regression
    on 50 000 examples, which costs minutes on CIFAR-10.  Batching all of a
    seed's shifts into one job pays that cost once per job instead of once per
    shift, which for the full 28-shift catalogue is a 28-fold saving on the
    setup.
    """
    split = load_split(dataset, seed=seed)
    reductors = _fit_reductors(split, reductor_names)

    rows: List[dict] = []
    for shift_name in shift_names:
        spec = SHIFTS[shift_name]
        # Seed from the shift name as well, so a shift's realisation does not
        # depend on which chunk it happened to be placed in.
        rng = np.random.default_rng(_cell_seed(seed, "img", dataset, shift_name))
        x_shift, y_shift = apply_shift(
            split.x_test, split.y_test, spec, split.spec.image_shape, rng
        )

        for n in sample_sizes:
            if n > len(x_shift) or n > len(split.x_val):
                continue
            src_raw = split.x_val[rng.choice(len(split.x_val), n, replace=False)]
            tgt_raw = x_shift[rng.choice(len(x_shift), n, replace=False)]

            for red_name, red in reductors.items():
                # BBSDh is a single categorical column; the continuous tests in
                # this package are not applicable to it.
                if red_name == "BBSDh":
                    continue
                src = red.transform(src_raw)
                tgt = red.transform(tgt_raw)
                for key in test_keys:
                    res = build_test(key)(src, tgt, rng)
                    rows.extend(
                        _rows_from_result(
                            res, study="image_pipeline", dataset=dataset,
                            test=key, reductor=red_name, shift=shift_name,
                            shift_family=spec.family,
                            shift_intensity=spec.intensity, delta=spec.delta,
                            is_null=int(spec.kind == "none"),
                            d=int(src.shape[1]), n=n, source_n=n, seed=seed,
                        )
                    )
    return rows


def _chunk(seq: Sequence, n_chunks: int) -> List[List]:
    """Split ``seq`` into at most ``n_chunks`` roughly equal contiguous pieces."""
    seq = list(seq)
    n_chunks = max(1, min(n_chunks, len(seq)))
    size = -(-len(seq) // n_chunks)  # ceiling division
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def run_image_pipeline(
    dataset: str,
    test_keys: Sequence[str],
    reductor_names: Sequence[str] = ("NoRed", "PCA", "SRP", "BBSDs"),
    shift_names: Optional[Sequence[str]] = None,
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run reduce-then-test over the shift catalogue on one image dataset.

    Follows the paper's protocol: the reduction is fitted on the training
    split, the source sample is the (unshifted) validation split, and the
    target sample is drawn from the shifted test split.  Averaging over
    ``seeds`` reproduces the paper's five random splits.

    One asymmetry in the reference implementation is *not* reproduced here.
    That code passes the full source sample to the univariate tests but
    subsamples the source to the target's size before the kernel test, so MMD
    is run at ``n`` versus ``n`` while KS is run at ``10000`` versus ``n``.
    That difference alone advantages the univariate tests at small ``n``, and
    since this project is explicitly about small-sample behaviour, both sides
    are matched to the same ``n`` here.  The ``source_n`` column records the
    choice so the two conventions can be compared directly if wanted.
    """
    if shift_names is None:
        shift_names = list(SHIFTS)

    # Aim for a few jobs per worker so the pool stays busy even when shifts
    # differ in cost, while keeping the per-job setup (split load, reductor
    # fits) amortised over several shifts.
    import os

    workers = os.cpu_count() or 4 if n_jobs in (-1, 0) else abs(n_jobs)
    chunks_per_seed = max(1, -(-(workers * 2) // max(1, len(seeds))))
    jobs = [
        _Job(_image_chunk, dataset, seed, chunk, list(test_keys),
             list(reductor_names), list(sample_sizes))
        for seed in seeds
        for chunk in _chunk(shift_names, chunks_per_seed)
    ]
    out = _parallel(jobs, n_jobs, verbose, f"img:{dataset}")
    return pd.DataFrame([row for cell in out for row in cell])


class _Job:
    """A picklable zero-argument callable.

    ``joblib`` must pickle each task to ship it to a worker process, and a
    local closure or lambda cannot be pickled on Windows, where processes are
    spawned rather than forked.  This small wrapper is the portable
    alternative.
    """

    __slots__ = ("fn", "args")

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def __call__(self):
        return self.fn(*self.args)
