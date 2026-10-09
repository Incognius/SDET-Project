# shiftbench

A unified evaluation harness for two-sample tests under distribution shift.

Course project, *Statistical Detection and Estimation Theory*. Faculty mentor:
Prof. Praful.

---

## 1. Purpose

The project compares classical, model-based two-sample tests against modern
non-parametric and classifier-based tests, and identifies where each one fails.
Four workstreams are implementing different tests (Kolmogorov–Smirnov; z- and
t-tests; multiple-testing corrections; the generalised likelihood ratio test).
Results from those workstreams are only comparable if they are measured on
identical data, calibrated identically, and reported on identical axes.

This package provides that common substrate. It contributes three things:

1. **A reproducible benchmark.** Deterministic dataset splits and a faithful
   reimplementation of the synthetic shift catalogue of Rabanser, Günnemann and
   Lipton, *Failing Loudly* (NeurIPS 2019), with parameters taken from the
   authors' reference implementation rather than from the paper's prose — the
   two disagree, and Section 5 records how.
2. **A calibration-first protocol.** Every test is verified to hold its nominal
   false-alarm rate before any power number is reported. A test that rejects
   18 % of the time under the null will look powerful on every alternative;
   comparing it to a correctly calibrated test is not a comparison of power.
3. **A probe suite with known ground truth.** Seven synthetic distribution
   pairs, each breaking exactly one modelling assumption while holding
   everything else fixed, so that a test's failure can be attributed to a named
   assumption rather than described as poor performance.

---

## 2. Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later. No GPU is required and no neural network is trained; the
heaviest dependency is scikit-learn.

Datasets are cached outside the repository. Set the cache location before the
first run:

```bash
export SHIFTBENCH_DATA=/path/to/cache     # Windows: set SHIFTBENCH_DATA=C:\path\to\cache
```

If unset, the cache defaults to `ponnam/data/raw`, which is `.gitignore`d. On a
OneDrive-synchronised checkout, set it to a local path — the datasets total
roughly 250 MB and there is no reason to synchronise them.

---

## 3. Reproducing the results

The four studies must be run in this order; each depends on the one before it.

```bash
python scripts/00_verify_data.py        # download, split, write the manifest
python scripts/01_calibration.py        # Type-I error under a true null
python scripts/02_probe_power.py        # power curves on synthetic probes
python scripts/03_image_pipeline.py --dataset mnist
python scripts/04_latency.py            # wall-clock cost per test
python scripts/05_figures.py            # figures for the presentation
```

Every script accepts `--help`. Results are written to `results/` as Parquet
(raw, one row per replicate) and CSV (summarised).

Studies 1–3 parallelise across processes with `--jobs` (default: all cores,
`--jobs 1` for serial). Seeds are derived from the cell's own coordinates
rather than from a shared generator, so `--jobs 1` and `--jobs 16` produce
identical numbers.

Run `04_latency.py` on an otherwise idle machine; it pins itself to one core
and reports the minimum over repeated passes, but it cannot compensate for
fourteen other workers competing for cache.

### Two targeted studies

The GMM GLRT costs three EM fits per permutation, so it is not run over the
full grid. Its purpose is one specific claim — that a mixture model recovers
what a single-Gaussian model cannot see — and it is run only where that claim
lives:

```bash
python scripts/02_probe_power.py --probes mixture \
    --tests glrt_gauss glrt_gmm2 ks_bonferroni mmd c2st \
    --eps 0.5 0.7 0.9 --sizes 20 50 100 200 --replicates 50
```

The dimension sweep tests the Reddi et al. claim that kernel-test power decays
polynomially in dimension against a *fair* alternative — one whose intrinsic
difficulty is held fixed as `d` grows, which `mean_sparse` at constant `ε` is:

```bash
python scripts/02_probe_power.py --probes mean_sparse \
    --dims 2 8 32 128 512 --eps 0.8 --sizes 50 100 200
```

### Verifying that your data matches

`00_verify_data.py` writes `results/MANIFEST.json`, containing a SHA-256 digest
of each dataset split. The manifest is committed; the data is not, since it is
large and derived deterministically from this code. To confirm your copy
matches before contributing results:

```bash
python scripts/00_verify_data.py --check results/MANIFEST.json
```

A mismatch means a different seed, a different library version, or a corrupted
download. All three silently invalidate a cross-workstream comparison and none
are otherwise visible.

---

## 4. The unified specification

`configs/unified.yaml` is the contract between the workstreams. It fixes the
significance levels, the sample-size grid, the replicate counts, the
permutation budget, the reduction dimension and the dataset list. A result
produced against different values is not comparable and should not be placed on
a shared axis.

The settings that matter most:

| Setting | Value | Reason |
| --- | --- | --- |
| `alphas` | 0.01, 0.05, 0.10 | The paper fixes 0.05; reporting three makes the conservatism of a Bonferroni correction visible as a curve rather than a point. |
| `sample_sizes` | 10 … 1000 | The paper's grid. The lower end is the regime the project's central claim concerns. |
| `n_permutations` | 500 | Bounds the smallest attainable p-value at 1/501 ≈ 0.002, below the smallest reported α. |
| `calibration_replicates` | 500 | Binomial standard error at α = 0.05 is then ≈ 0.010, enough to resolve a genuine miscalibration. |
| `reduction_k` | 32 | The paper's setting; 32 principal components explain roughly 80 % of CIFAR-10's variance. |
| `match_source_target` | true | See Section 5.3. |

---

## 5. Fidelity to the source material

The benchmark reproduces *Failing Loudly* where doing so is correct, and
departs from it where the reference implementation is misleading. Every
departure is listed here.

### 5.1 The paper's prose understates the image shifts

Section 4 of the paper describes the image shift as random rotations
`{10, 40, 90}`, translation percentages `{0.05, 0.2, 0.4}` and zoom percentages
`{0.1, 0.2, 0.4}`. The reference implementation additionally applies a shear of
`{0.1, 0.2, 0.3}`, a random **horizontal flip** at medium intensity, and both
horizontal and vertical flips at large intensity.

This is not a cosmetic difference. A horizontal flip is a far more aggressive
perturbation of MNIST than a 40-degree rotation, so an implementation following
only the prose measures a substantially weaker shift and will report lower
detection rates for every method. This package follows the code.

### 5.2 Gaussian-noise intensities are in raw 0–255 units

The quoted standard deviations `σ ∈ {1, 10, 100}` are divided by 255 before
being added to data already scaled to `[0, 1]`, and the sum is clipped back to
`[0, 1]`. Effective standard deviations are therefore approximately
`{0.0039, 0.0392, 0.3922}`.

The clipping matters more than the scaling. At `σ = 100` a large fraction of
pixels saturate at exactly 0 or 1, so the shifted marginal is strongly bimodal
with atoms at both boundaries. That boundary mass, not the added variance, is
what most tests actually detect — and it is also what breaks the continuity
assumption underlying the KS test's null distribution.

### 5.3 Source and target sample sizes are matched here, and are not in the reference

The reference implementation passes the **full** source sample to the
univariate tests but subsamples the source to the target's size before the
kernel test. A KS test therefore runs at 10 000 against *n*, while MMD runs at
*n* against *n*.

That asymmetry flatters the univariate family at precisely the small *n* this
project is about. Both sides are matched here. The `source_n` column records
the choice, so the other convention can be reproduced for comparison.

### 5.4 The MMD bandwidth is not the standard median heuristic

The reference implementation calls `torch_two_sample` with
`alphas=[1/median_dist]`, where `median_dist` is the median *Euclidean*
distance, and that library's kernel is `exp(-α‖x−y‖²)`. The effective kernel is
therefore `exp(-‖x−y‖²/median_dist)`.

The standard median heuristic is `exp(-‖x−y‖²/(2σ²))` with `σ²` the median of
the *squared* distances. Since the median squared distance is roughly the
square of the median distance, the paper's bandwidth is wider by a factor of
order `median_dist` — about tenfold on normalised CIFAR-10.

Both are implemented (`mmd` and `mmd_median_sq`) because the difference is
large enough to move power curves. On the `mixture` probe at n = 100, d = 16,
the paper's bandwidth returns p = 0.79 where the standard heuristic returns
p = 0.07.

### 5.5 Omissions

* **The adversarial (FGSM) shift is not implemented.** It requires a trained
  neural network to generate the adversarial examples. Every other shift in the
  catalogue is reproduced.
* **The two autoencoder reductions (`UAE`, `TAE`) are not implemented.** Their
  only role is to produce a 32-dimensional embedding, and PCA already provides a
  linear one to compare against.
* **`BBSDs`/`BBSDh` use multinomial logistic regression, not ResNet-18.** The
  mechanism under test is "project onto a label classifier's outputs", and a
  convex classifier exercises it without a training budget. The reduction is
  consequently weaker than the paper's, so BBSD results here are a lower bound
  on what the paper's version achieves, not a reproduction of it.

The last three omissions follow the project brief's direction that training a
model is not in itself recognised as progress.

---

## 6. Package layout

```
ponnam/
├── shiftbench/
│   ├── datasets.py        Deterministic loading and splitting; split digests
│   ├── _download.py       Fast mirrors (the canonical ones throttle to ~15 kB/s)
│   ├── shifts.py          The Failing Loudly shift catalogue, exact parameters
│   ├── synthetic.py       Seven probes with known ground truth
│   ├── reductors.py       NoRed, PCA, SRP, BBSDs, BBSDh
│   ├── tests/
│   │   ├── base.py        TwoSampleTest interface; permutation machinery;
│   │   │                  Bonferroni / Holm / Simes / Fisher combiners
│   │   ├── univariate.py  KS, Welch-t, z, Anderson–Darling, Mann–Whitney
│   │   ├── parametric.py  Hotelling T², diagonal T², Gaussian GLRT,
│   │   │                  GMM GLRT, Chen–Qin
│   │   ├── nonparametric.py  MMD, energy distance
│   │   └── classifier.py  C2ST
│   └── experiment.py      Calibration, probe-power and image-pipeline drivers
├── scripts/               The five numbered entry points
├── configs/unified.yaml   The cross-workstream contract
└── results/               Outputs (figures and tables; data is gitignored)
```

### Adding a test

Subclass `TwoSampleTest`, implement `_run(x, y, rng) -> TestResult`, and
register it in `shiftbench/tests/__init__.py`. It is then automatically
available to every script and every figure. The interface is the only
requirement:

```python
from shiftbench.tests.base import TwoSampleTest, TestResult, permutation_pvalue

class MyTest(TwoSampleTest):
    name = "my_test"

    def _run(self, x, y, rng):
        p, stat, _ = permutation_pvalue(x, y, my_statistic, 500, rng)
        return TestResult(p, stat, self.name)
```

Any new test must clear `scripts/01_calibration.py` before its power numbers
are used.

---

## 7. Test inventory and ownership

| Owner | Registry keys |
| --- | --- |
| KS | `ks_bonferroni`, `ks_holm`, `ks_simes`, `ks_fisher` |
| z / t | `z_bonferroni`, `t_bonferroni`, `t_simes` |
| Corrections | `ad_bonferroni`, `mwu_bonferroni`, and every `*_holm` / `*_simes` / `*_fisher` pairing |
| GLRT | `glrt_gauss`, `glrt_gauss_asym`, `glrt_gmm2`, `glrt_gmm3` |
| Shared comparators | `hotelling`, `hotelling_perm`, `hotelling_ridge`, `diag_hotelling`, `chen_qin`, `mmd`, `mmd_median_sq`, `energy`, `c2st`, `c2st_knn` |

---

## 8. The probe suite

Each probe is a pair of distributions parameterised by an effect size `ε` and a
dimension `d`. Setting `ε = 0` draws both samples from the same distribution,
which is what the calibration study consumes.

| Probe | Construction | Structurally blind to |
| --- | --- | --- |
| `mean_sparse` | Mean shift in one coordinate | — |
| `mean_dense` | Same total shift norm spread over all coordinates | — (but Bonferroni-corrected marginals lose heavily) |
| `variance` | Equal means, scaled covariance | every mean-only test: t, z, Hotelling, Chen–Qin |
| `correlation` | Identical marginals, equicorrelated joint | **every marginal test** (KS, t, z, AD, MWU) *and* every mean-only test |
| `mixture` | Two-component mixture matched to `N(0, I)` in mean **and** covariance | t, z, Hotelling, Chen–Qin, Gaussian GLRT |
| `heavy_tail` | Student-t rescaled to unit variance | mean-only tests |
| `skew` | Skewed with mean and variance held fixed | mean-only and variance-only tests |

"Blind" is a structural claim, not an empirical one: the quantity the test
examines is identical under `p` and `q` by construction, so no sample size
rescues it. A flat power curve sitting at the nominal level confirms the
construction rather than indicating a bug.

`correlation` and `mixture` are the two that carry the argument.

- **`correlation`** is the case no multiple-testing correction can rescue. The
  evidence is not present in any one-dimensional marginal, so the entire
  multiple-univariate family has a ceiling that is structural rather than a
  matter of power. Confirmed empirically: at n = 100, d = 16, ε = 0.8, the
  Gaussian GLRT, MMD and the energy test all return p ≈ 0.002, while KS,
  Welch-t, z, Hotelling and Chen–Qin all return p > 0.08.
- **`mixture`** separates "parametric" from "Gaussian". The first two moments
  match exactly, so a single-Gaussian GLRT is blind by construction while a
  two-component GMM GLRT has the right model class.

---

## 9. Methodological commitments

**Permutation calibration by default.** Under `H₀` the pooled sample is
exchangeable, so permuting group assignments gives an exact finite-sample null
for any statistic, with no distributional assumption. This matters here because
the asymptotic nulls of several classical tests are visibly wrong at n = 10.

**The `+1` correction.** Permutation p-values are computed as
`(1 + #{T_perm ≥ T_obs}) / (1 + B)`, not `#{T_perm ≥ T_obs} / B`. The naive form
can return exactly zero, which is not a valid p-value and makes the test
anti-conservative. This is the difference between a test that holds its nominal
level and one that does not, and it is verified empirically rather than assumed.

**Reduction is fitted on training data only.** The source and target samples are
drawn from the validation and test splits respectively; the reduction never sees
the data it will be used to compare.

**Two kinds of "cannot compute", recorded differently.** They are not the same
thing and merging them would corrupt the power estimates in opposite directions.

*The test does not exist at this `(n, d)`.* Hotelling's `T²` requires inverting
a pooled covariance that is a sum of `n₁+n₂−2` rank-one terms, so it is singular
the moment `d ≥ n₁+n₂−2`. A practitioner who reaches for it there gets nothing.
That is genuinely **zero power**, so it is recorded as a non-rejection and
appears on the power curve as a flat line at zero — which is the finding, not a
gap in the data. The `singular` flag records why.

*We declined to spend the compute.* The permutation-calibrated Gaussian GLRT
costs `O(B·d³)`; at `d = 784` with `B = 500` that is of order 10¹² floating-point
operations per cell. The GMM GLRT needs three EM fits per permutation. These are
budget decisions of ours, not properties of the test, so those cells are flagged
`skipped`, carry `p_value = NaN`, and are **excluded** from rejection rates
rather than averaged in as zeros. Counting our own budget limit as a failure to
detect would understate the method.

**Detection after reduction is a weaker claim than it appears.** Every reduction
maps `D` dimensions onto `K ≪ D` and is therefore surjective, so each is capable
of mapping a genuinely shifted input distribution onto an unshifted latent one.
What is being tested is `p(f(x)) = q(f(x))`, not `p(x) = q(x)`. The paper argues
this is empirically benign outside adversarial settings; it remains an
assumption and is stated rather than glossed.

---

## 10. Current status

Implemented and verified:

- Dataset loading, deterministic splitting, digest manifest
- Full shift catalogue (28 shifts) less the adversarial shift
- Seven synthetic probes
- Twenty-three registered tests, all passing the calibration check
- Calibration, probe-power, image-pipeline and latency drivers
- Figure generation

Calibration results at n = 50, d = 16 under a Gaussian null, 300 replicates:
every permutation-calibrated test holds its nominal level within three binomial
standard errors. Two tests are *conservative* — `ks_fisher` (0.007 at α = 0.05)
and `glrt_gauss_asym` (0.007 at α = 0.05). Both under-reject, which costs power
but does not invalidate the false-alarm guarantee. The causes are identified:
Fisher's method compounds the discreteness-induced conservatism of small-sample
KS p-values, and the asymptotic GLRT's χ² reference distribution is wrong at
these sample sizes given the shrinkage applied to keep the covariance estimates
non-singular.

---

## 11. References

1. S. Rabanser, S. Günnemann, Z. C. Lipton. *Failing Loudly: An Empirical Study
   of Methods for Detecting Dataset Shift.* NeurIPS 2019. arXiv:1810.11953.
   Reference implementation: <https://github.com/steverab/failing-loudly>
2. A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola. *A Kernel
   Two-Sample Test.* JMLR 13 (2012), 723–773.
3. D. Lopez-Paz, M. Oquab. *Revisiting Classifier Two-Sample Tests.* ICLR 2017.
   arXiv:1610.06545.
4. S. X. Chen, Y.-L. Qin. *A Two-Sample Test for High-Dimensional Data with
   Applications to Gene-Set Testing.* Annals of Statistics 38(2), 2010.
   arXiv:1002.4547.
5. S. J. Reddi, A. Ramdas, B. Póczos, A. Singh, L. Wasserman. *On the Decreasing
   Power of Kernel and Distance Based Nonparametric Hypothesis Tests in High
   Dimensions.* AAAI 2015. arXiv:1406.2083.
6. F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, D. J. Sutherland. *Learning Deep
   Kernels for Non-Parametric Two-Sample Tests.* ICML 2020. arXiv:2002.09116.
7. Z. C. Lipton, Y.-X. Wang, A. Smola. *Detecting and Correcting for Label Shift
   with Black Box Predictors.* ICML 2018. arXiv:1802.03916.
