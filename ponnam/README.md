# Generalised Likelihood Ratio Tests for Distribution Shift

Course project, *Statistical Detection and Estimation Theory*.
Faculty mentor: Prof. Praful.

---

## 1. Objective

A model trained on one distribution eventually sees data drawn from another.
Deciding whether drift has occurred is a two-sample hypothesis test: given a
reference sample and a new sample, determine whether they arise from the same
distribution.

Standard practice reaches for non-parametric tests — maximum mean discrepancy,
classifier two-sample tests — because the true distribution is unknown. That
generality is not free: it costs statistical power, and it scales poorly in
dimension and sample size. Real data frequently follows an approximately known
structure, and a parametric test built on that structure can detect shift from
substantially fewer samples.

This work implements the **generalised likelihood ratio test** as the general
recipe for constructing such parametric tests, and quantifies when the
assumption pays for itself. Non-parametric methods are retained as baselines,
not as the subject of study.

---

## 2. The generalised likelihood ratio test

For a parametric family `{f(·; θ) : θ ∈ Θ}` and a null restriction `Θ₀ ⊂ Θ`,

$$\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta; Z)}{\sup_{\theta \in \Theta} L(\theta; Z)}, \qquad G = -2\log\Lambda.$$

In the two-sample problem the null asserts that both samples share one
parameter and the alternative allows each its own, giving

$$G = 2\left[\ell(\hat\theta_X; X) + \ell(\hat\theta_Y; Y) - \ell(\hat\theta_0; X \cup Y)\right].$$

Wilks' theorem gives `G → χ²_r` under `H₀`, where `r` is the difference in free
parameter count between the two models — **provided the regularity conditions
hold**. Section 5 reports what happens when they do not.

### 2.1 The five model cases

| Case | Model | Free parameters under `H₁` | Wilks `r` | Defined for `d ≥ n`? |
| --- | --- | --- | --- | --- |
| **A** | `N(μ, Σ)`, `Σ` known | `μ` per sample | `d` | yes |
| **B** | `N(μ, Σ)`, `Σ` unknown but common | `μ` per sample | `d` | **no** |
| **C** | `N(μ, diag(σ²))` | `μ, σ²` per sample | `2d` | yes |
| **D** | `N(μ, Σ)`, both free | `μ, Σ` per sample | `d + d(d+1)/2` | no |
| **E** | Gaussian mixture, `k` components | all mixture parameters | **not valid** | no |

Each case is a different answer to "what do we assume the data looks like".
Moving down the table weakens the assumption and increases the number of
parameters that must be estimated — which is precisely the trade the project
measures.

### 2.2 Two facts that drive the results

**Case B is Hotelling's `T²`.** The likelihood ratio for a mean shift under a
common unknown covariance satisfies

$$\Lambda^{2/(n+m)} = \left(1 + \frac{T^2}{n+m-2}\right)^{-1},$$

so `G = (n+m)·log(1 + T²/(n+m−2))` is a strictly increasing function of `T²`.
**The GLRT and Hotelling's test are the same test** — identical rejection
regions, identical power. They differ only in which reference distribution sets
the threshold: Wilks gives the asymptotic `χ²_d`, while Hotelling's exact result

$$\frac{n+m-d-1}{d\,(n+m-2)}\,T^2 \sim F_{d,\;n+m-d-1}$$

holds for every `n` under normality. Where both exist, the exact `F` should be
preferred. The registry exposes all three calibrations of this one statistic
(`glrt_b_mean`, `glrt_b_mean_asym`, `glrt_b_mean_exactf`) so the claim is
measured rather than asserted.

**Case E has no valid asymptotic null.** Under `H₀` the alternative's extra
mixture components are unidentifiable and the null parameter lies on the
boundary of the alternative's parameter space. Wilks' regularity conditions
fail, and `G` is not asymptotically `χ²_r` for any `r`. Quoting a chi-squared
p-value for a mixture likelihood ratio is a standard and serious error, so
`GMMGLRT` does not accept an `asymptotic` calibration argument at all — the
mistake is made structurally impossible rather than merely discouraged.

---

## 3. Calibration

Every case supports two calibrations.

**Asymptotic.** The Wilks `χ²_r` reference. Closed form and immediate, and an
*approximation* whose finite-sample quality is exactly what must be checked
before it is trusted.

**Permutation (default).** Under `H₀` the pooled sample is exchangeable, so
every assignment of `n` of the `n+m` rows to the first group is equally likely.
Drawing `B` such assignments yields an exact finite-sample null for any
statistic, under no distributional assumption and regardless of whether the
model is correctly specified.

P-values use

$$p = \frac{1 + \#\{G_{\text{perm}} \ge G_{\text{obs}}\}}{1 + B}$$

rather than the naive ratio. The naive form can return exactly zero, which is
not a valid p-value and renders the test anti-conservative; the corrected form
is provably valid at any level for any `B`. Verified empirically: at `B = 49`
the measured `P(p ≤ 0.05) = 0.0415` with the p-value floor at exactly
`1/(B+1) = 0.0200`.

---

## 4. Installation and use

```bash
pip install -r requirements.txt
```

Python 3.10 or later. No GPU, and no neural network is trained.

Datasets are cached outside the repository:

```bash
export SHIFTBENCH_DATA=/path/to/cache     # Windows: set SHIFTBENCH_DATA=C:\path\to\cache
```

If unset, the cache defaults to `ponnam/data/raw`, which is ignored by git. On
a OneDrive-synchronised checkout, set it to a local path; the datasets total
roughly 380 MB.

### Reproduction

The studies must be run in order; each depends on the one before it.

```bash
python scripts/00_verify_data.py        # download, split, write the digest manifest
python scripts/01_calibration.py        # Type-I error under a true null
python scripts/02_probe_power.py        # power curves on synthetic probes
python scripts/03_image_pipeline.py --dataset mnist
python scripts/04_latency.py            # wall-clock cost per test
python scripts/05_figures.py            # figures
```

Every script accepts `--help`. Studies 1–3 parallelise with `--jobs` (default:
all cores). Per-cell seeds are derived from the cell's own coordinates rather
than from a shared generator, so `--jobs 1` and `--jobs 16` produce identical
numbers. Run `04_latency.py` on an otherwise idle machine; it pins itself to a
single core and reports the minimum over repeated passes.

### Verifying that your data matches

`00_verify_data.py` writes `results/MANIFEST.json`, holding a SHA-256 digest per
dataset split. The manifest is committed; the data is not, since it is large and
derived deterministically from this code.

```bash
python scripts/00_verify_data.py --check results/MANIFEST.json
```

A mismatch indicates a different seed, a different library version, or a
corrupted download — each of which silently invalidates a cross-workstream
comparison and none of which is otherwise visible.

---

## 5. Result: the asymptotic null must be verified, not assumed

The project brief directs that the GLRT's theoretical chi-squared behaviour be
verified empirically before it is relied upon. It does not hold.

Measured over 500 replicates per cell, at nominal `α = 0.05`, case D (`d = 32`):

| Calibration | `n = 20` | `n = 50` | `n = 200` |
| --- | --- | --- | --- |
| Asymptotic `χ²` | **0.198** | 0.055 | **0.000** |
| Permutation | 0.053 | 0.045 | 0.049 |

This is the **same statistic on the same data**; only the reference
distribution differs. The asymptotic version rejects four times too often at
`n = 20` and never at `n = 200`. It does not merely drift — it crosses the
nominal level from above to below as `n` grows, so no single correction factor
repairs it. The permutation calibration holds its level throughout.

Two mechanisms are responsible, and they pull in opposite directions. Each
per-sample covariance MLE has rank at most `n − 1`, so at `d = 32, n = 20` the
log-determinants diverge and must be stabilised by shrinkage toward a scaled
identity. The shrunk statistic is no longer the exact likelihood ratio, so
Wilks no longer applies even asymptotically. At small `n` the rank deficiency
inflates `G` relative to `χ²₅₂₈`; at large `n` the shrinkage bias dominates and
deflates it.

A corollary, relevant to the z/t workstream: the univariate z-test — case A's
assumption that the variance is known, applied per coordinate — rejects at
**0.093** against a nominal 0.05 at `n = 20`, converging to the t-test's 0.050
by `n = 50`. That gap is the cost of the known-variance assumption, measured.

---

## 6. The synthetic probe suite

Image shifts are realistic but opaque: when a test fails on `large_img_shift`
it is not evident which assumption was violated. The probes are the opposite —
each breaks exactly one assumption with everything else held fixed.

Each is parameterised by an effect size `ε` and dimension `d`. Setting `ε = 0`
draws both samples from the same distribution, which is what the calibration
study consumes.

| Probe | Construction | Structurally blind |
| --- | --- | --- |
| `mean_sparse` | mean shift in one coordinate | — |
| `mean_dense` | same shift norm spread over all coordinates | — (Bonferroni-corrected marginals lose heavily) |
| `variance` | equal means, scaled covariance | GLRT A, B; Hotelling; Chen–Qin |
| `correlation` | identical marginals, equicorrelated joint | GLRT A, B, C; **every** marginal test; Hotelling; Chen–Qin |
| `mixture` | two-component mixture matched to `N(0,I)` in mean **and** covariance | GLRT A, B, C, D |
| `heavy_tail` | Student-t rescaled to unit variance | mean-only tests |
| `skew` | skewed with mean and variance fixed | mean-only and variance-only tests |

"Blind" is structural, not empirical: the quantity the test examines is
identical under `p` and `q` by construction, so no sample size rescues it. A
flat power curve at the nominal level confirms the construction.

Two probes carry the argument, and both are verified numerically:

- **`correlation`** has unit marginal variances with off-diagonal correlation
  0.800, against 0.008 for the reference. Identical marginals, different joint.
  This is the ceiling on cases A–C and on the entire per-coordinate family —
  the evidence is not present in any one-dimensional marginal, so no
  multiplicity correction can recover it. Case D, which models the full
  covariance, detects it.
- **`mixture`** matches `N(0, I)` in mean and variance to within 1 % at
  `ε = 0.8`. Cases A–D are therefore blind by construction however much data
  they receive; case E has the correct model class.

Together these establish the project's central structure: **each GLRT case is
exactly blind to the shifts its model cannot express**, and the hierarchy
A ⊂ B ⊂ C ⊂ D ⊂ E buys sensitivity to successively more general alternatives at
the cost of estimating successively more parameters.

---

## 7. Benchmark fidelity

The real-data benchmark reproduces Rabanser, Günnemann and Lipton, *Failing
Loudly* (NeurIPS 2019), with parameters taken from the authors' reference
implementation rather than the paper's prose. The two disagree; every departure
is recorded here.

**The prose understates the image shifts.** Section 4 lists rotations
`{10, 40, 90}`, translations `{0.05, 0.2, 0.4}` and zooms `{0.1, 0.2, 0.4}`. The
code additionally applies shear `{0.1, 0.2, 0.3}`, a random horizontal flip at
medium intensity, and both flips at large intensity. A horizontal flip is a far
stronger perturbation of MNIST than a 40-degree rotation, so an implementation
following only the prose measures a materially weaker shift. This package
follows the code.

**Gaussian-noise intensities are in raw 0–255 units.** The quoted
`σ ∈ {1, 10, 100}` is divided by 255 before being added to data already scaled
to `[0,1]`, and the result is clipped back to `[0,1]`. Effective standard
deviations are approximately `{0.0039, 0.0392, 0.3922}`. The clipping matters
more than the scaling: measured on MNIST at `σ = 100`, the fraction of
exactly-zero pixels falls from 0.809 to 0.424 while saturation at 1 rises from
0.007 to 0.060, and the mean rises from 0.131 to 0.247. The detectable signal is
boundary mass, not added variance — and that same clipping violates the
continuity assumption underlying the KS test's null distribution.

**Source and target sample sizes are matched here; they are not in the
reference.** That code passes the full source sample to the univariate tests but
subsamples the source to the target's size before the kernel test, so KS runs at
10 000 against `n` while MMD runs at `n` against `n`. The asymmetry flatters the
univariate family at exactly the small `n` this project concerns. Both sides are
matched here, and the `source_n` column records the choice.

**The MMD bandwidth is not the standard median heuristic.** The reference calls
`torch_two_sample` with `alphas=[1/median_dist]` against a kernel
`exp(−α‖x−y‖²)`, giving `exp(−‖x−y‖²/median_dist)`. The standard heuristic is
`exp(−‖x−y‖²/(2σ²))` with `σ²` the median *squared* distance — roughly a tenfold
difference in width on normalised data. Both are implemented (`mmd`,
`mmd_median_sq`) because the difference moves conclusions: on the `mixture`
probe at `n = 100, d = 16`, the reference bandwidth returns `p = 0.79` where the
standard one returns `p = 0.07`.

**Omissions.** The adversarial (FGSM) shift requires a trained network and is
not implemented; every other shift in the catalogue is. The two autoencoder
reductions are omitted, PCA providing a linear 32-dimensional embedding to
compare against. `BBSDs`/`BBSDh` use multinomial logistic regression rather than
ResNet-18, so BBSD results here are a lower bound on the paper's rather than a
reproduction of them.

---

## 8. Recording conventions

**Two kinds of "cannot compute", recorded differently.**

*The test does not exist at this `(n, d)`.* Case B requires inverting a pooled
covariance that is a sum of `n+m−2` rank-one terms, so it is singular once
`d ≥ n+m−2`. A practitioner reaching for it there obtains nothing. That is
genuinely **zero power**, recorded as a non-rejection and appearing on the power
curve as a flat line at zero — the finding, not a gap. The `singular` flag
records why.

*The compute was declined.* Case D's permutation test costs `O(B·d³)`; at
`d = 784` with `B = 500` that is of order 10¹² floating-point operations per
cell. Case E needs three EM fits per permutation. These are budget decisions,
not properties of the test, so such cells are flagged `skipped`, carry
`p_value = NaN`, and are **excluded** from rejection rates rather than averaged
in as zeros.

**Reductions are fitted on training data only.** The source and target samples
are drawn from the validation and test splits; the reduction never sees the data
it will be used to compare.

**Detection after reduction is a weaker claim than it appears.** Every reduction
maps `D` dimensions onto `K ≪ D` and is therefore surjective, so each can map a
genuinely shifted input distribution onto an unshifted latent one. What is
tested is `p(f(x)) = q(f(x))`, not `p(x) = q(x)`.

---

## 9. Package layout

```
ponnam/
├── shiftbench/
│   ├── tests/
│   │   ├── glrt.py            Cases A–E, both calibrations        [subject]
│   │   ├── base.py            TwoSampleTest interface; permutation
│   │   │                      machinery; Bonferroni/Holm/Simes/Fisher
│   │   ├── parametric.py      Hotelling T², diagonal T², Chen–Qin  [comparators]
│   │   ├── univariate.py      KS, Welch-t, z, Anderson–Darling,
│   │   │                      Mann–Whitney                         [comparators]
│   │   ├── nonparametric.py   MMD, energy distance                 [baselines]
│   │   └── classifier.py      C2ST                                 [baselines]
│   ├── synthetic.py           Seven probes with known ground truth
│   ├── datasets.py            Deterministic loading, splitting, digests
│   ├── _download.py           Fast mirrors (canonical ones throttle to ~15 kB/s)
│   ├── shifts.py              Failing Loudly catalogue, exact parameters
│   ├── reductors.py           NoRed, PCA, SRP, BBSDs, BBSDh
│   └── experiment.py          Calibration, probe-power, image-pipeline drivers
├── scripts/                   Six numbered entry points
├── configs/unified.yaml       Cross-workstream specification
└── results/                   Figures, summary tables, manifest
```

### Adding a GLRT case

Subclass `GLRT`, supply `_statistic` and `_wilks_df`, and register the variant:

```python
from shiftbench.tests.glrt import GLRT

class MyGLRT(GLRT):
    name = "glrt_mine"

    def _wilks_df(self, d):
        return float(d)                       # free parameters gained under H1

    def _statistic(self, x, y):
        return 2.0 * (ll(x) + ll(y) - ll(np.concatenate([x, y])))
```

Set `wilks_df_valid = False` if the regularity conditions fail; constructing the
class with `calibration="asymptotic"` will then raise rather than return an
invalid p-value. Any new case must clear `scripts/01_calibration.py` before its
power numbers are used.

---

## 10. Test registry

| Role | Keys |
| --- | --- |
| **GLRT (subject)** | `glrt_a_knowncov`, `glrt_b_mean`, `glrt_c_diag`, `glrt_d_meancov`, `glrt_e_gmm2`, `glrt_e_gmm3`, each with an `_asym` counterpart where Wilks applies, plus `glrt_b_mean_exactf` |
| Classical comparators | `hotelling`, `hotelling_ridge`, `diag_hotelling`, `chen_qin`, `ks_bonferroni`, `ks_holm`, `ks_simes`, `ks_fisher`, `t_bonferroni`, `t_simes`, `z_bonferroni`, `ad_bonferroni`, `mwu_bonferroni` |
| Non-parametric baselines | `mmd`, `mmd_median_sq`, `energy`, `c2st`, `c2st_knn` |

---

## 11. References

1. S. Rabanser, S. Günnemann, Z. C. Lipton. *Failing Loudly: An Empirical Study
   of Methods for Detecting Dataset Shift.* NeurIPS 2019. arXiv:1810.11953.
   Reference implementation: <https://github.com/steverab/failing-loudly>
2. S. S. Wilks. *The Large-Sample Distribution of the Likelihood Ratio for
   Testing Composite Hypotheses.* Annals of Mathematical Statistics 9(1), 1938.
3. H. Hotelling. *The Generalization of Student's Ratio.* Annals of Mathematical
   Statistics 2(3), 1931.
4. A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola. *A Kernel
   Two-Sample Test.* JMLR 13 (2012), 723–773.
5. D. Lopez-Paz, M. Oquab. *Revisiting Classifier Two-Sample Tests.* ICLR 2017.
   arXiv:1610.06545.
6. S. X. Chen, Y.-L. Qin. *A Two-Sample Test for High-Dimensional Data with
   Applications to Gene-Set Testing.* Annals of Statistics 38(2), 2010.
   arXiv:1002.4547.
7. S. J. Reddi, A. Ramdas, B. Póczos, A. Singh, L. Wasserman. *On the Decreasing
   Power of Kernel and Distance Based Nonparametric Hypothesis Tests in High
   Dimensions.* AAAI 2015. arXiv:1406.2083.
