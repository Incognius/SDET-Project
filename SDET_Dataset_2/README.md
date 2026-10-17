# SDET_Dataset_2: which test catches which shift?

## The question

We have a reference sample X and a new sample Y. Did the distribution change? We compare three classical tests on three simple kinds of shift.

## The data

- Reference: X ~ N(0, I) in d = 10 dimensions.
- New sample Y, with one thing changed:

| Shift | What changes | Levels |
|---|---|---|
| **Mean** | every feature's mean moves by a/√10 | a = 0 … 1.6 |
| **Variance** | every feature's std is multiplied by s | s = 1 … 2 |
| **Mean, heavy-tailed** | same mean shift, but both samples use Student-t₃ noise (not Gaussian) | a = 0 … 1.6 |

- a = 0 and s = 1 mean "no shift", which is how we measure false alarms.
- Each setting is repeated 100 times, with n = 10, 20, 50 or 100 samples per group.
- The data comes from `generate.py`, fixed seed, saved in `data/synthetic/`.

## The tests (α = 0.05)

| Test | Idea | Checks |
|---|---|---|
| **t-test** | compares the averages of each feature | mean only |
| **KS test** | compares the full distribution of each feature | anything, one feature at a time |
| **GLRT** | likelihood ratio of a Gaussian model, compared against χ²(20) | mean and variance, assumes Gaussian |

The t-test and KS run on each of the 10 features separately, and flag a shift if **any** feature has p < 0.05. A z-test gives almost the same results as the t-test once n ≥ 50, so it is left out.

## Results (n = 100)

| | No shift (false alarm) | Mean a = 0.8 | Variance s = 1.2 | Heavy-tailed, no shift |
|---|---|---|---|---|
| t-test | 0.48 | 1.00 | 0.48 | 0.38 |
| KS | 0.35 | 0.96 | 0.52 | 0.29 |
| GLRT | **0.05** | 0.97 | **0.96** | 0.93 |

**1. Testing 10 features separately means about 40% false alarms.** Each feature has a 5% chance of a false alarm, so the chance that at least one of 10 does is 1 − 0.95¹⁰ ≈ 0.40 (`results/figures/01_false_alarms.png`).

**2. The GLRT works when its assumption holds.** On Gaussian data it gives 5% false alarms as promised. It catches variance shifts that the t-test cannot see at all (0.96 vs 0.48 at s = 1.2), with KS in between (`02_detection.png`).

**3. The GLRT breaks when its assumption is wrong.** On heavy-tailed data its statistic no longer follows χ²(20) (`03_glrt_vs_chi2.png`), so it raises a false alarm 93% of the time. KS makes no distribution assumption and is the most stable of the three.

**Takeaway:** a parametric test (GLRT) beats the simple tests when the data really is close to Gaussian, and fails badly when it is not.

## Reproduce

```
python generate.py --only synthetic   # data
python run_tests.py                   # results/rejection_rates.csv
python make_figures.py                # results/figures/
```
