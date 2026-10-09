"""
================================================================================
Two-Sample z/t-Test for Distribution-Shift Detection  —  Synthetic-Data Study
================================================================================

Project thesis (from the getting-started brief):
  When production data roughly follows a known structure, a CLASSICAL PARAMETRIC
  test that assumes that structure can detect a distribution shift with far
  fewer samples / lower latency than a generic NONPARAMETRIC test.

This script builds every core hypothesis-testing concept on the simplest case —
a MEAN SHIFT detected by the two-sample z/t-test — and then runs all six
"directions to explore" on synthetic Gaussian data, with the KS test (and, for
the multivariate parts, Hotelling's T^2) as nonparametric / parametric
baselines.

Run:
    python3 ztest_shift_detection.py                 # everything
    python3 ztest_shift_detection.py --only 0 1      # just a couple of sections
    python3 ztest_shift_detection.py --fast          # fewer Monte-Carlo trials

Sections:
    0  Core concepts: z-test, t-test, p-value, Type-I error, power
    1  Efficiency      — samples needed to match power
    2  Misspecification— how non-Gaussian can data get before the t-test breaks
    3  Latency         — sequential parametric test vs repeated batch KS
    4  Combination     — per-dim t-tests + Bonferroni vs Hotelling's T^2
    5  Compression     — PCA / LDA-direction reduction vs per-dimension
    6  Random proj.    — t-test on random linear combinations + Bonferroni

Every section prints progress and saves a figure into ./figures/.
================================================================================
"""

import argparse
import time
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # headless; we save PNGs rather than display
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------------------
ALPHA = 0.05                 # significance level (false-alarm budget)
RNG_SEED = 20261009
FIG_DIR = "figures"

# Monte-Carlo trial counts. --fast shrinks these for a quick sanity run.
N_TRIALS = 2000
N_TRIALS_FAST = 400


def log(msg, t0=None):
    """Timestamped progress line. Pass t0 to also print elapsed seconds."""
    if t0 is not None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}  (+{time.time()-t0:.1f}s)",
              flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==============================================================================
# TEST PRIMITIVES
# ==============================================================================
# Each returns a p-value under H0: "the two samples share the quantity tested".
# A SHIFT is declared when p < alpha.

def z_test(x, y, sigma):
    """Two-sample z-test for a mean shift, KNOWN common std `sigma`.

    Statistic  z = (xbar - ybar) / sqrt(sigma^2/nx + sigma^2/ny)  ~ N(0,1) under H0.
    Two-sided p-value from the standard Normal.
    """
    nx, ny = len(x), len(y)
    se = sigma * np.sqrt(1.0 / nx + 1.0 / ny)
    z = (x.mean() - y.mean()) / se
    p = 2.0 * stats.norm.sf(abs(z))      # sf = 1 - cdf, the upper tail
    return z, p


def welch_t_test(x, y):
    """Welch's two-sample t-test (variance UNKNOWN, not assumed equal).

    The realistic classical test. scipy handles the Welch-Satterthwaite dof.
    """
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return t, p


def ks_test(x, y):
    """Two-sample Kolmogorov-Smirnov test — the nonparametric baseline.

    Detects ANY distributional difference (not just the mean) by comparing
    empirical CDFs. No distributional assumption; that generality costs power.
    """
    d, p = stats.ks_2samp(x, y)
    return d, p


def hotelling_t2(X, Y):
    """Hotelling's T^2 — the MULTIVARIATE parametric generalisation of the t-test.

    Tests H0: mean vectors equal, using the pooled covariance so it exploits
    cross-dimension correlation structure. Returns an F-based p-value.
    X, Y: arrays of shape (n_samples, n_dims).
    """
    nx, ny = X.shape[0], Y.shape[0]
    p_dim = X.shape[1]
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    # pooled covariance
    Sp = ((nx - 1) * Sx + (ny - 1) * Sy) / (nx + ny - 2)
    Sp_inv = np.linalg.pinv(Sp)          # pinv for numerical safety
    diff = (mx - my)
    t2 = (nx * ny) / (nx + ny) * diff @ Sp_inv @ diff
    # convert to F statistic
    df1 = p_dim
    df2 = nx + ny - 1 - p_dim
    if df2 <= 0:
        return np.nan, 1.0               # not enough samples for the dimension
    f_stat = t2 * df2 / (df1 * (nx + ny - 2))
    p = stats.f.sf(f_stat, df1, df2)
    return t2, p


def bonferroni_pvalue(pvals):
    """Combine many per-dimension p-values into one decision via Bonferroni.

    Reject the global H0 if ANY dimension's p < alpha/m. Equivalently the
    combined p-value is min(p)*m clipped to 1. Controls family-wise Type-I error.
    """
    pvals = np.asarray(pvals)
    m = len(pvals)
    return min(1.0, np.min(pvals) * m)


# ==============================================================================
# SYNTHETIC DATA GENERATORS
# ==============================================================================

def gaussian_1d(rng, n, mu=0.0, sigma=1.0):
    return rng.normal(mu, sigma, size=n)


def make_noncentral(rng, n, kind, shift, sigma=1.0, df=5, skew=4.0, eps=0.1):
    """Draw n samples from a distribution with a controllable MEAN shift but a
    chosen SHAPE, so we can test the t-test's robustness to misspecification.

    All variants are standardised to unit variance, then `shift` is added to the
    mean, so `shift` is always measured in standard deviations (a clean effect
    size regardless of the base shape).

    kind:
      'gaussian' — the assumption holds exactly
      't'        — Student-t with `df` dof: heavy tails, worse as df shrinks
      'skew'     — skew-normal with shape `skew`: asymmetric
      'contam'   — Gaussian with a fraction `eps` of high-variance outliers
    """
    if kind == "gaussian":
        base = rng.normal(0, 1, size=n)
    elif kind == "t":
        base = rng.standard_t(df, size=n)
        base /= np.sqrt(df / (df - 2.0))          # standardise to unit variance
    elif kind == "skew":
        base = stats.skewnorm.rvs(a=skew, size=n, random_state=rng)
        m, s = stats.skewnorm.stats(a=skew, moments="mv")
        base = (base - m) / np.sqrt(s)            # standardise
    elif kind == "contam":
        mask = rng.random(n) < eps
        base = rng.normal(0, 1, size=n)
        base[mask] = rng.normal(0, 5, size=mask.sum())
        base /= base.std()                        # standardise empirically
    else:
        raise ValueError(f"unknown kind {kind}")
    return base * sigma + shift


# ==============================================================================
# SECTION 0 — CORE CONCEPTS
# ==============================================================================
def section0_core(rng, n_trials):
    """Demonstrate the vocabulary: run one test, then estimate Type-I error &
    power by Monte Carlo, and draw the sampling distribution of the statistic
    under H0 vs H1."""
    log("SECTION 0: core concepts (z-test, t-test, p-value, Type-I, power)")
    t0 = time.time()

    n = 50
    sigma = 1.0

    # --- one illustrative run under a real shift ---
    x = gaussian_1d(rng, n, mu=0.0, sigma=sigma)
    y = gaussian_1d(rng, n, mu=0.5, sigma=sigma)   # true 0.5-sigma shift
    z, pz = z_test(x, y, sigma)
    t, pt = welch_t_test(x, y)
    d, pk = ks_test(x, y)
    log(f"  one sample (true shift=0.5sigma, n={n}): "
        f"z={z:.3f} p={pz:.4f} | t={t:.3f} p={pt:.4f} | KS={d:.3f} p={pk:.4f}")

    # --- Type-I error: NO shift, fraction of false alarms should be ~alpha ---
    log(f"  estimating Type-I error over {n_trials} no-shift trials ...")
    rej_z = rej_t = rej_k = 0
    tstats_h0 = []
    for i in range(n_trials):
        xx = gaussian_1d(rng, n, 0.0, sigma)
        yy = gaussian_1d(rng, n, 0.0, sigma)       # same distribution
        _, pz_ = z_test(xx, yy, sigma)
        tt, pt_ = welch_t_test(xx, yy)
        _, pk_ = ks_test(xx, yy)
        rej_z += pz_ < ALPHA
        rej_t += pt_ < ALPHA
        rej_k += pk_ < ALPHA
        tstats_h0.append(tt)
        if (i + 1) % max(1, n_trials // 4) == 0:
            log(f"    ... {i+1}/{n_trials} H0 trials done")
    log(f"  Type-I error (target {ALPHA}): "
        f"z={rej_z/n_trials:.3f}  t={rej_t/n_trials:.3f}  KS={rej_k/n_trials:.3f}")

    # --- Power: WITH a 0.5-sigma shift, fraction correctly rejected ---
    log(f"  estimating power over {n_trials} shifted trials ...")
    shift = 0.5
    pw_z = pw_t = pw_k = 0
    tstats_h1 = []
    for i in range(n_trials):
        xx = gaussian_1d(rng, n, 0.0, sigma)
        yy = gaussian_1d(rng, n, shift, sigma)
        _, pz_ = z_test(xx, yy, sigma)
        tt, pt_ = welch_t_test(xx, yy)
        _, pk_ = ks_test(xx, yy)
        pw_z += pz_ < ALPHA
        pw_t += pt_ < ALPHA
        pw_k += pk_ < ALPHA
        tstats_h1.append(tt)
        if (i + 1) % max(1, n_trials // 4) == 0:
            log(f"    ... {i+1}/{n_trials} H1 trials done")
    log(f"  Power at shift={shift}sigma, n={n}: "
        f"z={pw_z/n_trials:.3f}  t={pw_t/n_trials:.3f}  KS={pw_k/n_trials:.3f}")

    # --- figure: sampling distribution of the t-statistic, H0 vs H1 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-6, 8, 60)
    ax.hist(tstats_h0, bins=bins, alpha=0.6, density=True, label="H0 (no shift)")
    ax.hist(tstats_h1, bins=bins, alpha=0.6, density=True,
            label=f"H1 (shift={shift}$\\sigma$)")
    crit = stats.t.ppf(1 - ALPHA / 2, df=2 * n - 2)
    ax.axvline(crit, color="k", ls="--", lw=1)
    ax.axvline(-crit, color="k", ls="--", lw=1,
               label=f"reject region (|t|>{crit:.2f})")
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("density")
    ax.set_title("Section 0: sampling distribution of the t-statistic\n"
                 "overlap in the reject region = power; H0 mass there = Type-I error")
    ax.legend()
    fig.tight_layout()
    path = f"{FIG_DIR}/section0_core_concepts.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# SECTION 1 — EFFICIENCY: samples needed to match power
# ==============================================================================
def power_at(rng, test, n, shift, n_trials, sigma=1.0):
    """Monte-Carlo power of `test` at sample size n and given shift."""
    rej = 0
    for _ in range(n_trials):
        x = gaussian_1d(rng, n, 0.0, sigma)
        y = gaussian_1d(rng, n, shift, sigma)
        if test == "t":
            _, p = welch_t_test(x, y)
        elif test == "ks":
            _, p = ks_test(x, y)
        elif test == "z":
            _, p = z_test(x, y, sigma)
        rej += p < ALPHA
    return rej / n_trials


def min_n_for_power(rng, test, shift, target, n_grid, n_trials, sigma=1.0):
    """Smallest n on the grid reaching `target` power. Returns (n, power_curve)."""
    curve = []
    hit = None
    for n in n_grid:
        pw = power_at(rng, test, n, shift, n_trials, sigma)
        curve.append(pw)
        if hit is None and pw >= target:
            hit = n
    return hit, curve


def section1_efficiency(rng, n_trials):
    """How many fewer samples does the t-test need to reach 80% power vs KS,
    when the Gaussian assumption holds? Also give the analytic t-test answer."""
    log("SECTION 1: efficiency (sample size to reach target power)")
    t0 = time.time()
    shift = 0.4
    target = 0.80
    n_grid = [10, 15, 20, 30, 40, 50, 60, 75, 90, 110, 130, 160, 200]

    log(f"  shift={shift}sigma, target power={target}, grid={n_grid}")
    log("  sweeping t-test ...")
    n_t, curve_t = min_n_for_power(rng, "t", shift, target, n_grid, n_trials)
    log(f"    t-test reaches {target} power at n>={n_t}")
    log("  sweeping KS test ...")
    n_k, curve_k = min_n_for_power(rng, "ks", shift, target, n_grid, n_trials)
    log(f"    KS test reaches {target} power at n>={n_k}")

    # analytic check for the t/z-test: n per group for two-sided test
    # n = (z_{1-a/2} + z_{power})^2 * 2 / shift^2   (equal n, unit variance)
    za = stats.norm.ppf(1 - ALPHA / 2)
    zb = stats.norm.ppf(target)
    n_analytic = (za + zb) ** 2 * 2.0 / shift ** 2
    log(f"    analytic z-test sample size for {target} power: "
        f"n~={n_analytic:.1f} per group")
    if n_t and n_k:
        log(f"    EFFICIENCY: KS needs ~{n_k/n_t:.2f}x the samples of the t-test")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_grid, curve_t, "o-", label="t-test (parametric)")
    ax.plot(n_grid, curve_k, "s-", label="KS test (nonparametric)")
    ax.axhline(target, color="k", ls="--", lw=1, label=f"target {target}")
    ax.set_xlabel("samples per group (n)")
    ax.set_ylabel("power")
    ax.set_title(f"Section 1: efficiency — power vs sample size "
                 f"(shift={shift}$\\sigma$)")
    ax.legend()
    fig.tight_layout()
    path = f"{FIG_DIR}/section1_efficiency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# SECTION 2 — MISSPECIFICATION ENVELOPE
# ==============================================================================
def section2_misspecification(rng, n_trials):
    """The t-test assumes Gaussianity. Draw from increasingly non-Gaussian
    shapes and track (a) its Type-I error — does it stay calibrated at alpha? —
    and (b) its power vs KS. Where the Type-I error runs away, the test's
    'advantage' is an illusion."""
    log("SECTION 2: misspecification envelope (how non-Gaussian can we go)")
    t0 = time.time()
    n = 50
    shift = 0.5

    # Student-t with shrinking dof: df=inf ~ Gaussian, df=3 very heavy-tailed.
    df_list = [50, 20, 10, 7, 5, 4, 3]
    log(f"  base shape = Student-t, dof sweep {df_list}, n={n}, shift={shift}sigma")

    t1_t, t1_k, pw_t, pw_k = [], [], [], []
    for df in df_list:
        rej_t0 = rej_k0 = rej_t1 = rej_k1 = 0
        for _ in range(n_trials):
            # Type-I (no shift)
            x0 = make_noncentral(rng, n, "t", 0.0, df=df)
            y0 = make_noncentral(rng, n, "t", 0.0, df=df)
            rej_t0 += welch_t_test(x0, y0)[1] < ALPHA
            rej_k0 += ks_test(x0, y0)[1] < ALPHA
            # Power (shift)
            x1 = make_noncentral(rng, n, "t", 0.0, df=df)
            y1 = make_noncentral(rng, n, "t", shift, df=df)
            rej_t1 += welch_t_test(x1, y1)[1] < ALPHA
            rej_k1 += ks_test(x1, y1)[1] < ALPHA
        t1_t.append(rej_t0 / n_trials)
        t1_k.append(rej_k0 / n_trials)
        pw_t.append(rej_t1 / n_trials)
        pw_k.append(rej_k1 / n_trials)
        log(f"    df={df:>2}: Type-I t={t1_t[-1]:.3f} KS={t1_k[-1]:.3f} | "
            f"power t={pw_t[-1]:.3f} KS={pw_k[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = [1.0 / d for d in df_list]   # 1/df: 0 = Gaussian, larger = heavier tails
    axes[0].plot(xs, t1_t, "o-", label="t-test")
    axes[0].plot(xs, t1_k, "s-", label="KS")
    axes[0].axhline(ALPHA, color="k", ls="--", lw=1, label=f"nominal {ALPHA}")
    axes[0].set_xlabel("1/dof  (0 = Gaussian, larger = heavier tails)")
    axes[0].set_ylabel("Type-I error")
    axes[0].set_title("Calibration under misspecification")
    axes[0].legend()
    axes[1].plot(xs, pw_t, "o-", label="t-test")
    axes[1].plot(xs, pw_k, "s-", label="KS")
    axes[1].set_xlabel("1/dof")
    axes[1].set_ylabel("power")
    axes[1].set_title(f"Power under misspecification (shift={shift}$\\sigma$)")
    axes[1].legend()
    fig.suptitle("Section 2: misspecification envelope (Student-t base)")
    fig.tight_layout()
    path = f"{FIG_DIR}/section2_misspecification.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# SECTION 3 — DETECTION LATENCY: sequential vs repeated batch
# ==============================================================================
def section3_latency(rng, n_trials):
    """Streaming setting: samples arrive one at a time from the NEW stream.
    - Parametric sequential test: a running z-test that updates every sample and
      fires the first time p<alpha (crude SPRT-flavoured monitor).
    - Nonparametric batch: re-run KS on the whole new-stream-so-far every `batch`
      samples.
    Measure average number of NEW samples until detection, and false-alarm rate
    on a no-shift stream. Fewer samples-to-detect = lower latency."""
    log("SECTION 3: detection latency (sequential parametric vs repeated batch KS)")
    t0 = time.time()
    ref_n = 500
    sigma = 1.0
    shift = 0.5
    max_stream = 300
    batch = 20
    # Bonferroni-style per-look alpha so repeated looks don't inflate false alarms
    # (both monitors get the same total budget, approximately).
    per_look_alpha = ALPHA / (max_stream)          # seq: one look per sample
    per_look_alpha_batch = ALPHA / (max_stream // batch)

    log(f"  ref_n={ref_n}, shift={shift}sigma, max_stream={max_stream}, "
        f"batch={batch}")

    def run_once(with_shift):
        ref = gaussian_1d(rng, ref_n, 0.0, sigma)
        mu_new = shift if with_shift else 0.0
        stream = gaussian_1d(rng, max_stream, mu_new, sigma)
        # sequential running z-test
        seq_detect = None
        running_sum = 0.0
        for k in range(1, max_stream + 1):
            running_sum += stream[k - 1]
            ybar = running_sum / k
            se = sigma * np.sqrt(1.0 / ref_n + 1.0 / k)
            z = (ref.mean() - ybar) / se
            p = 2.0 * stats.norm.sf(abs(z))
            if p < per_look_alpha:
                seq_detect = k
                break
        # repeated batch KS
        ks_detect = None
        for k in range(batch, max_stream + 1, batch):
            _, p = ks_test(ref, stream[:k])
            if p < per_look_alpha_batch:
                ks_detect = k
                break
        return seq_detect, ks_detect

    # --- false alarms on a no-shift stream ---
    log(f"  measuring false-alarm rate over {n_trials} no-shift streams ...")
    fa_seq = fa_ks = 0
    for i in range(n_trials):
        s, k = run_once(with_shift=False)
        fa_seq += s is not None
        fa_ks += k is not None
        if (i + 1) % max(1, n_trials // 4) == 0:
            log(f"    ... {i+1}/{n_trials}")
    log(f"  false-alarm rate: sequential={fa_seq/n_trials:.3f}  "
        f"batch-KS={fa_ks/n_trials:.3f}")

    # --- latency on a shifted stream (only count runs that detected) ---
    log(f"  measuring detection latency over {n_trials} shifted streams ...")
    seq_lat, ks_lat = [], []
    seq_miss = ks_miss = 0
    for i in range(n_trials):
        s, k = run_once(with_shift=True)
        if s is not None:
            seq_lat.append(s)
        else:
            seq_miss += 1
        if k is not None:
            ks_lat.append(k)
        else:
            ks_miss += 1
        if (i + 1) % max(1, n_trials // 4) == 0:
            log(f"    ... {i+1}/{n_trials}")
    seq_mean = np.mean(seq_lat) if seq_lat else float("nan")
    ks_mean = np.mean(ks_lat) if ks_lat else float("nan")
    log(f"  mean samples-to-detect: sequential={seq_mean:.1f} "
        f"(missed {seq_miss}) | batch-KS={ks_mean:.1f} (missed {ks_miss})")

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max_stream + batch, batch)
    ax.hist(seq_lat, bins=bins, alpha=0.6, label=f"sequential z (mean {seq_mean:.0f})")
    ax.hist(ks_lat, bins=bins, alpha=0.6, label=f"batch KS (mean {ks_mean:.0f})")
    ax.set_xlabel("new samples until detection")
    ax.set_ylabel("count")
    ax.set_title("Section 3: detection latency (lower = faster)")
    ax.legend()
    fig.tight_layout()
    path = f"{FIG_DIR}/section3_latency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# SECTION 4 — COMBINATION RULES: per-dim t + Bonferroni vs Hotelling's T^2
# ==============================================================================
def make_mv(rng, n, d, shift_vec, corr=0.0):
    """n samples of d-dim Gaussian, equicorrelation `corr`, mean = shift_vec."""
    cov = np.full((d, d), corr) + np.eye(d) * (1 - corr)
    return rng.multivariate_normal(shift_vec, cov, size=n)


def section4_combination(rng, n_trials):
    """Multivariate mean shift spread thinly across many dimensions.
    - Bonferroni over per-dimension t-tests (ignores correlation).
    - Hotelling's T^2 (uses the full covariance → exploits correlation).
    - KS per-dim + Bonferroni as the nonparametric baseline.
    Vary the correlation between dimensions; T^2 should pull ahead as |corr| grows."""
    log("SECTION 4: combination rules (Bonferroni t vs Hotelling T^2 vs Bonferroni KS)")
    t0 = time.time()
    n = 60
    d = 10
    total_shift = 0.8                  # spread across all dims
    shift_vec = np.full(d, total_shift / np.sqrt(d))
    zero_vec = np.zeros(d)
    corr_list = [0.0, 0.2, 0.4, 0.6, 0.8]
    log(f"  n={n}, d={d}, total_shift={total_shift}, corr sweep {corr_list}")

    pw_bonf_t, pw_hot, pw_bonf_ks, t1_hot = [], [], [], []
    for corr in corr_list:
        rej_bt = rej_h = rej_bk = fa_h = 0
        for _ in range(n_trials):
            # power (shift present)
            X = make_mv(rng, n, d, zero_vec, corr)
            Y = make_mv(rng, n, d, shift_vec, corr)
            pvals_t = [welch_t_test(X[:, j], Y[:, j])[1] for j in range(d)]
            pvals_k = [ks_test(X[:, j], Y[:, j])[1] for j in range(d)]
            rej_bt += bonferroni_pvalue(pvals_t) < ALPHA
            rej_bk += bonferroni_pvalue(pvals_k) < ALPHA
            rej_h += hotelling_t2(X, Y)[1] < ALPHA
            # Type-I for Hotelling (no shift) — check it stays calibrated
            X0 = make_mv(rng, n, d, zero_vec, corr)
            Y0 = make_mv(rng, n, d, zero_vec, corr)
            fa_h += hotelling_t2(X0, Y0)[1] < ALPHA
        pw_bonf_t.append(rej_bt / n_trials)
        pw_hot.append(rej_h / n_trials)
        pw_bonf_ks.append(rej_bk / n_trials)
        t1_hot.append(fa_h / n_trials)
        log(f"    corr={corr:.1f}: power Bonf-t={pw_bonf_t[-1]:.3f} "
            f"Hotelling={pw_hot[-1]:.3f} Bonf-KS={pw_bonf_ks[-1]:.3f} | "
            f"T2 Type-I={t1_hot[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(corr_list, pw_bonf_t, "o-", label="Bonferroni + per-dim t")
    ax.plot(corr_list, pw_hot, "^-", label="Hotelling's $T^2$")
    ax.plot(corr_list, pw_bonf_ks, "s-", label="Bonferroni + per-dim KS")
    ax.set_xlabel("between-dimension correlation")
    ax.set_ylabel("power")
    ax.set_title(f"Section 4: combination rules "
                 f"(d={d}, total shift={total_shift})")
    ax.legend()
    fig.tight_layout()
    path = f"{FIG_DIR}/section4_combination.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# SECTION 5 — COMPRESSION: PCA / LDA-direction reduction vs per-dimension
# ==============================================================================
def section5_compression(rng, n_trials):
    """High-dim shift concentrated in ONE unknown direction. Compare:
    - per-dimension t + Bonferroni (pays a d-fold multiplicity penalty)
    - PCA to top-1 component (unsupervised; may miss a low-variance shift dir)
    - mean-difference direction projection (supervised 'LDA-like' compression,
      the strongest classical move when it's valid)
    Report power for each."""
    log("SECTION 5: compression (per-dim vs PCA vs mean-difference projection)")
    t0 = time.time()
    n = 60
    d = 20
    shift_mag = 0.7
    # shift lives along a single random unit direction
    log(f"  n={n}, d={d}, shift_mag={shift_mag} along one random direction")

    pw_perdim = pw_pca = pw_meandir = 0
    for i in range(n_trials):
        u = rng.normal(size=d); u /= np.linalg.norm(u)      # shift direction
        X = rng.normal(size=(n, d))
        Y = rng.normal(size=(n, d)) + shift_mag * u          # shifted along u

        # per-dimension t + Bonferroni
        pvals = [welch_t_test(X[:, j], Y[:, j])[1] for j in range(d)]
        pw_perdim += bonferroni_pvalue(pvals) < ALPHA

        # PCA on pooled data -> project onto top component -> t-test
        pooled = np.vstack([X, Y])
        pooled_c = pooled - pooled.mean(axis=0)
        _, _, Vt = np.linalg.svd(pooled_c, full_matrices=False)
        pc1 = Vt[0]
        px, py = X @ pc1, Y @ pc1
        pw_pca += welch_t_test(px, py)[1] < ALPHA

        # mean-difference direction (estimated on the SAME data -> slightly
        # optimistic, but illustrates the supervised-compression ceiling)
        w = Y.mean(axis=0) - X.mean(axis=0); w /= (np.linalg.norm(w) + 1e-12)
        pw_meandir += welch_t_test(X @ w, Y @ w)[1] < ALPHA

        if (i + 1) % max(1, n_trials // 4) == 0:
            log(f"    ... {i+1}/{n_trials}")

    log(f"  power: per-dim+Bonf={pw_perdim/n_trials:.3f}  "
        f"PCA-top1={pw_pca/n_trials:.3f}  "
        f"mean-dir={pw_meandir/n_trials:.3f}")

    # Type-I check for the mean-direction trick (data-snooping risk)
    log(f"  checking mean-direction Type-I error (no shift) ...")
    fa = 0
    for _ in range(n_trials):
        X = rng.normal(size=(n, d)); Y = rng.normal(size=(n, d))
        w = Y.mean(axis=0) - X.mean(axis=0); w /= (np.linalg.norm(w) + 1e-12)
        fa += welch_t_test(X @ w, Y @ w)[1] < ALPHA
    log(f"  mean-direction Type-I (target {ALPHA}): {fa/n_trials:.3f}  "
        f"<- note: fitting the direction on the same data inflates this")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["per-dim\n+Bonferroni", "PCA top-1", "mean-diff\ndirection"]
    vals = [pw_perdim / n_trials, pw_pca / n_trials, pw_meandir / n_trials]
    ax.bar(names, vals, color=["#4477aa", "#228833", "# ccbb44".replace(" ", "")])
    ax.axhline(ALPHA, color="k", ls="--", lw=1, label=f"chance ~ {ALPHA}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_ylabel("power")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Section 5: compression strategies (d={d}, 1-direction shift)")
    ax.legend()
    fig.tight_layout()
    path = f"{FIG_DIR}/section5_compression.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# SECTION 6 — RANDOM-PROJECTION t-test
# ==============================================================================
def section6_random_projection(rng, n_trials):
    """The t-test analogue of random-projection KS. Instead of testing each raw
    dimension, draw `k` random unit directions, t-test the projected samples,
    and combine via Bonferroni. Compare power vs per-raw-dimension Bonferroni,
    as a function of how many projections k you use."""
    log("SECTION 6: random-projection t-test + Bonferroni")
    t0 = time.time()
    n = 60
    d = 30
    shift_mag = 0.6
    k_list = [1, 3, 5, 10, 20, 40]
    log(f"  n={n}, d={d}, shift_mag={shift_mag}, #projections sweep {k_list}")

    # baseline: per-raw-dim t + Bonferroni
    base_rej = 0
    for _ in range(n_trials):
        u = rng.normal(size=d); u /= np.linalg.norm(u)
        X = rng.normal(size=(n, d))
        Y = rng.normal(size=(n, d)) + shift_mag * u
        pvals = [welch_t_test(X[:, j], Y[:, j])[1] for j in range(d)]
        base_rej += bonferroni_pvalue(pvals) < ALPHA
    base_power = base_rej / n_trials
    log(f"  baseline per-raw-dim+Bonferroni power = {base_power:.3f}")

    rp_power = []
    for k in k_list:
        rej = 0
        for _ in range(n_trials):
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            X = rng.normal(size=(n, d))
            Y = rng.normal(size=(n, d)) + shift_mag * u
            # k random unit directions
            P = rng.normal(size=(d, k))
            P /= np.linalg.norm(P, axis=0, keepdims=True)
            Xp, Yp = X @ P, Y @ P                     # (n, k)
            pvals = [welch_t_test(Xp[:, j], Yp[:, j])[1] for j in range(k)]
            rej += bonferroni_pvalue(pvals) < ALPHA
        rp_power.append(rej / n_trials)
        log(f"    k={k:>2} projections: power={rp_power[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_list, rp_power, "o-", label="random-projection t + Bonferroni")
    ax.axhline(base_power, color="C1", ls="--",
               label=f"per-raw-dim + Bonferroni ({base_power:.2f})")
    ax.set_xlabel("number of random projections k")
    ax.set_ylabel("power")
    ax.set_title(f"Section 6: random-projection t-test (d={d})")
    ax.legend()
    fig.tight_layout()
    path = f"{FIG_DIR}/section6_random_projection.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"  saved {path}", t0)


# ==============================================================================
# MAIN
# ==============================================================================
SECTIONS = {
    0: section0_core,
    1: section1_efficiency,
    2: section2_misspecification,
    3: section3_latency,
    4: section4_combination,
    5: section5_compression,
    6: section6_random_projection,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="run only these section numbers (0-6)")
    ap.add_argument("--fast", action="store_true",
                    help="fewer Monte-Carlo trials for a quick run")
    args = ap.parse_args()

    n_trials = N_TRIALS_FAST if args.fast else N_TRIALS
    rng = np.random.default_rng(RNG_SEED)

    import os
    os.makedirs(FIG_DIR, exist_ok=True)

    to_run = args.only if args.only is not None else list(SECTIONS.keys())
    log("=" * 70)
    log(f"Two-sample z/t-test shift-detection study")
    log(f"alpha={ALPHA}, trials/section={n_trials}, sections={to_run}")
    log("=" * 70)

    grand_t0 = time.time()
    for s in to_run:
        if s not in SECTIONS:
            log(f"  (skipping unknown section {s})")
            continue
        print()
        SECTIONS[s](rng, n_trials)
    print()
    log(f"ALL DONE. total elapsed {time.time()-grand_t0:.1f}s. "
        f"Figures in ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
