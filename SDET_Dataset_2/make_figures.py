"""Plot the rejection rates written by run_tests.py into results/figures/.

Run from this folder:  python make_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
ALPHA = 0.05
D = 10
NS = [10, 20, 50, 100]

FAMILIES = ["mean_dense", "variance", "heavytail_mean"]
FAMILY_TITLE = {"mean_dense": "Mean shift", "variance": "Variance shift",
                "heavytail_mean": "Mean shift, heavy-tailed data"}
LEVEL_LABEL = {"mean_dense": "shift size a", "variance": "std multiplier s",
               "heavytail_mean": "shift size a"}

# Validated categorical order (blue, orange, aqua); markers give a second encoding.
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "t":    dict(color="#2a78d6", marker="o", label="t-test"),
    "ks":   dict(color="#eb6834", marker="s", label="KS test"),
    "glrt": dict(color="#1baf7a", marker="^", label="GLRT"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})

rates = defaultdict(dict)  # (family, n, test) -> {k: rate}
levels = defaultdict(dict)
with open(RES / "rejection_rates.csv") as fh:
    for r in csv.DictReader(fh):
        rates[(r["family"], int(r["n"]), r["test"])][int(r["k"])] = float(r["reject_rate"])
        levels[r["family"]][int(r["k"])] = float(r["level"])


def alpha_line(ax):
    ax.axhline(ALPHA, color=TEXT2, lw=1, ls=":", zorder=0)


def finish(fig, path, legend_from=None):
    if legend_from is not None:
        h, lab = legend_from.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path.relative_to(HERE))


def fig_false_alarms():
    """Rejection rate when nothing changed (k = 0)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    panels = [(["mean_dense", "variance"], "Gaussian data"), (["heavytail_mean"], "Heavy-tailed data (t₃)")]
    for ax, (fams, title) in zip(axes, panels):
        for t, st in STYLE.items():
            y = [np.mean([rates[(f, n, t)][0] for f in fams]) for n in NS]
            ax.plot(NS, y, **st)
        alpha_line(ax)
        ax.set(xscale="log", xticks=NS, xticklabels=NS, ylim=(0, 1.02), xlabel="samples per group n",
               title=title)
    axes[0].set_ylabel("false-alarm rate")
    fig.suptitle("False alarms when there is no shift (target 0.05, dotted)", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    finish(fig, FIG / "01_false_alarms.png", legend_from=axes[0])


def fig_power(n=100):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        x = [levels[fam][k] for k in sorted(levels[fam])]
        for t, st in STYLE.items():
            c = rates[(fam, n, t)]
            ax.plot(x, [c[k] for k in sorted(c)], **st)
        alpha_line(ax)
        ax.set(ylim=(-0.02, 1.02), xlabel=LEVEL_LABEL[fam], title=FAMILY_TITLE[fam])
    axes[0].set_ylabel("detection rate")
    fig.suptitle(f"How often each test detects the shift (n = {n}, d = {D}; leftmost point = no shift)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    finish(fig, FIG / "02_detection.png", legend_from=axes[0])


def fig_glrt_check(n=100):
    nulls = np.load(RES / "glrt_null_stats.npz")
    df = 2 * D
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    panels = [(["mean_dense", "variance"], "Gaussian data: matches theory"),
              (["heavytail_mean"], "Heavy-tailed data: does not")]
    for ax, (fams, title) in zip(axes, panels):
        s = np.concatenate([nulls[f"{f}_n{n}"] for f in fams])
        hi = max(np.percentile(s, 99), stats.chi2.ppf(0.999, df))
        ax.hist(s, bins=np.linspace(0, hi, 30), density=True, color=STYLE["glrt"]["color"],
                alpha=0.45, edgecolor=SURFACE, linewidth=1, label="GLRT statistic (no shift)")
        xs = np.linspace(0, hi, 300)
        ax.plot(xs, stats.chi2.pdf(xs, df), color=TEXT, lw=1.8, label=f"χ²({df}) theory")
        ax.axvline(stats.chi2.ppf(1 - ALPHA, df), color=TEXT2, ls=":", lw=1)
        ax.set(title=title, xlabel="−2 log Λ", ylabel="density")
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"Is the GLRT statistic χ²-distributed? (n = {n}, dotted = rejection cut-off)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    finish(fig, FIG / "03_glrt_vs_chi2.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for old in FIG.glob("*.png"):
        old.unlink()
    fig_false_alarms()
    fig_power()
    fig_glrt_check()


if __name__ == "__main__":
    main()
