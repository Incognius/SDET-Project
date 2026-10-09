"""Produce the figures for the presentation from the saved result tables.

Reads whatever studies have been run and writes PNG and PDF versions of each
figure to ``results/figures/``.  Figures are the presentation's main currency,
so each one is built to make a single claim legible rather than to display
everything that was measured.

Figures produced
----------------
``calibration.png``
    Measured Type-I error against nominal level, per test.  Points on the
    diagonal are correctly calibrated; points above it are anti-conservative
    and their power numbers cannot be trusted.
``power_<probe>.png``
    Power against sample size at fixed alpha, one panel per effect size.  The
    horizontal line at alpha marks the floor: a curve sitting on it is a
    structurally blind test.
``crossover.png``
    Samples required to reach 80% power, per test and probe.  This is the
    sample-complexity comparison in its most direct form.
``latency.png``
    Seconds per test against sample size, on log axes.

Usage::

    python scripts/05_figures.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A fixed, colour-blind-safe palette so that a test keeps the same colour in
# every figure.  Consistency across figures is what lets a reader track one
# method through the whole deck.
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#1F4E79", "#A23B72", "#2E7D32", "#6A1B9A", "#EF6C00",
]


def _style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _colours(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sorted(names))}


def _save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def fig_calibration(results_dir, out_dir):
    path = os.path.join(results_dir, "calibration_summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    agg = df.groupby(["test", "alpha"])["type_I_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colours = _colours(agg["test"].unique())
    lims = [0.005, 0.2]
    ax.plot(lims, lims, "k--", lw=1, zorder=1, label="perfect calibration")
    for test, sub in agg.groupby("test"):
        sub = sub.sort_values("alpha")
        ax.plot(sub["alpha"], sub["type_I_rate"], "o-", ms=4, lw=1.2,
                color=colours[test], label=test, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(0.0008, 0.3)
    ax.set_xlabel("nominal level  $\\alpha$")
    ax.set_ylabel("measured Type-I error")
    ax.set_title("Calibration: measured false-alarm rate vs nominal level\n"
                 "(above the diagonal = anti-conservative)")
    ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    _save(fig, out_dir, "calibration")


def fig_power(results_dir, out_dir, alpha=0.05):
    path = os.path.join(results_dir, "probe_power.parquet")
    if not os.path.exists(path):
        return
    df = pd.read_parquet(path)
    df = df[df["alpha"] == alpha]
    g = (df.groupby(["probe", "eps", "d", "test", "n"])["reject"]
         .mean().reset_index().rename(columns={"reject": "power"}))

    colours = _colours(g["test"].unique())
    for probe, sub in g.groupby("probe"):
        eps_vals = sorted(sub["eps"].unique())
        fig, axes = plt.subplots(
            1, len(eps_vals), figsize=(3.4 * len(eps_vals), 3.3), sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, eps in zip(axes, eps_vals):
            cell = sub[sub["eps"] == eps]
            for test, s in cell.groupby("test"):
                s = s.sort_values("n")
                ax.plot(s["n"], s["power"], "o-", ms=3, lw=1.2,
                        color=colours[test], label=test)
            ax.axhline(alpha, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("samples per side, $n$")
            ax.set_title(f"$\\epsilon$ = {eps}")
        axes[0].set_ylabel(f"power at $\\alpha$ = {alpha}")
        fig.suptitle(f"Power on the '{probe}' probe "
                     f"(dashed line = nominal level; a curve on it is blind)",
                     y=1.04)
        axes[-1].legend(ncol=1, fontsize=6, loc="center left",
                        bbox_to_anchor=(1.02, 0.5))
        _save(fig, out_dir, f"power_{probe}")


def fig_crossover(results_dir, out_dir):
    path = os.path.join(results_dir, "probe_crossover.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    eps = df["eps"].max()
    sub = df[df["eps"] == eps]
    pivot = sub.pivot_table(index="test", columns="probe", values="n_for_80pct")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3.5,
                                    0.3 * len(pivot.index) + 2.2))
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#E8E8E8")  # never reached 80% power
    im = ax.imshow(np.log10(masked), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, "never" if not np.isfinite(v) else f"{int(v)}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if np.isfinite(v) and v > 100 else "black")
    ax.grid(False)
    ax.set_title(f"Samples needed for 80% power  ($\\epsilon$ = {eps}, "
                 f"$\\alpha$ = 0.05)\ngrey = never reached on the grid")
    fig.colorbar(im, ax=ax, label="$\\log_{10}$ n")
    _save(fig, out_dir, "crossover")


def fig_latency(results_dir, out_dir):
    path = os.path.join(results_dir, "latency.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    dims = sorted(df["d"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    colours = _colours(df["test"].unique())
    for ax, d in zip(axes, dims):
        cell = df[df["d"] == d]
        for test, s in cell.groupby("test"):
            s = s.sort_values("n")
            ax.plot(s["n"], s["seconds"] * 1000, "o-", ms=3, lw=1.2,
                    color=colours[test], label=test)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("samples per side, $n$")
        ax.set_title(f"d = {d}")
    axes[0].set_ylabel("milliseconds per test (min of passes)")
    axes[-1].legend(ncol=1, fontsize=6, loc="center left",
                    bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Latency: wall-clock cost of one test", y=1.03)
    _save(fig, out_dir, "latency")


def fig_image_pipeline(results_dir, out_dir, alpha=0.05):
    paths = [p for p in os.listdir(results_dir)
             if p.startswith("image_pipeline_") and p.endswith(".parquet")]
    for p in paths:
        dataset = p[len("image_pipeline_"):-len(".parquet")]
        df = pd.read_parquet(os.path.join(results_dir, p))
        df = df[(df["alpha"] == alpha) & (df["is_null"] == 0)]
        if df.empty:
            continue
        g = (df.groupby(["reductor", "test", "n"])["reject"]
             .mean().reset_index().rename(columns={"reject": "detection"}))
        reds = sorted(g["reductor"].unique())
        fig, axes = plt.subplots(1, len(reds), figsize=(3.4 * len(reds), 3.3),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        colours = _colours(g["test"].unique())
        for ax, red in zip(axes, reds):
            cell = g[g["reductor"] == red]
            for test, s in cell.groupby("test"):
                s = s.sort_values("n")
                ax.plot(s["n"], s["detection"], "o-", ms=3, lw=1.2,
                        color=colours[test], label=test)
            ax.axhline(alpha, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("samples from target, $n$")
            ax.set_title(red)
        axes[0].set_ylabel(f"detection rate at $\\alpha$ = {alpha}")
        axes[-1].legend(ncol=1, fontsize=6, loc="center left",
                        bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"{dataset}: detection rate averaged over all shifts", y=1.04)
        _save(fig, out_dir, f"image_pipeline_{dataset}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    out_dir = os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    _style()

    print(f"writing figures to {out_dir}")
    fig_calibration(results_dir, out_dir)
    fig_power(results_dir, out_dir, args.alpha)
    fig_crossover(results_dir, out_dir)
    fig_latency(results_dir, out_dir)
    fig_image_pipeline(results_dir, out_dir, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
