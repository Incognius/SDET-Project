"""shiftbench -- a unified harness for two-sample tests under distribution shift.

Built for the SDET course project *Two-sample tests for distribution shifts*.
The package exists so that four people implementing different tests measure
them on identical data, with identical calibration, and report comparable
numbers.

Layout
------
``datasets``    Canonical, deterministic loading and splitting of the image
                datasets used by *Failing Loudly*.
``shifts``      Faithful reimplementation of that paper's synthetic shift
                catalogue, with exact parameters taken from its reference code.
``synthetic``   Controlled probes whose ground truth is known by construction;
                each breaks exactly one modelling assumption.
``reductors``   The reduce-then-test pipeline's first stage.
``tests``       Every two-sample test, behind one interface.
``experiment``  Calibration, power-curve and real-data drivers.

Usage
-----
>>> from shiftbench.tests import build
>>> from shiftbench.synthetic import sample_probe
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> x, y = sample_probe("mean_sparse", 100, 100, 32, 0.5, rng)
>>> build("ks_bonferroni")(x, y, rng).p_value < 0.05
True
"""

__version__ = "0.1.0"

from . import datasets, experiment, reductors, shifts, synthetic, tests  # noqa: F401

__all__ = ["datasets", "shifts", "synthetic", "reductors", "tests", "experiment"]
