# SDET_Dataset_2: simplified shift benchmark

This dataset replaces the 126-scenario `SDET_Dataset_1` with 8 shift families. Each family changes one thing, so a test's result has one explanation.

| Part | Family | What changes | Level values (k = 0 is always the null) |
|---|---|---|---|
| Synthetic | `mean_sparse` | Mean of feature 1 moves by a | a = 0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6 |
| Synthetic | `mean_dense` | Every mean moves by a/√d (same total distance) | same a |
| Synthetic | `variance` | Every std multiplied by s | s = 1, 1.05, 1.1, 1.2, 1.35, 1.5, 2 |
| Synthetic | `correlation` | Features become equicorrelated at r; marginals stay N(0, 1) | r = 0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5 |
| Synthetic | `heavytail_mean` | `mean_dense` with Student-t (3 df) noise in both P and Q | same a |
| MNIST | `gn` | Gaussian pixel noise σ ∈ {1, 10, 100}/255 on a fraction δ of images | 3 intensities × δ ∈ {0.1, 0.5, 1.0} |
| MNIST | `img` | Random affine transform (rotation, shift, shear, zoom, flips) | 3 intensities × δ |
| MNIST | `ko` | Delete a fraction δ of the digit-0 images (label shift) | δ ∈ {0.1, 0.5, 1.0} |

The synthetic reference is P = N(0, I) with d ∈ {10, 50}, n ∈ {10, 20, 50, 100} (matched n = m), and 100 replicates per cell. The MNIST shifts are the *Failing Loudly* shifts with the parameters of the paper's code (`ponnam/shiftbench/shifts.py`). The adversarial and combined shifts are left out.

## Files

```
data/manifest.json                 every scenario: parameters, plain-English description, sha256
data/synthetic/<family>_d<d>.npz   X__n{n} (R,n,d) from P;  Y__n{n}__m{k} (R,n,d) from Q_k
                                   S__levels, S__delta (K,d), S__Sigma_Q (K,d,d), S__Sigma_P
data/mnist/source.npz              x_train (50k, fits reducers), x_source (10k, reference pool P), labels
data/mnist/target_<shift>.npz      x (shifted target pool), y;  target_none.npz is the null pool
```

Synthetic arrays are float32. MNIST pixels are uint8 (0–255), except the `gn` pools, which are float16 in [0, 1] because rounding would erase the σ = 1/255 noise.

Within a synthetic replicate, every Y_k is built from the same base noise draw, so power curves are smooth in k. Different k are therefore not independent replications.

## Regenerate

From the repository root (about 3 minutes; MNIST downloads on first run):

```
python SDET_Dataset_2/generate.py
```

The experiments are `ponnam/scripts/09_simple_benchmark.py` and `ponnam/scripts/10_simple_figures.py`.
