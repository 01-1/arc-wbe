# Filament Stage-1 Deterministic Grid Propagation Gate (2026-07-06)

Offline analysis-only run under `paired_fly_logs/fingerprint_theory/`. No Fly, network, pytest, or tracked-file edits. MLPs use `local_engine.build_mlp`, width 256, depth 32, seeds 11 and 22. Truth and initialization both use the same 400k antithetic sample set by design, so this gate removes initialization error and tests deterministic propagation machinery only.

## Grid Construction

For each branch layer K, the full branch activation sample is centered and diagonalized. The r=1 grid uses empirical equal-mass quantile cells along the top eigen-score `a = u.(y_K - mean)`. Each node stores its empirical cell mass, empirical conditional mean, and covariance `C_resid + Var(a|cell) uu^T`, where `C_resid` is pooled residual covariance after subtracting a 513-bin fine empirical conditional-mean curve. Thus the mixture mean exactly matches the sample mean at K up to roundoff, and the within-cell latent variance tiles the filament between node means.

Propagation uses exact linear Gaussian moment propagation per node, then nonzero-mean Gaussian ReLU marginal moments plus the GL16 Price-identity bivariate covariance closure ported from the existing estimator/pretest code.

## Bias-MSE Table

| K | G | seed 11 | seed 22 | mean |
|---:|---:|---:|---:|---:|
| 16 | 1 | 3.205e-05 | 4.105e-05 | 3.655e-05 |
| 16 | 9 | 2.545e-05 | 3.941e-05 | 3.243e-05 |
| 16 | 17 | 2.540e-05 | 3.947e-05 | 3.244e-05 |
| 16 | 33 | 2.545e-05 | 3.954e-05 | 3.249e-05 |
| 16 | 65 | 2.550e-05 | 3.960e-05 | 3.255e-05 |
| 24 | 1 | 1.355e-05 | 1.594e-05 | 1.474e-05 |
| 24 | 9 | 1.135e-05 | 1.457e-05 | 1.296e-05 |
| 24 | 17 | 1.135e-05 | 1.453e-05 | 1.294e-05 |
| 24 | 33 | 1.136e-05 | 1.453e-05 | 1.295e-05 |
| 24 | 65 | 1.137e-05 | 1.454e-05 | 1.296e-05 |

## Convergence

| K | raw p | floor-fit p | floor | largest-G MSE |
|---:|---:|---:|---:|---:|
| 16 | -0.002 | -0.002 | 0.000e+00 | 3.255e-05 |
| 24 | 0.000 | 0.000 | 0.000e+00 | 1.296e-05 |

## Verdict

- K=24, G=65 mean final bias-MSE: `1.296e-05`
- Selected convergence order p: `0.000`
- Verdict: **MACHINERY INSUFFICIENT AS CONSTRUCTED**

## Per-Layer Diagnostics

Mean MSE across seeds for selected anchors.

| K | G | layer | mean MSE |
|---:|---:|---:|---:|
| 16 | 1 | 16 | 3.346e-21 |
| 16 | 1 | 17 | 1.979e-06 |
| 16 | 1 | 18 | 4.342e-06 |
| 16 | 1 | 19 | 5.804e-06 |
| 16 | 1 | 20 | 9.007e-06 |
| 16 | 1 | 21 | 1.019e-05 |
| 16 | 1 | 22 | 1.116e-05 |
| 16 | 1 | 23 | 1.458e-05 |
| 16 | 1 | 24 | 1.840e-05 |
| 16 | 1 | 25 | 1.903e-05 |
| 16 | 1 | 26 | 2.295e-05 |
| 16 | 1 | 27 | 2.690e-05 |
| 16 | 1 | 28 | 2.812e-05 |
| 16 | 1 | 29 | 3.564e-05 |
| 16 | 1 | 30 | 3.674e-05 |
| 16 | 1 | 31 | 4.025e-05 |
| 16 | 1 | 32 | 3.655e-05 |
| 16 | 65 | 16 | 3.346e-21 |
| 16 | 65 | 17 | 1.806e-06 |
| 16 | 65 | 18 | 3.980e-06 |
| 16 | 65 | 19 | 5.121e-06 |
| 16 | 65 | 20 | 7.691e-06 |
| 16 | 65 | 21 | 8.581e-06 |
| 16 | 65 | 22 | 9.497e-06 |
| 16 | 65 | 23 | 1.243e-05 |
| 16 | 65 | 24 | 1.567e-05 |
| 16 | 65 | 25 | 1.626e-05 |
| 16 | 65 | 26 | 1.978e-05 |
| 16 | 65 | 27 | 2.347e-05 |
| 16 | 65 | 28 | 2.397e-05 |
| 16 | 65 | 29 | 3.050e-05 |
| 16 | 65 | 30 | 3.241e-05 |
| 16 | 65 | 31 | 3.615e-05 |
| 16 | 65 | 32 | 3.255e-05 |
| 24 | 1 | 24 | 3.487e-21 |
| 24 | 1 | 25 | 1.942e-06 |
| 24 | 1 | 26 | 3.912e-06 |
| 24 | 1 | 27 | 5.851e-06 |
| 24 | 1 | 28 | 8.158e-06 |
| 24 | 1 | 29 | 1.009e-05 |
| 24 | 1 | 30 | 9.897e-06 |
| 24 | 1 | 31 | 1.234e-05 |
| 24 | 1 | 32 | 1.474e-05 |
| 24 | 65 | 24 | 3.487e-21 |
| 24 | 65 | 25 | 1.710e-06 |
| 24 | 65 | 26 | 3.435e-06 |
| 24 | 65 | 27 | 5.101e-06 |
| 24 | 65 | 28 | 7.215e-06 |
| 24 | 65 | 29 | 8.972e-06 |
| 24 | 65 | 30 | 9.059e-06 |
| 24 | 65 | 31 | 1.108e-05 |
| 24 | 65 | 32 | 1.296e-05 |

## Optional r=2 Check

| seed | K | grid | nodes | final MSE |
|---:|---:|---|---:|---:|
| 11 | 24 | 9x9 | 81 | 4.914e-06 |
| 22 | 24 | 9x9 | 81 | 8.502e-06 |

## Recommended Next Action

Do not invest in estimator initialization yet. Use the per-layer diagnostics to isolate whether error enters immediately after K from node closure, accumulates smoothly from Gaussian covariance closure, or remains from r=1 truncation.
