# Per-MLP MSE Distribution Check

This folder captures the quick distribution check for one 50-MLP run. The
question was whether the per-MLP `final_layer_mse` values looked Gaussian or
log-normal.

## Data

The raw pasted run table is normalized into
[`mlp_mse_values.csv`](mlp_mse_values.csv). The fitted values below use the
`final_layer_mse` column.

## Result

The 50 per-MLP MSEs are well described by a log-normal model:

```text
MSE ~ LogNormal(mu=-13.166, sigma=0.589)
```

Observed summary:

| statistic | value |
|---|---:|
| arithmetic mean | `2.30152e-6` |
| median | `1.825e-6` |
| sample sd | `1.63038e-6` |
| coefficient of variation | `0.708` |
| min | `4.40e-7` |
| max | `8.89e-6` |
| max/min | `20.2x` |

The log-normal fit has geometric mean `1.914e-6`. The observed low and high
tails are about `-2.50` and `+2.61` log-standard-deviations from the fitted log
mean, which is ordinary for 50 draws.

## Model Checks

The simple model comparison favored log-normal:

| model | AIC delta vs best |
|---|---:|
| log-normal | `0.0` |
| gamma | `6.0` |
| Weibull | `12.6` |
| raw normal | `35.9` |

Goodness checks did not find evidence against log-normality:

| check | value |
|---|---:|
| KS p-value, fitted log-normal | `0.73` |
| Shapiro p-value on `log(MSE)` | `0.84` |
| Anderson statistic on `log(MSE)` | `0.275` |

The raw normal model is much weaker: its KS p-value is about `0.045`, and the
largest MSE is roughly `4.0` raw-scale standard deviations above the arithmetic
mean.

## 95% Spreads

For an individual MLP under the fitted log-normal:

```text
6.04e-7 to 6.06e-6
```

For the arithmetic mean of 50 MLPs, using a Fenton-Wilkinson approximation for
the mean of log-normal draws:

```text
1.92e-6 to 2.74e-6
```

So a 50-MLP average around `2.7e-6` is near the high end but still inside this
run's fitted 95% mean interval. A 50-MLP average above roughly `2.9e-6` would
start to look high relative to this model.

## Reproduce

Run:

```bash
python analysis/mse-lognormal/analyze_mse_distribution.py
```
