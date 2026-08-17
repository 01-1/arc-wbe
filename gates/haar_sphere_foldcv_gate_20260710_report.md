# Haar-sphere first-layer fold-CV gate

Successful shards: `100/100`; checksums: `PASS`.

**FAIL**

## Exact MSE summary

| method | exact rep mean MSE | per-MLP mean MSE | median | q10 | q90 | worst |
|---|---:|---:|---:|---:|---:|---:|
| current | 2.677484e-06 | 2.677484e-06 | 2.322645e-06 | 1.245449e-06 | 4.780111e-06 | 8.641382e-06 |
| raw_haar | 2.732770e-06 | 2.732770e-06 | 2.421795e-06 | 1.167837e-06 | 4.264778e-06 | 8.628815e-06 |
| haar_cv | 2.691146e-06 | 2.691146e-06 | 2.371736e-06 | 1.353122e-06 | 4.274801e-06 | 9.038321e-06 |

## MSE ratio summary

| ratio | mean | median | q10 | q90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| current_over_cv | 1.0567 | 0.9936 | 0.5658 | 1.6762 | 0.3262 | 2.7318 |
| raw_over_cv | 1.0206 | 1.0023 | 0.8283 | 1.2262 | 0.7029 | 1.3855 |

## Three-rep Haar-CV decomposition

| component | mean | median | q10 | q90 |
|---|---:|---:|---:|---:|
| haar_cv bias_squared | 9.819731e-07 | 7.689571e-07 | 3.414837e-07 | 1.698644e-06 |
| haar_cv variance | 1.709173e-06 | 1.495751e-06 | 7.508824e-07 | 2.804331e-06 |
| haar_cv total | 2.691146e-06 | 2.371736e-06 | 1.353122e-06 | 4.274801e-06 |

## Frozen gate decisions

- `complete_100`: **PASS**.
- `checksums`: **PASS**.
- `haar_cv_mean_mse`: **FAIL**.
- `current_over_cv_mean_ratio`: **FAIL**.
- `current_over_cv_median_ratio`: **FAIL**.
- `current_over_cv_q10_ratio`: **FAIL**.
- `all_ratios_at_least_0.70`: **FAIL**.
- `haar_cv_mean_squared_bias`: **PASS**.
