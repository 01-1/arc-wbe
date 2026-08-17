# Spherical Stein Haar fold-CV gate

Successful shards: `100/100`; checksums: `PASS`.

**FAIL**

| method | mean MSE | median | q10 | q90 |
|---|---:|---:|---:|---:|
| current | 2.748727e-06 | 2.410619e-06 | 1.312123e-06 | 4.749918e-06 |
| raw_haar | 5.657141e-06 | 4.730585e-06 | 2.578641e-06 | 9.336716e-06 |
| stein | 5.702312e-06 | 4.745822e-06 | 2.634567e-06 | 9.704327e-06 |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_stein | 0.5313 | 0.4817 | 0.3026 | 0.8076 | 0.1751 |
| raw_over_stein | 0.9961 | 0.9933 | 0.9558 | 1.0395 | 0.9262 |

## Gate decisions

- `complete_100`: **PASS**.
- `checksums`: **PASS**.
- `stein_mean_mse`: **FAIL**.
- `current_over_stein_mean`: **FAIL**.
- `current_over_stein_median`: **FAIL**.
- `current_over_stein_q10`: **FAIL**.
- `all_ratio_min`: **FAIL**.
- `stein_bias`: **FAIL**.
