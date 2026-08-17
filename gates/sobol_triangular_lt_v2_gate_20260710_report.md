# Sobol triangular linear-transform Stage-A v2 gate

Successful shards: `100/100`; checksums: `PASS`.

**FAIL**

| method | mean MSE | median | q10 | q90 | worst |
|---|---:|---:|---:|---:|---:|
| current | 2.546576e-06 | 1.921558e-06 | 1.031850e-06 | 4.797766e-06 | 1.412386e-05 |
| sobol_sphere | 2.679661e-06 | 2.183985e-06 | 1.044406e-06 | 5.315005e-06 | 1.397389e-05 |
| sobol_triangular | 2.977580e-06 | 2.362255e-06 | 1.020018e-06 | 5.268388e-06 | 2.235376e-05 |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_triangular | 1.1871 | 0.9140 | 0.4193 | 1.9944 | 0.0750 |
| sobol_over_triangular | 1.2146 | 0.8941 | 0.4851 | 2.5141 | 0.0689 |

## Importance concentration

| top k | mean | median | q10 | q90 |
|---:|---:|---:|---:|---:|
| 1 | 0.009636 | 0.009343 | 0.007962 | 0.011285 |
| 2 | 0.018236 | 0.017931 | 0.015746 | 0.020761 |
| 4 | 0.034039 | 0.033357 | 0.030585 | 0.038571 |
| 8 | 0.062961 | 0.061945 | 0.057241 | 0.069476 |
| 16 | 0.114870 | 0.113718 | 0.106747 | 0.122989 |
| 32 | 0.207252 | 0.205983 | 0.197323 | 0.217863 |

QR max-absolute error: mean `2.237856e-06`, worst `3.814697e-06`.
QR max-relative error: mean `4.266941e-07`, worst `7.804429e-07`.

## Stage-A gate decisions

- `complete_100`: **PASS**.
- `checksums`: **PASS**.
- `triangular_mean_mse`: **FAIL**.
- `current_over_triangular_global`: **FAIL**.
- `sobol_over_triangular_global`: **FAIL**.
- `current_over_triangular_median`: **FAIL**.
- `current_over_triangular_q10`: **FAIL**.
- `all_ratios_min`: **FAIL**.
