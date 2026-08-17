# Sobol pilot-Jacobian LT Stage A

Returned rows: `98/100`; failures: `0`; checksums: `PASS`.
Observed replication counts: `[1]`.

**FAIL**

| method | mean MSE | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current | 2.519012e-06 | 1.862777e-06 | 1.031387e-06 | 4.587188e-06 | 4.425275e-07 |
| sobol_unrotated | 2.562714e-06 | 1.933702e-06 | 9.588202e-07 | 4.974016e-06 | 3.789688e-07 |
| sobol_lt | 3.528917e-06 | 2.219147e-06 | 9.836681e-07 | 5.147682e-06 | 4.727838e-07 |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_lt | 1.1612 | 0.9720 | 0.3247 | 2.0654 | 0.0683 |
| unrotated_over_lt | 1.1249 | 0.8155 | 0.3586 | 2.0887 | 0.1370 |

## Bias/variance proxy

- `current`: bias² mean `2.519012e-06`, variance mean `0.000000e+00`.
- `sobol_unrotated`: bias² mean `2.562714e-06`, variance mean `0.000000e+00`.
- `sobol_lt`: bias² mean `3.528917e-06`, variance mean `0.000000e+00`.

## Pilot singular-value concentration

- `top_1`: mean `0.297485`, median `0.284131`.
- `top_2`: mean `0.492908`, median `0.483669`.
- `top_4`: mean `0.754580`, median `0.755834`.
- `top_8`: mean `1.000000`, median `1.000000`.

## Gate decisions

- `complete_100`: **FAIL**.
- `checksums`: **PASS**.
- `returned_rows_no_failures`: **FAIL**.
- `one_fixed_replication_stage_shape`: **PASS**.
- `lt_mean_mse`: **FAIL**.
- `current_over_lt_global`: **FAIL**.
- `unrotated_over_lt_global`: **FAIL**.
- `current_over_lt_median`: **FAIL**.
- `current_over_lt_q10`: **FAIL**.
- `current_over_lt_min`: **FAIL**.
