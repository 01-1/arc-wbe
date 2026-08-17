# Unnormalized Gaussian Sobol RQMC Stage-A gate

Successful shards: `100/100`; checksums: `PASS`.

**FAIL**

| method | mean MSE | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current | 2.546583e-06 | 1.921486e-06 | 1.031921e-06 | 4.797766e-06 | 4.425275e-07 |
| iid_gaussian | 3.768793e-06 | 2.293030e-06 | 8.774979e-07 | 6.786283e-06 | 5.261705e-07 |
| sobol_gaussian | 3.558828e-06 | 2.484191e-06 | 1.038571e-06 | 7.525542e-06 | 4.366924e-07 |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_sobol | 1.0269 | 0.8156 | 0.3697 | 2.1596 | 0.0941 |
| iid_over_sobol | 1.4480 | 0.9800 | 0.3135 | 2.8598 | 0.1107 |

## Label-free input diagnostics

Radius, covariance, and coordinate-mean statistics are computed on positive representatives only; `antipode_max_abs` is computed on the constructed positive/negative pair and should be exactly zero.
### sobol_gaussian
- `radius_mean` mean `1.598434e+01`, median `1.598431e+01`, q10 `1.598380e+01`, q90 `1.598491e+01`.
- `radius_std` mean `7.075128e-01`, median `7.076993e-01`, q10 `6.989613e-01`, q90 `7.172106e-01`.
- `radius_q10` mean `1.508258e+01`, median `1.508167e+01`, q10 `1.506582e+01`, q90 `1.510123e+01`.
- `radius_q90` mean `1.689250e+01`, median `1.689203e+01`, q10 `1.687690e+01`, q90 `1.690773e+01`.
- `cov_rel_fro` mean `8.537326e-02`, median `8.545036e-02`, q10 `8.091545e-02`, q90 `8.962980e-02`.
- `cov_max_offdiag` mean `1.975675e-01`, median `2.166659e-01`, q10 `1.302008e-01`, q90 `2.239845e-01`.
- `coordinate_mean_max_abs` mean `3.512567e-04`, median `3.401497e-04`, q10 `2.737629e-04`, q90 `4.374210e-04`.
- `antipode_max_abs` mean `0.000000e+00`, median `0.000000e+00`, q10 `0.000000e+00`, q90 `0.000000e+00`.
### iid_gaussian
- `radius_mean` mean `1.598477e+01`, median `1.598595e+01`, q10 `1.596930e+01`, q90 `1.599601e+01`.
- `radius_std` mean `7.062035e-01`, median `7.071700e-01`, q10 `6.953731e-01`, q90 `7.149965e-01`.
- `radius_q10` mean `1.508247e+01`, median `1.507857e+01`, q10 `1.505947e+01`, q90 `1.510889e+01`.
- `radius_q90` mean `1.689163e+01`, median `1.689311e+01`, q10 `1.686441e+01`, q90 `1.691651e+01`.
- `cov_rel_fro` mean `2.505571e-01`, median `2.505703e-01`, q10 `2.492799e-01`, q90 `2.520594e-01`.
- `cov_max_offdiag` mean `6.667195e-02`, median `6.610226e-02`, q10 `6.213687e-02`, q90 `7.160363e-02`.
- `coordinate_mean_max_abs` mean `4.667015e-02`, median `4.550837e-02`, q10 `4.017777e-02`, q90 `5.406454e-02`.
- `antipode_max_abs` mean `0.000000e+00`, median `0.000000e+00`, q10 `0.000000e+00`, q90 `0.000000e+00`.

## Frozen gate decisions

- `complete_100`: **PASS**.
- `checksums`: **PASS**.
- `sobol_mean_mse`: **FAIL**.
- `current_over_sobol_global`: **FAIL**.
- `current_over_sobol_median`: **FAIL**.
- `current_over_sobol_q10`: **FAIL**.
- `current_over_sobol_min`: **FAIL**.
