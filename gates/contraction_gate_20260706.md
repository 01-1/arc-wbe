# the reference entrant Contraction Gate (truth-bank edition)

Fly-bank research run. Each Machine rebuilt its bank MLP from seed, loaded only its truth-bank row, ran the perturbation and toy-state computations in place, and returned summary JSON. No estimator was run or scored.

## Checksum rebuild

Checksum rows verified: 100/100 matched. First row index `0` seed `6604632520249517929` local SHA256 `44f7b460863d2847b6edd0c4d8148fee2a5d9ac2d6ef6f8d59f92d7db2402563` vs bank `44f7b460863d2847b6edd0c4d8148fee2a5d9ac2d6ef6f8d59f92d7db2402563`.

## Q1. Perturbation contraction

| inject layer | type | MSE factor median [q10,q90] | amplitude factor median [q10,q90] | terminal/first MSE median |
|---:|---|---:|---:|---:|
| 2 | iid_gaussian | 0.907 [0.869,0.955] | 0.952 [0.932,0.977] | 0.0492 |
| 2 | bias_all | 0.968 [0.934,1.009] | 0.984 [0.967,1.004] | 0.307 |
| 2 | top2_bias | 0.955 [0.895,1.003] | 0.977 [0.946,1.002] | 0.193 |
| 2 | orthogonal_bias | 0.891 [0.849,0.948] | 0.944 [0.921,0.973] | 0.0277 |
| 8 | iid_gaussian | 0.904 [0.857,0.957] | 0.951 [0.926,0.978] | 0.108 |
| 8 | bias_all | 0.949 [0.902,1.003] | 0.974 [0.950,1.002] | 0.306 |
| 8 | top2_bias | 0.978 [0.934,1.022] | 0.989 [0.967,1.011] | 0.517 |
| 8 | orthogonal_bias | 0.897 [0.848,0.948] | 0.947 [0.921,0.974] | 0.0846 |
| 16 | iid_gaussian | 0.909 [0.855,0.956] | 0.954 [0.925,0.978] | 0.256 |
| 16 | bias_all | 0.943 [0.887,0.994] | 0.971 [0.942,0.997] | 0.426 |
| 16 | top2_bias | 0.977 [0.927,1.024] | 0.988 [0.963,1.012] | 0.71 |
| 16 | orthogonal_bias | 0.914 [0.869,0.957] | 0.956 [0.932,0.978] | 0.278 |
| 24 | iid_gaussian | 0.930 [0.858,1.007] | 0.964 [0.926,1.003] | 0.65 |
| 24 | bias_all | 0.936 [0.866,1.020] | 0.967 [0.931,1.010] | 0.661 |
| 24 | top2_bias | 0.981 [0.888,1.071] | 0.991 [0.943,1.035] | 0.848 |
| 24 | orthogonal_bias | 0.919 [0.848,0.982] | 0.959 [0.921,0.991] | 0.587 |

## Q2. Crude state-propagated toys

| toy | per-MLP slope median [q10,q90] | aggregate slope | MSE factor/layer | L31/L30 median [q10,q90] | final MSE median |
|---|---:|---:|---:|---:|---:|
| plain_particles_n512 | -0.0767 [-0.1059,-0.0379] | -0.0786 | 0.924 | 0.975 [0.821,1.124] | 5.56e-05 |
| rank2_reproject_n512 | -0.0006 [-0.0329,0.0404] | 0.0014 | 1.001 | 0.990 [0.829,1.140] | 0.114 |

## Q3. Distinguishability from plain sampling

| toy | corr plain | corr antithetic | corr the reference entrant | residual slope vs plain | residual log RMS | distinct call |
|---|---:|---:|---:|---:|---:|---|
| plain_particles_n512 | 0.995 | 0.997 | 0.989 | -0.0113 | 0.119 | False |
| rank2_reproject_n512 | -0.344 | -0.309 | -0.197 | 0.0686 | 1.815 | True |

## Decision

Overall preregistered call: **INCONCLUSIVE**. Relevant Q1 median MSE factor was `0.943`/layer; Q1 contract-scale call `True`.

Terminal-drop note: none of these crude propagated-state mechanisms directly supplies the reference entrant's additional ~25x final-layer discontinuity. That still requires an explicit final-layer allocation, refinement, or readout switch beyond smooth deep contraction.

## Run metadata

- Fly JSONL: `['paired_fly_logs/fingerprint_theory/contraction_gate_20260706_fly_results.jsonl', 'paired_fly_logs/fingerprint_theory/contraction_gate_20260706_fly_retry_missing_29_32_50_53.jsonl']`
- Bank MLP records: 100
- Failures: 0
- Machine-side Q1 samples: 1024
- Machine-side Q2 particles: 512
