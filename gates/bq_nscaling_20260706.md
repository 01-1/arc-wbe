# NNGP BQ Error-vs-N Scaling and Kernel Spectrum

Scope: offline analysis only. No estimator edits, no Fly, no network, no pytest. All new artifacts are under `paired_fly_logs/fingerprint_theory/`.

## Method

- Reused the validated depth-32 arc-cosine NNGP kernel and current randomized antithetic Hadamard sign-block design.
- Pair means are exact: each `512`-point antithetic block pair reduces to the Walsh spectrum of the relative sign mask.
- Optimal BQ solves are exact in the block-constant invariant subspace. The right hand side `k_mu` is constant for all equal-radius sign points, so the full solve reduces to the block-pair mean matrix.
- The `8192` spectrum of `K/N` is exact via Walsh diagonalization into `256 * 2` small block matrices.
- `m` and `k_mu` use the prior validated MC estimates from `nngp_design_pretest_results.json`; the calibration step anchors the model to measured estimator variance at `N=8192`.

## Error Scaling

| N | blocks | equal err^2 | optimal err^2 | equal/optimal | pair mean | MC c/N through grader | MC c/N through Fly |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 4 | 0.000119642968 | 0.000118761983 | 1.007418 | 0.974739317 | 8e-06 | 9.44e-06 |
| 4096 | 8 | 0.000114979688 | 0.000114107543 | 1.007643 | 0.974734653 | 4e-06 | 4.72e-06 |
| 8192 | 16 | 0.000112648496 | 0.000111780753 | 1.007763 | 0.974732322 | 2e-06 | 2.36e-06 |
| 16384 | 32 | 0.000111482486 | 0.00011061694 | 1.007825 | 0.974731156 | 1e-06 | 1.18e-06 |
| 32768 | 64 | 0.000110899345 | 0.000110034897 | 1.007856 | 0.974730573 | 5e-07 | 5.9e-07 |

Pair-mean estimator standard error: `0`, because this run used the exact antithetic-Hadamard block sum rather than random pair sampling.

## Spectrum

`K/N` eigendecomposition at `N=8192` used `exact_walsh_antithetic_block_diagonalization`.
Top eigenvalue: `0.974732322`. Sum of eigenvalues: `1`.

| fit range | beta in lambda_k ~ k^-beta | R^2 |
|---|---:|---:|
| ranks 2-500 | 0.5224 | 0.5012 |
| ranks 20-500 | 0.7303 | 0.6060 |
| ranks 50-500 | 0.9786 | 0.6855 |

Top 20 eigenvalues:

`0.974732, 1.35005e-05, 1.34992e-05, 1.34989e-05, 1.34988e-05, 1.34981e-05, 1.3498e-05, 1.34977e-05, 1.34976e-05, 1.34974e-05, 1.34974e-05, 1.34971e-05, 1.3497e-05, 1.3497e-05, 1.34968e-05, 1.34967e-05, 1.34966e-05, 1.34964e-05, 1.3496e-05, 1.34959e-05`

The full top-500 list is in `bq_nscaling_20260706_results.json`.

## Calibration

| anchor | model/reality @ 8192 | observed/model multiplier | calibrated optimal err^2 @ 32768 | calibrated optimal err^2 @ 30720 | MC c/N @ 30720 |
|---|---:|---:|---:|---:|---:|
| grader | 56.3x | `0.017754` | 1.95e-06 | 1.95e-06 | 5.33e-07 |
| fly_net_of_floor | 47.7x | `0.02095` | 2.31e-06 | 2.31e-06 | 6.29e-07 |

Note: the table names the multiplicative calibration as observed/model; multiply raw NNGP err^2 by this ratio to match the measured `N=8192` variance anchor.

## Verdict

**EVALUATION-BASED LANE CLOSED** for the cluster's top entry under this validated average-case model. The calibrated optimal prediction near `30k` evaluations remains above `1e-6`, at least `10x` the entry-1 `~1e-7` error budget, and the kernel spectrum does not show the fast tail decay needed for quadrature superconvergence.
