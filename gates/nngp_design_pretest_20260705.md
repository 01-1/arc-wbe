# NNGP-Kernel Cubature-Optimality Pre-Test

Scope: offline analysis only. No estimator edits, no tracked-file edits, no Fly, no network, no pytest.

## Kernel Validation

Validated against 200 self-generated width-256/depth-32 He MLPs.

| pair | dot/width | NNGP | empirical output product | rel error |
|---|---:|---:|---:|---:|
| [0, 0] | 1.0000 | 1 | 1.02323 | 2.32% |
| [0, 1] | 0.0000 | 0.97472 | 0.976606 | 0.19% |
| [0, 2] | -1.0000 | 0.973418 | 0.95739 | -1.65% |
| [3, 4] | -0.0010 | 0.974718 | 1.07329 | 10.11% |
| [3, 5] | -0.0079 | 0.974699 | 1.09774 | 12.62% |
| [4, 5] | -0.0223 | 0.97466 | 1.03309 | 6.00% |

## Error Table

| design | weighting | err^2 | ratio vs current equal | notes |
|---|---:|---:|---:|---|
| current_hadamard_antithetic | equal | 0.0001126485 | 1.0000 | wKw 0.97473232 |
| current_hadamard_antithetic first 1024 | BQ optimal | 0.00012807365 | n/a | subset equal/optimal 1.0070 |
| iid_gaussian_sphere | equal | 0.00011346353 | 1.0072 | wKw 0.97473314 |
| multiradius_sign | equal | 0.00011179645 | 0.9924 | wKw 0.97270122 |
| more_smaller_orthogonal_blocks | equal | 0.0001129635 | 1.0028 | wKw 0.97473264 |

## Gate Verdict

Best full-8192 equal-weight point-set improvement over current equal weights: `1.008x`.
Candidate is **DEAD** under the pre-registered `>= 1.3x` gate.
Current-design row-sum relative std: `1.605e-09`; relative range `5.750e-09`. With constant `k_mu` on equal-radius sign points, this is the direct BQ equal-weight optimality diagnostic.

Caveat: this is the raw NNGP prior/design proxy. The production estimator also applies first-layer recoloring and variance matching, so this bounds cubature-design headroom rather than proving production MSE exactly.
