# Haar-sphere first-layer control-variate gate (pre-registration)

Status: **PRE-REGISTERED BEFORE FLY**. This is a research-only payload; it
does not modify `estimator.py` or the estimator history.

## Frozen construction

For each of the 100 truth-bank MLPs and each of three independent
replications, generate 16 independent Haar orthogonal `256 x 256` bases by
Gaussian QR with deterministic diagonal-sign normalization. Transpose each
basis into row directions and scale by `sqrt(256)`. Add the exact antipodes
without drawing another basis. Propagate both halves through the raw network
in fp32 using only the existing L3 Strassen matmul path; do not use first-layer
recoloring, variance matching, or sample-dependent transport.

For each paired row, define `C = 0.5 * abs(pre_half)` and
`G = 0.5 * (final_positive + final_negative)`. Split the 16 bases into fixed
8/8 folds. On each training fold, center `C` and `G` and fit one multivariate
ridge coefficient matrix with the preregistered penalty
`lambda = 0.1 * trace(Ct.T @ Ct) / 256`. On the held-out fold use

`mean(G_test) + (mu_C - mean(C_test)) @ B`,

where the exact fixed-sphere feature expectation is

`mu_C_j = sqrt(d) * Gamma(d/2) /
          (2 * sqrt(pi) * Gamma((d+1)/2)) * ||W0[:,j]||`.

Swap folds and average the two held-out estimates. Evaluate the exact radial
factor with `lgamma` and multiply both the Haar-CV estimate and the direct raw
Haar estimate by

`r_d = sqrt(2) * Gamma((d+1)/2) /
       (sqrt(d) * Gamma(d/2))`.

The current `hadamard_st3_b16` research helper is run with the same three
replication seeds for a paired absolute comparison; it cannot tune the Haar
candidate.

## Frozen pass rule

Aggregate exact final-layer MSE against the truth-bank final means. Report
mean, per-MLP median/q10/q90, worst tail, current/Haar-CV and raw-Haar/Haar-CV
ratios, and a three-rep squared-bias/within-rep-variance decomposition. The
gate passes only if all of these hold: Haar-CV mean MSE `<= 1.6e-6`,
current/Haar-CV ratio `>= 1.35`, per-MLP median ratio `>= 1.20`, q10 ratio
`>= 0.90`, no per-MLP ratio `< 0.70`, and mean three-rep squared bias
`<= 1.0e-6`. No ridge or block sweep is permitted.

## Legality and artifacts

The payload uses only rebuilt MLP weights, fresh Haar directions, and the
truth-bank final means for post-hoc research scoring. It verifies every bank
weight checksum before propagation. No public/private grader outputs or
estimator score path is used. Planned artifacts are
`haar_sphere_foldcv_payload_20260710.py`,
`haar_sphere_foldcv_aggregate_20260710.py`,
`haar_sphere_foldcv_manifest_20260710.json`, the Fly JSONL, and the aggregate
JSON/report.
