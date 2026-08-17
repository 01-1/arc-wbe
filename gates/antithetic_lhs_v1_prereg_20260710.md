# Antithetic Gaussian Latin-hypercube gate v1 (pre-registration)

Research-only Stage-A measurement; no estimator changes. Width `d=256`, total
rows `N=8192`, positive representatives `M=4096`. For each coordinate,
pair IDs 0..4095 are independently permuted across rows, independent jitter
`v in (0,1)` is drawn and clipped to the open-interval-safe bounds
`[2^-53, 1-2^-53]`, and `u_low=(pair_id+v)/N`. The reviewed dependency-free
`ndtri_dependency_free` implementation converts `u_low` to a negative Gaussian
quantile; an independent orientation bit chooses `z_low` or `-z_low` per
coordinate. The route uses these positive representatives and exact row
antipodes, with no row normalization, rotation, radial factor, or raw-input
recoloring. Values are cast to fp32 before the audited current route transforms.

For each of 100 checksum-verified truth-bank MLPs and one fixed replication,
measure: corrected current 16-independent-Hadamard-block b16; ordinary iid
standard-normal positive half with exact antipodes; and this antithetic Gaussian
LHS candidate. All vectors are fixed before truth access. Label-free LHS audits
include exact one-per-stratum coverage, max full-sample coordinate mean, row-radius
summary, diagonal second moments, covariance Frobenius relative error, and max
off-diagonal; the
IID control reports the same coordinate/radius/covariance fields (with exact
strata marked not applicable).

Frozen PASS requires 100/100 valid checksums, LHS MSE `<=1.8e-6`, global
current/LHS ratio `>=1.25`, and per-MLP median/q10/min ratios
`>=1.10/0.85/0.65`. IID is diagnostic only. A three-rep confirmation requires
separate authorization; no estimator mode is prepared here.
