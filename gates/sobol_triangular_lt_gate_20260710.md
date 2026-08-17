# Sobol triangular linear-transform Stage-A gate (pre-registration)

Status: **PRE-REGISTERED BEFORE FLY**. Research-only; no estimator or history
files are modified. This gate uses one fixed replication per MLP and stops if
Stage A fails.

## Frozen Stage-A construction

For each of 100 truth-bank MLPs, compare exactly three 8192-row routes:

1. corrected `hadamard_st3_b16` with 16 genuinely independent positive
   Hadamard half-bases and exact antipodes;
2. unrotated Owen-scrambled Sobol sphere;
3. the triangular linear-transform Sobol sphere.

The two Sobol routes share exactly the same 4096 normal-score Sobol rows:
scrambled `scipy.stats.qmc.Sobol(d=256, scramble=True).random_base2(m=12)`,
clipped only to `(2^-53, 1-2^-53)`, transformed by `ndtri`, normalized rowwise,
and scaled by `sqrt(256)`. Both use exact antipodes, the current global
first-layer ReLU mean/covariance recolor, fp32 centered application of the
strength-1.5 first-successor variance match, and fp32 L3 propagation. No
radial multiplier is applied.

## Frozen triangular label-free transform

From an independent fixed pilot stream, generate exactly 8 normalized-Gaussian
pilot inputs and 8 independent Rademacher final-output probes. Forward the raw
fp32 MLP and backprop each probe scalar through the ReLU gates only to the first
hidden post-ReLU activation. The importance of first-hidden unit `j` is the
mean squared gradient over the 8-by-8 pilot/probe combinations. Sort descending
with index-stable ties. Compute a complete QR of `W0[:, order]`, normalize QR
diagonal signs deterministically, and set `x = z @ Q.T`. Before reading truth,
verify on a fixed small subset that `(x @ W0)[:, order]` agrees with `z @ R` to
fp32 tolerance. Report top-1/2/4/8/16/32 importance shares and QR max absolute
and relative errors.

## Frozen Stage-A pass rule

Stage A passes only with 100/100 valid checksums and all of: triangular mean
final MSE `<=1.8e-6`; global current/triangular ratio `>=1.25`; global
unrotated/triangular ratio `>=1.15`; per-MLP current/triangular median `>=1.10`;
q10 `>=0.85`; and no per-MLP ratio below `0.65`. If Stage A passes, a separate
three-rep Stage B is authorized with unchanged parameters and thresholds
triangular MSE `<=1.6e-6`, current/triangular `>=1.35`, median `>=1.20`, q10
`>=0.90`, minimum `>=0.70`, and mean three-rep squared-bias proxy `<=1e-6`.

Truth is read only after all three estimates and diagnostics are fixed. Every
shard verifies its stored weight checksum. No parameter sweep or estimator
scoring is permitted.

Planned unique artifacts use the `sobol_triangular_lt_*_20260710` prefix.
