# Odd low-rank transport v1 truth-bank gate (pre-registration)

## Purpose and authorization boundary

This frozen Stage-A gate tests whether an aligned antipodal odd state can be
transported through the suffix by a scheduled low-rank left-row carrier without
destroying its coordinatewise marginal energy. It is one closure-science
candidate against the corrected current `b16` route, not a sampler sweep or a
second estimator mode.

A PASS authorizes only a separate estimator implementation/economics audit.
There is no automatic mode, Stage B, rerun, blend, follow-up, or edit to
`estimator.py` or `ESTIMATOR_HISTORY.md`.

## Frozen corrected-current route

- Width 256 and depth 32.
- Sixteen genuinely independent positive randomized Hadamard bases in their
  original block order, followed by exact antipodes. Each positive row remains
  aligned with the corresponding negative row after the global recolor.
- Three independent stable route streams. All four subspace-refresh stream
  families are distinct from the route streams and are independently derived
  for each rep and refresh from only the rebuilt MLP seed and fixed constants.
- Exact global first-ReLU Gaussian mean/covariance recolor.
- The current strength-1.5 first-successor variance update, with fp32 centered
  scale application and writeback.
- Float32 state/weight propagation with level-3 Strassen arithmetic.
- Layers `W1..W4` are exact and shared. The current arm then continues exactly
  through every `W5..W31`; its final vector is the mean final pair-even state.

The payload rebuilds each MLP and verifies the SHA256 of all fp32 weights against
the bank before the row can count.

## Frozen low-rank carrier

After shared layer 4, split aligned rows as

```text
E = (Hplus + Hminus)/2
O = (Hplus - Hminus)/2.
```

The schedule is fixed:

- refresh rank 64 after layer 4; use for layers 5 through 8;
- refresh rank 32 after layer 8; use for layers 9 through 16;
- refresh rank 16 after layer 16; use for layers 17 through 24;
- refresh rank 8 after layer 24; use for layers 25 through 31.

At each refresh, with `r` the scheduled rank and oversampling fixed to 8, use
one independent stable-stream Rademacher sketch entirely under flopscope:

```text
Omega: d x (r+8), entries +/-1/sqrt(r+8)
Y = O @ Omega
G = Y.T @ Y + jitter*I
L = lower_cholesky(G)
Q0 = solve(L, Y.T).T
B0 = Q0.T @ O
S = symmetric(B0 @ B0.T)
Q = Q0 @ U_top
```

`eigh` is ascending and the retained top-r eigenvectors are reversed into
descending order, fixing deterministic eig ordering. At refresh only, report
the unscaled projection `Q@(Q.T@O)`: relative Frobenius residual, captured
energy, projection identity residual, `Q` orthogonality, unjittered minimum
Gram eigenvalue, jitter, post-jitter minimum, retained minimum range eigenvalue,
maximum range eigenvalue, and finiteness.

At every transported layer, with fixed `Q`, compute

```text
C = Q.T @ O
Qgram = Q.T @ Q                         # once per fixed-Q segment
target_i = sum_p O[p,i]^2
projected_i = sum_k C[k,i] * (Qgram@C)[k,i]
s_i = sqrt(max(target_i,tiny) / max(projected_i,tiny))
C_scaled = C * s[None,:]
A = E @ W
B = Q @ (C_scaled @ W)
plus = relu(A+B)
minus = relu(A-B)
E = (plus+minus)/2
O = (plus-minus)/2
```

There is no scale clipping beyond the frozen tiny numerator/denominator
protection. This preserves the signed rank-r row geometry while restoring each
input-coordinate odd second moment before `W`. The post-scale coordinate energy
is checked with the same `Qgram` small-matrix identity; no per-layer `Q@C` is
materialized for diagnostics, so the carrier retains one `P x r x d` apply per
layer. Record each layer's maximum and mean coordinate relative energy error and
scale minimum/median/maximum. The candidate final vector is `mean(E, axis=0)`.

## Label-free block correction diagnostic

Before truth access, retain the exact and candidate final pair-even rows in the
original 16 Hadamard blocks of 256 aligned pairs. Average pairs within each block,
then center the 16 block means separately for every output coordinate. Only after
that per-coordinate centering, pool block-coordinate values and report per rep:

```text
Var(exact_block_mean - candidate_block_mean) / Var(exact_block_mean)
corr(exact_block_mean, candidate_block_mean)
Var(candidate_block_mean) / Var(exact_block_mean)
```

The aggregator reports distributions of all three statistics and their raw
pooled variances. This diagnostic asks whether an unbiased two-level correction
could plausibly pay. It is label-free, diagnostic-only, not a second candidate,
and changes neither standalone vector nor PASS threshold.

## Dtypes and truth boundary

All estimator weights, route states, sketches, `Q0`, `B0`, `Q`, `Qgram`, `C`,
energy scaling, and transported `A/B/E/O` operations are fp32. Small Gram,
Cholesky/solve, and eigendecomposition operations are fp64 for stability; their
materialized bases are cast back to fp32. To reproduce the corrected current
route exactly, its established first-recolor and first-successor moment algebra
is fp64, while centered applications, scale application, recolor writeback, and
all propagated states are fp32. Every scientific operation uses flopscope.

For all three reps, exact current vectors, candidate vectors, refresh/layer
diagnostics, pair reconstruction, and block diagnostics are fixed before
`bank["truths"]` is accessed. NumPy is used only for bank/checksum I/O and final
scalar/vector serialization at that established boundary.

## Frozen numerical tolerances

These tolerances are fixed before any run:

- shared layer-4 pair reconstruction maximum relative Frobenius error `<= 2e-7`;
- `Q.T@Q` relative Frobenius error `<= 2e-5`;
- projection residual and captured energy each within `[0,1]` up to `5e-5`;
- `abs(captured_energy + projection_residual^2 - 1) <= 5e-4`;
- Gram jitter strictly positive, post-jitter minimum eigenvalue strictly
  positive, and retained range eigenvalue `>= -5e-5`;
- maximum coordinatewise post-scale odd-energy relative error `<= 2e-4`;
- all scales strictly positive and ordered `min <= median <= max`;
- every pair/Q/projection/Gram/energy value finite.

Block diagnostic variances nonnegative, exact block variance `> 1e-12`, and
correlation in `[-1,1]` up to `5e-5` are reported as a separate diagnostic
integrity flag. That flag is explicitly non-gating and is excluded from the
overall verdict.

## Frozen Stage A and aggregation

Run exactly one all-100 generic payload campaign with three reps per MLP,
`FLY_MLPS=100`, `FLY_MIN_RESULTS=100`, and both runner/payload result windows at
420 seconds. Use the unique `odd_lr_transport_v1_20260710` paths. There is no
automatic follow-up.

For current and candidate separately, report per-rep mean MSE, one-rep mean MSE
`M1`, and MSE `M3` of the three-rep prediction mean. Compute

```text
bias2 = max((3*M3 - M1)/2, 0)
var16 = max(M1 - bias2, 0)
projected_MSE(B) = bias2 + var16*16/B, B in {25,26,27}.
```

Also report the current sanity decomposition, candidate decomposition, global
`current_M1/candidate_M1`, its per-MLP distribution, all refresh/layer numerical
diagnostic distributions, and the diagnostic-only block-correction distributions.

Overall PASS is common integrity AND every science/numerical gate:

- exactly 100 rows indexed `0..99`, all bank weight checksums valid;
- zero failures, pending rows, duplicates, or schema-invalid rows;
- candidate `M1 <= 2.80e-6`;
- global `current_M1 / candidate_M1 >= 0.90`;
- candidate `bias2 <= 2.5e-7`;
- candidate projected `B=27` MSE `<= 1.52e-6`;
- all frozen pair/Q/projection/Gram/energy numerical diagnostics above are valid.

The block-correction integrity flag and its distributions are always reported
but do not participate in common integrity, science gates, or overall PASS.

## Exact proposed command

```sh
FLY_MLPS=100 FLY_MIN_RESULTS=100 \
FLY_MAX_RESULT_SECONDS=420 FLY_PAYLOAD_MAX_RESULT_SECONDS=420 \
FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_manifest.json \
FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_payload.py paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_aggregate.py analysis/truth_bank/truth_bank.npz" \
FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_stagea_fly.jsonl \
make fly-payload
```

After a coordinator-approved run only, aggregate with:

```sh
python paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_aggregate.py \
  paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_stagea_fly.jsonl \
  --output paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_stagea_results.json \
  --report paired_fly_logs/fingerprint_theory/odd_lr_transport_v1_20260710_stagea_report.md
```

No Fly launch has occurred. Coordinator audit is required before use.
