# Antipodal odd-state Rao--Blackwell K8 gate (pre-registration)

## Purpose

Test one new conditional-law carrier, not another input sampler variant.  The
current antithetic route propagates both members of every natural `x,-x` pair
through all 32 ReLU layers.  This candidate propagates both members exactly
through an initial prefix, then Rao--Blackwellizes the odd member under a
frozen Gaussian conditional model and propagates only the even member through
the remaining suffix.

The gate tests the closure science only.  It does not authorize an estimator
mode, a parameter sweep, or a Fly scorer run.  Its `B=27` projection is
conditional on two later exact implementation identities: batched FWHT for
the randomized-Hadamard first layer and folding the first covariance recolor
into `W1`.  Those implementation changes are not part of this gate.

## Frozen current route

- Width 256, depth 32, 16 genuinely independent positive randomized
  Hadamard bases followed by their exact antipodes.  Do not use the old
  interleaved `_block_rows` helper.
- Three fixed, independent streams derived only from the rebuilt MLP seed and
  fixed stream constants.
- Exact global first-ReLU mean/covariance recolor against `W0.T @ W0`.
- Current strength-1.5 first-successor variance update.
- Float32 ensemble propagation and level-3 Strassen arithmetic matching the
  current estimator.
- Pair alignment remains `positive_rows[p] <-> negative_rows[p]` after every
  exact prefix operation.

The initial recolored first-ReLU activation is state 0.  The prefix then
applies weights `W1,...,W8`, including ReLU and the current update after `W1`.
The branch is the activation after `W8`: exactly eight post-recolor propagated
ReLU layers have completed.  The closure suffix uses `W9,...,W31` (23
layers).  The exact current route continues separately through `W31`.

## Frozen K8 closure

At the branch, for aligned activation rows `h_plus,h_minus`, define

```
e = (h_plus + h_minus) / 2
o = (h_plus - h_minus) / 2
q = o**2
g = mean(q)
r[p] = mean_i(q[p,i]) / max(g, tiny)
c[i] = mean_p(q[p,i])
```

Thus `mean(r)=1`, `mean(c)=g`, and `r[:,None]*c[None,:]` preserves both row
and coordinate means of `q`.  Record its relative Frobenius residual.

For each frozen suffix weight matrix `W`, compute entirely inside flopscope:

```
a[p,j] = e[p,:] @ W[:,j]
base[j] = sum_i c[i] * W[i,j]**2
s2[p,j] = max(r[p] * base[j], tiny)
s = sqrt(s2)
alpha = a / s
phi = NormalPDF(alpha)
Phi = NormalCDF(alpha)
e_new = s*phi + a*Phi
A = (a**2 + s2)*Phi + a*s*phi
C = 0                                                   if a <= 0
    (a**2-s2)*(2*Phi-1) + 2*a*s*phi                    if a > 0
v_o = max((A-C)/2, 0)
```

`e_new` is the exact scalar conditional mean of the next pair-even ReLU state
when the pair-odd preactivation is `N(0,s2)`.  `v_o` is the corresponding
conditional second moment of the pair-odd state.  Refactor `v_o` with the same
`g,r,c` rule before the next layer.  Propagate `e_new` as float32 and report
`mean(e_new,axis=0)` for closure-layer means.  This rank-one nonnegative
row-by-coordinate variance carrier is the only candidate.

## Truth and accounting boundary

All current/candidate vectors for all three streams and every label-free
diagnostic must be fixed under flopscope before the truth-bank array is read.
NumPy is allowed only for checksum/input I/O and scalar/final serialization at
the established boundary.  Each Machine rebuilds the MLP and validates its
weight SHA256 before a result can count.

## Frozen Stage A

- One all-100 generic `make fly-payload` run.
- Three independent route/closure replications per MLP.
- `FLY_MLPS=100`, `FLY_MIN_RESULTS=100`, and both result windows 420 seconds.
- Unique JSONL/results/report paths containing `odd_rb_k8_v1_20260710`.
- No automatic rerun, Stage B, mode, sweep, blend, or estimator/history edit.

The aggregator reports completeness, explicit failures, pending indices,
duplicates, checksums, per-rep and three-rep-mean final-layer current/candidate MSE,
global and per-MLP ratios, factor residuals, and all numerical-range checks.
All MSE, bias, variance, and projection thresholds in this gate refer only to
the final `W31` mean vector of shape `(256,)`; all 23 closure steps remain
computed and diagnosed. For candidate and current separately, with `M1` the
mean one-rep final-layer MSE and `M3` the final-layer MSE of the three-rep
prediction mean, compute

```
bias2 = max((3*M3 - M1)/2, 0)
var16 = max(M1 - bias2, 0)
projected_MSE(B) = bias2 + var16 * 16/B, B in {25,26,27}
```

Overall PASS is common integrity AND every science gate:

- exactly 100/100 checksum-valid rows;
- zero failures, pending rows, and duplicates;
- candidate `M1 <= 2.65e-6`;
- global `current_M1 / candidate_M1 >= 0.98`;
- candidate `bias2 <= 2.5e-7`;
- candidate projected `B=27` MSE `<= 1.52e-6`;
- every emitted estimate/diagnostic finite, every `s2 > 0`, and every
  `v_o >= 0` up to a `1e-12` serialization tolerance.

Passing Stage A only permits a separate implementation/economics audit of the
FWHT, folded recolor, and K8 closure composition.  Only a later plain
`make fly` result below `1.6e-7` can establish the repository goal.

## Exact proposed command

```sh
FLY_MLPS=100 FLY_MIN_RESULTS=100 \
FLY_MAX_RESULT_SECONDS=420 FLY_PAYLOAD_MAX_RESULT_SECONDS=420 \
FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/odd_rb_k8_v1_manifest_20260710.json \
FLY_PAYLOAD_FILES="paired_fly_logs/fingerprint_theory/odd_rb_k8_v1_payload_20260710.py paired_fly_logs/fingerprint_theory/odd_rb_k8_v1_aggregate_20260710.py estimator.py local_engine.py analysis/truth_bank/truth_bank.npz" \
FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/odd_rb_k8_v1_stagea_fly_20260710.jsonl \
make fly-payload
```

No Fly launch has occurred; coordinator approval is required.
