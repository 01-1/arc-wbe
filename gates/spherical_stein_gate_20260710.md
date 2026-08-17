# Spherical Stein Haar fold-CV gate (pre-registration)

Status: PRE-REGISTERED BEFORE FLY. Research-only; no estimator or history
files are modified.

For each of 100 truth-bank MLPs and three independent replications, generate
8 independent Haar orthogonal bases with rows scaled by `sqrt(256)` and add
exact antipodes. Propagate the raw bias-free network in fp32. In parallel,
propagate the directional derivative induced by
`A0=W0@W0.T`, centered by `trace(A0)/d I` and Frobenius-normalized to
`sqrt(d)`. For every row, use `q=(x.T@A@x)/d`, `V=A@x-q*x`, and initialize
`dh=V`; apply each ReLU gate to both the forward and derivative paths.

At the final layer form `H=dh+f*(trace(A)-d*q)`. Pair antipodes as
`G=(f_pos+f_neg)/2`, `K=(H_pos+H_neg)/2`. Split bases 4/4. On each training
fold fit one scalar beta per output as centered `cov(K,G)/(var(K)+ridge)`,
with frozen ridge `1e-3*mean(var(K))`; correct the held-out mean by
`mean(G)-beta*mean(K)`, swap folds, and average. Multiply raw Haar and Stein
estimates by the exact `E[chi_d]/sqrt(d)` radial factor.

The paired current estimate is the fixed `hadamard_st3_b16` research route
with the same MLP and replication seed; it cannot tune the candidate. Truth
means are used only after fitting for post-hoc scoring. Every shard verifies
its stored weight checksum.

Pass requires all of: Stein mean MSE `<=1.6e-6`, current/Stein mean ratio
`>=1.35`, per-MLP median ratio `>=1.20`, q10 `>0.90`, minimum `>=0.70`, and
mean three-rep squared bias `<=1e-6`. No A/ridge/block sweep is allowed.

Frozen launch:

```text
make fly-payload FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/spherical_stein_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/spherical_stein_payload_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/spherical_stein_gate_20260710_fly.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```
