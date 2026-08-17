# Hadamard-oriented antithetic Gaussian LHS gate (pre-registration)

Status: **PREPARED, NOT LAUNCHED**. Research-only measurement. No estimator
or history files are modified and truth is accessed only after all route
vectors and label-free diagnostics are fixed.

## Frozen construction

For each of the 100 checksum-valid width-256/depth-32 truth-bank MLPs, use one
replication and exactly three 8192-row routes: corrected current b16,
independent-orientation LHS control, and Hadamard-oriented LHS candidate.
Each route has 4096 positive rows and exact row antipodes.

Generate one shared magnitude table. For every coordinate independently,
permute pair IDs `0..4095`, draw open-interval jitter, set
`u_low=(id+jitter)/8192`, and set `magnitude=-ndtri_dependency_free(u_low)`.
The reviewed dependency-free normal generator is imported read-only from
`sobol_runtime_feasibility_generator_20260710.py` and packaged with the run.

The independent LHS control uses an independent Rademacher sign per cell. The
candidate uses exactly 16 independent positive randomized Hadamard half-bases:
`H * column_flip_vector` per block, concatenated as positive bases only, never
interleaved or sliced with antipodes. Both LHS arms use the exact same shared
magnitudes. There is no row normalization, raw-input recolor, rotation, or
radial factor. Streams are separate and frozen for magnitudes, independent
signs, Hadamard signs, and the corrected current reference.

All routes use the exact global first-layer ReLU mean/covariance recolor,
fp32 centered strength-1.5 first-successor match, and fp32 L3 suffix. The
payload audits exact one-per-coordinate normal-probability strata, exact
antipodes, covariance diagonal/offdiagonal moments, and fixes all estimates
before reading truth.

## Frozen PASS rule

Require `100/100` valid checksums; candidate mean MSE `<=1.8e-6`; current /
candidate global ratio `>=1.25`; and per-MLP current/candidate median, q10,
and minimum ratios `>=1.10`, `0.85`, and `0.65`. Independent-LHS ratios are
diagnostic only. No estimator mode or Stage B is planned.

## Proposed command — PAUSE before launch

```text
make fly-payload FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/hadamard_lhs_v1_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/hadamard_lhs_v1_payload_20260710.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/hadamard_lhs_v1_gate_20260710_fly.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```
