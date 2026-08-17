# Sobol pilot-Jacobian linear-transform gate (v2 infrastructure correction)

Status: **PRE-REGISTERED BEFORE V2 FLY**. Research-only. No estimator or history
file is modified; truth is accessed only after all route estimates are fixed.
The v1 launch is recorded as zero-row infrastructure-only (`scipy` was absent
from the Fly image) and did not consume Stage A. This v2 changes only the
Sobol generator infrastructure: it imports the reviewed dependency-free
`sobol_normal_rows` implementation, whose LMS Sobol uniforms match SciPy
exactly, and packages that generator with the payload.

## Frozen Stage-A construction

On each of the 100 truth-bank width-256/depth-32 MLPs, run exactly one fixed
replication with exactly three 8192-row routes:

1. corrected current `hadamard_st3_b16`: 16 genuinely independent positive
   Hadamard half-bases, exact antipodes, no interleaved helper;
2. unrotated Owen-scrambled Sobol sphere;
3. pilot-Jacobian linear-transform Sobol sphere.

Both Sobol routes use the exact same 4096 Sobol normal-score rows from the
reviewed dependency-free `sobol_normal_rows(seed)` generator, which is the
fixed-d=256, m=12 LMS-scrambled Sobol construction matching SciPy's uniforms
exactly. Exact antipodes are formed through the positive/negative ReLU halves
after the first fp32 L3 Strassen matmul. No radial factor is applied.
All three routes inherit the current global first-layer exact ReLU
mean/covariance recolor, fp32 centered first-successor variance match with
strength `1.5`, and fp32 L3 propagation.

For the LT route, an independent fixed stream generates exactly 8 normalized
Gaussian pilot input directions and 8 independent Rademacher output probes.
Each pilot is forwarded through the raw fp32 MLP with every ReLU gate saved;
the scalar Rademacher-probed final output is backpropagated exactly to one
256-vector input gradient. The 256x8 gradient matrix receives a complete QR.
The first 8 Q columns are deterministically sign-normalized by the QR diagonal;
the complete 256x256 Q defines `x = z @ Q.T`, placing the pilot-gradient span
in Sobol coordinates `0:8`. The pilot singular-value energy fractions for
top 1/2/4/8 are reported but cannot affect Q or estimates.

Frozen streams are:

```text
sobol   = 0x50B01_0710
current = 0xC0A1_0710
pilot   = 0x4A43_0710
derived = (mlp_seed ^ stream ^ (rep * 0x9E3779B9)) % 2**32
```

## Stage-A PASS rule

Require all: 100/100 returned shards and checksums; LT mean final MSE `<=
1.8e-6`; current/LT global mean-MSE ratio `>=1.25`; unrotated/LT global ratio
`>=1.15`; per-MLP current/LT median `>=1.10`, q10 `>=0.85`, and minimum
`>=0.65`.

If any Stage-A condition fails, stop and report exact metrics. If it passes,
run Stage B immediately with the same fixed construction and streams, using
three replications and no parameter changes.

## Stage-B PASS rule

Require LT mean final MSE `<=1.6e-6`; current/LT global ratio `>=1.35`; median
`>=1.20`; q10 `>=0.90`; minimum `>=0.70`; and the three-rep LT squared-bias
proxy `<=1e-6`. Require 100/100 returned shards and checksums.

## Frozen commands

Stage A:

```text
make fly-payload FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/sobol_jacobian_lt_v2_stagea_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/sobol_jacobian_lt_v2_payload_20260710.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/sobol_jacobian_lt_v2_stagea_fly_20260710.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```

Stage B, only after Stage-A PASS:

```text
make fly-payload FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/sobol_jacobian_lt_v2_stageb_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/sobol_jacobian_lt_v2_payload_20260710.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/sobol_jacobian_lt_v2_stageb_fly_20260710.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```

One packaging-only retry is allowed if and only if packaging fails before any
Machine runs; no runtime retry, parameter change, or partial-result
substitution is allowed.
