# Sobol-sphere recolor v2 gate (pre-registration)

Status: PRE-REGISTERED BEFORE FLY. Research-only; no estimator or history
files are modified. The original v1 gate produced zero estimates because the
stable Fly image lacked SciPy; this v2 changes only that infrastructure
dependency.

The reviewed dependency-free generator
`sobol_runtime_feasibility_generator_20260710.py` implements the fixed
256-dimensional, `m=12` LMS-scrambled Sobol construction and was validated
against SciPy 1.16.2: the generated Sobol uniforms have maximum absolute
error `0.0` for the fixed construction. It is bytewise science-equivalent at
the Sobol-uniform level for this gate, while requiring only NumPy and the
standard library at runtime.

For each of the 100 truth-bank MLPs and three independent replications, the
candidate generates exactly 4096 positive sphere rows in 256 dimensions using
the reviewed generator's `sobol_normal_rows(derived_rep_seed)`, which performs
the fixed Sobol uniforms, `(2^-53, 1-2^-53)` clipping, inverse-normal
transform, row normalization, and `sqrt(256)` scaling. It appends exact
antipodes by evaluating the positive and negative ReLU halves after the first
fp32 L3 Strassen matmul, giving 8192 rows.

The candidate is `sobol_sphere_recolor`. The control is
`iid_sphere_recolor`, made from 4096 independent normalized Gaussian sphere
rows with the same `sqrt(256)` scaling and exact antipodes. Both sphere
methods use the exact same global first-layer ReLU mean/covariance recolor as
the current route, the fp32 first-successor centered application of fixed
strength `1.5`, and fp32 L3 Strassen propagation. No radial factor is applied.

The paired current estimate is the unchanged `hadamard_st3_b16` research
route: 16 genuinely independent positive Hadamard half-bases, exact first-layer
recolor, fp32 first-successor variance matching, and L3 propagation. The
interleaved `_block_rows` helper is not used. All weight checksums are
verified. Truth is read only after all methods and replications are fixed.

The derived seeds are unchanged from v1:
`(mlp_seed ^ stream ^ (rep * 0x9E3779B9)) % 2**32`, with streams
`0x50B01_0710` for Sobol, `0x1D5A_0710` for iid sphere, and `0xC0A1_0710`
for current Hadamard. No scramble, transform, or N sweep is permitted.

PASS requires all of:

- Sobol mean MSE `<= 1.6e-6`;
- current/Sobol global mean-MSE ratio `>= 1.35`;
- per-MLP current/Sobol median ratio `>= 1.20`;
- per-MLP current/Sobol q10 ratio `>= 0.90`;
- per-MLP current/Sobol minimum ratio `>= 0.70`;
- Sobol three-rep mean squared-bias proxy `<= 1e-6`;
- 100/100 successful shards and valid weight checksums.

Frozen launch command:

```text
make fly-payload FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/sobol_sphere_v2_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py paired_fly_logs/fingerprint_theory/sobol_sphere_v2_payload_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/sobol_sphere_v2_gate_20260710_fly.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```

If and only if packaging fails before payload science begins, one packaging-only
retry is permitted with the same frozen command and artifact names. No runtime
retry, parameter change, or partial-result substitution is permitted.
