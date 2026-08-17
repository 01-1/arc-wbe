# Unnormalized Gaussian Sobol RQMC Stage-A gate (pre-registration)

Status: PRE-REGISTERED; preparation only. No Fly launch is authorized by this
artifact. This is a new `sobol_gaussian_v1_*_20260710` namespace and does not
overwrite any prior Sobol/Haar/Stein artifact.

For each of the 100 checksum-verified truth-bank MLPs, run exactly one fixed
replication with three paired 8192-row estimates. Each route has exactly 4096
positive rows followed by exact antipodal negatives through the existing
first-layer shortcut:

1. corrected current `hadamard_st3_b16`: 16 genuinely independent positive
   Hadamard bases;
2. iid standard-normal positive rows (control);
3. Owen/LMS-scrambled Sobol uniforms in 256 dimensions, transformed
   coordinatewise by the reviewed dependency-free `ndtri_dependency_free`
   implementation (candidate).

The candidate uses `sobol_uniform(seed)` with the fixed reviewed bounds
`[2^-53, 1-2^-53]`, then Acklam `ndtri_dependency_free`, cast to fp32. There is
no row normalization, radius rescaling, rotation/LT transform, input recolor,
or radial multiplier. Thus it retains the standard-normal radial law. The iid
control uses the same fp32 propagation route but independent Gaussian rows.

All three routes use the exact global first-layer ReLU mean/covariance recolor,
fp32 centered application of the fixed 1.5 first-successor variance match, and
fp32 L3 Strassen propagation. Weight checksums are verified before estimation.
All estimates are fixed before truth is read. Label-free diagnostics report
positive-row radius mean/std/q10/q90, covariance relative Frobenius error and
maximum off-diagonal, and maximum coordinate mean magnitude for Sobol and iid.
These covariance and mean diagnostics are on positive representatives; a
separate `antipode_max_abs` diagnostic checks the constructed positive/negative
rows and must be exactly zero.

Frozen constants: `d=256`, `depth=32`, `positive_rows=4096`, `antithetic_rows=8192`,
`blocks=16`, `reps=1`, streams `0x50B01_0710`, `0x1D5A_0710`, and
`0xC0A1_0710`, with derived seed
`(mlp_seed ^ stream ^ rep*0x9E3779B9) mod 2^32`.

PASS requires 100/100 successful checksum-verified shards, Sobol-Gaussian mean
MSE `<=1.8e-6`, current/Sobol global ratio `>=1.25`, per-MLP median ratio
`>=1.10`, q10 `>=0.85`, and minimum `>=0.65`. iid/Sobol ratios and diagnostics
are reported but are not gates. Only if this Stage A passes may a coordinator
authorize a separate frozen three-rep confirmation.

Proposed, not-yet-authorized command:

```text
make fly-payload FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/sobol_gaussian_v1_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py paired_fly_logs/fingerprint_theory/sobol_gaussian_v1_payload_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/sobol_gaussian_v1_stagea_fly_20260710.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```
