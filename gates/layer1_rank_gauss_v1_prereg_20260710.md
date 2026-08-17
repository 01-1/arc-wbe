# First-layer Gaussian rank transport v1 (pre-registration)

Research-only, label-free until post-hoc truth scoring. This is a new
collision-free gate and does not alter estimator code or prior artifacts.

Each of the 100 checksum-verified width-256, depth-32 truth-bank MLPs gets one
replication with 4096 positive rows from exactly 16 independent randomized
positive Hadamard half-bases. The raw first-layer preactivation is computed
once in fp32 and paired as `pre=[pre_half;-pre_half]`, for 8192 rows. Both arms
share this tensor. The current arm applies ReLU, the exact global first-layer
ReLU mean/covariance recolor, fp32 centered strength-1.5 first-successor
variance matching, and fp32 L3 propagation.

The candidate stable-sorts the 4096 shared positive-half preactivation
magnitudes in each of the 256 output coordinates. The fixed positive table
`q_k=Phi^-1((k+0.5)/8192)` for `k=4096..8191`, scaled by
`sigma_j=sqrt((W0.T@W0)[j,j])`, is gathered back by inverse magnitude rank,
the original fp32 signs are applied, and the exact negative half is appended.
This is identical to full antipodal ranking when magnitudes are distinct, while
remaining antipode-safe under finite-fp32 magnitude ties. Zero magnitudes are
assigned the deterministic positive sign. The full transported sort is audited
against the complete symmetric fixed quantile table. The table is generated
from the reviewed dependency-free `ndtri_dependency_free` implementation. The
candidate then receives the same ReLU recolor, fp32 centered match, and L3
suffix. No row normalization, rotation, radial factor, blend, copula change,
or truth-dependent choice is permitted.

Diagnostics are frozen before truth access: raw and transported antipode
errors, raw magnitude-tie count/fraction, zero-value count, magnitude-rank
round-trip and full sorted-transport audits, transported per-coordinate mean and variance relative errors, each
arm's pre-recolor ReLU mean and diagonal second-moment errors against exact
Gaussian targets, sample ReLU covariance relative Frobenius error, and affine
recolor Frobenius distance from identity.

Stage-A PASS requires 100/100 valid checksums, candidate mean MSE `<=1.8e-6`,
global current/candidate `>=1.25`, per-MLP ratio median/q10/min
`>=1.10/.85/.65`, transported antipode max `<=1e-5`, exact full sorted
transport, exact magnitude-rank roundtrip, zero raw positive-half values, and
raw magnitude-tie fraction `<=1e-3`. There is no Stage B or estimator mode in
this gate.

The exact proposed command is:

```sh
FLY_MLPS=100 FLY_MIN_RESULTS=100 \
FLY_MAX_RESULT_SECONDS=420 \
FLY_PAYLOAD_MAX_RESULT_SECONDS=420 \
FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/layer1_rank_gauss_v1_manifest_20260710.json \
FLY_PAYLOAD_FILES="paired_fly_logs/fingerprint_theory/layer1_rank_gauss_v1_payload_20260710.py paired_fly_logs/fingerprint_theory/layer1_rank_gauss_v1_aggregate_20260710.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py estimator.py local_engine.py analysis/truth_bank/truth_bank.npz" \
FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/layer1_rank_gauss_v1_stagea_fly_20260710.jsonl \
make fly-payload
```

No launch has occurred; coordinator approval is required before running it.
