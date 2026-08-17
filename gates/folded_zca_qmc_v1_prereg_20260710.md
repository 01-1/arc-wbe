# Folded ZCA Gaussian-QMC/LHS v1 (pre-registration)

Research-only Stage-A preparation. This namespace is collision-free and does
not modify `estimator.py` or `ESTIMATOR_HISTORY.md`. Each of 100
checksum-verified width-256/depth-32 MLPs receives one fixed replication and
five paired 8192-row estimates: corrected 16-base current, LHS base, LHS ZCA,
Sobol-Gaussian base, and Sobol-Gaussian ZCA.

Each LHS and Sobol route has 4096 positive representatives followed by their
exact negatives. The audited dependency-free LHS midpoint-stratum/permutation
generator and reviewed LMS-scrambled Sobol generator are used without row
normalization, radius rescaling, rotation, or radial multiplier. The same
positive matrix `U` is used for each base/ZCA pair. For a pair, compute
`S=U.T@U/4096` in flopscope, eigendecompose symmetric float64 `S`, floor
eigenvalues at `1e-8`, and form symmetric `A=S^(-1/2)` in flopscope. Cast `A`
to fp32 and fold it as `W0_eff=A@W0`; never explicitly form `U@A` for the
candidate preactivation. The ZCA preactivation is `U@W0_eff` in fp32 L3
Strassen. The original `W0.T@W0` remains the target for exact global ReLU
mean/covariance recoloring.

All route propagation uses the actual-estimator fp64 first-successor variance
statistics with fp32 centered application/scale/writeback at strength 1.5 and
fp32 L3 propagation. Candidate vectors and diagnostics are fixed before truth
is read. All candidate math is flopscope; only generator constants/input
loading, scalar diagnostic export, final vectors, and post-truth MSE export
cross to NumPy.

Frozen diagnostics for each LHS/Sobol base/ZCA pair include input antipode
error; pre/post-ZCA covariance relative Frobenius error, diagonal maximum
error, off-diagonal RMS/maximum; floored eigenvalue min/median/max and
condition; pre-whitening radius mean/std and post-ZCA radius RMS derived from
the covariance; and base/ZCA preactivation covariance relative Frobenius error
against `W0.T@W0`.

PASS for either whitened candidate requires 100/100 checksums/no failures,
candidate MSE `<=1.8e-6`, current/candidate global ratio `>=1.25`, per-MLP
median/q10/min ratios `>=1.10/0.85/0.65`, post-ZCA input covariance relative
Frobenius maximum `<=1e-4`, transported antipode maximum `<=1e-5`, and ZCA
global MSE improvement over its same-row base `>=1.15`. Base controls are
diagnostic only. No sweep, blend, Stage B, or estimator mode is prepared.

Exact proposed command (not authorized yet):

```sh
FLY_MLPS=100 FLY_MIN_RESULTS=100 \
FLY_MAX_RESULT_SECONDS=420 FLY_PAYLOAD_MAX_RESULT_SECONDS=420 \
FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/folded_zca_qmc_v1_manifest_20260710.json \
FLY_PAYLOAD_FILES="paired_fly_logs/fingerprint_theory/folded_zca_qmc_v1_payload_20260710.py paired_fly_logs/fingerprint_theory/folded_zca_qmc_v1_aggregate_20260710.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py estimator.py local_engine.py analysis/truth_bank/truth_bank.npz" \
FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/folded_zca_qmc_v1_stagea_fly_20260710.jsonl \
make fly-payload
```

No Fly launch has occurred; coordinator approval is required.
