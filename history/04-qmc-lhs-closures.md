# 2026-07-10 Gaussian QMC/LHS closures

- **Unnormalized Gaussian Sobol v1 FAIL.** The `sobol_gaussian_v1` gate returned
  `100/100` checksum-valid rows with current/IID-Gaussian/Sobol-Gaussian mean
  MSE `2.546583e-6` / `3.768793e-6` / `3.558828e-6`. The current/Sobol ratio
  computed from global mean MSEs was `0.71557`; this is distinct from the
  per-MLP ratio mean `1.0269`, whose median/q10/min were `0.8156` / `0.3697` /
  `0.0941`. The candidate retained the standard Gaussian radial law, closing
  the normalization loophole left by Sobol-sphere. Verdict: **FAIL**; no
  estimator change. Artifacts: `sobol_gaussian_v1_prereg_20260710.md`,
  `sobol_gaussian_v1_payload_20260710.py`,
  `sobol_gaussian_v1_aggregate_20260710.py`,
  `sobol_gaussian_v1_manifest_20260710.json`,
  `sobol_gaussian_v1_stagea_fly_20260710.jsonl`,
  `sobol_gaussian_v1_stagea_results_20260710.json`, and
  `sobol_gaussian_v1_stagea_report_20260710.md`.

- **Exact antithetic Gaussian LHS v1 FAIL.** The gate returned `100/100`
  checksum-valid rows using exact strata and antipodes. Current/LHS/IID mean
  MSE was `2.5465865671e-6` / `4.0034978373e-6` / `3.7688097480e-6`; global
  current/LHS ratio was `0.6360904`, with per-MLP median/q10/min
  `0.800159` / `0.246370` / `0.105682`. Verdict: **FAIL**; no estimator
  change. Artifacts: `antithetic_lhs_v1_prereg_20260710.md`,
  `antithetic_lhs_v1_payload_20260710.py`,
  `antithetic_lhs_v1_aggregate_20260710.py`,
  `antithetic_lhs_v1_manifest_20260710.json`,
  `antithetic_lhs_v1_stagea_fly_20260710.jsonl`, and
  `antithetic_lhs_v1_stagea_report_20260710.md`.

- **Hadamard-oriented LHS v1 FAIL.** The gate returned `100/100` with exact
  strata and antipodes. Current/independent-LHS/Hadamard-LHS mean MSE was
  `2.546587103e-6` / `3.836092491e-6` / `3.979278485e-6`. The global
  current/Hadamard-LHS ratio from mean MSEs was `0.63996`, distinct from the
  per-MLP ratio mean `1.0637`; its median/q10/min were `0.7954` / `0.2854` /
  `0.0902`. Hadamard sign orientation reduced off-diagonal RMS from
  `0.015596` to `0.012033` but worsened MSE. Verdict: **FAIL**; no estimator
  change. Artifacts: `hadamard_lhs_v1_prereg_20260710.md`,
  `hadamard_lhs_v1_payload_20260710.py`,
  `hadamard_lhs_v1_aggregate_20260710.py`,
  `hadamard_lhs_v1_manifest_20260710.json`,
  `hadamard_lhs_v1_gate_20260710_fly.jsonl`,
  `hadamard_lhs_v1_stagea_results_20260710.json`, and
  `hadamard_lhs_v1_stagea_report_20260710.md`.

Combined conclusion is narrow: input-axis Gaussian QMC/LHS, including exact
antipodal strata, retained radial law, and Hadamard sign orientation, is closed.
This does not pre-judge first-layer-coordinate rank transport.

- **First-layer coordinatewise Gaussian anamorphosis v1 FAIL.** The
  `layer1_rank_gauss_v1` Stage-A gate returned `100/100` rows with zero
  failures/pending and valid checksums. Current/rank-Gaussian mean MSE was
  `2.521168251474e-6` / `2.555385657835e-6`; global current/candidate ratio
  was `0.9928583095`, with per-MLP median/q10/min
  `0.9865111337` / `0.9202015029` / `0.8317887341`. Exact antipodes,
  magnitude-rank roundtrip, and full sorted-target transport all had zero
  error; maximum magnitude-tie fraction was `1.230540293e-4`. Sparse raw
  zeros (max `1`, mean `0.05`) tripped the conservative integrity gate but
  do not affect the decisive performance FAIL. First-layer coordinatewise
  Gaussian anamorphosis with exact post-ReLU recolor is closed; no estimator
  change. Artifacts: `layer1_rank_gauss_v1_stagea_fly_20260710.jsonl`,
  `layer1_rank_gauss_v1_stagea_results_20260710.json`, and
  `layer1_rank_gauss_v1_stagea_report_20260710.md`.

- **Folded finite-set ZCA for Gaussian QMC/LHS v1 FAIL.** The frozen
  420-second Stage-A window returned `90/100` rows with zero explicit
  failures, ten pending indices (`4,10,21,37,51,74,81,84,96,98`), zero
  duplicates, and all 90 returned rows individually checksum-valid. Mean MSE
  for current/LHS-base/LHS-ZCA/Sobol-base/Sobol-ZCA was
  `2.651589827247e-6` / `4.219461683318e-6` / `4.270643995772e-6` /
  `3.741434150188e-6` / `3.624390535576e-6`. Global current/ZCA ratios were
  `0.6208876` LHS and `0.7315961` Sobol; their per-MLP median/q10/min ratios
  were `0.7790406/0.2697721/0.1313155` and
  `0.8207934/0.3595786/0.1024458`. LHS-base/ZCA and Sobol-base/ZCA were only
  `0.9880153` and `1.0322933`, both below the frozen `1.15` improvement bar.
  Whitening itself was numerically successful: post-ZCA covariance relative
  Frobenius error was at most `7.3753e-8` LHS and `7.4167e-8` Sobol,
  antipodes were exact, and preactivation covariance error was about `1e-6`.
  The incomplete tail fails common integrity, but returned-row performance
  independently fails decisively: exact finite-set input covariance whitening
  is not the missing Gaussian QMC/LHS defect. No estimator change. Artifacts:
  `folded_zca_qmc_v1_stagea_fly_20260710.jsonl`,
  `folded_zca_qmc_v1_stagea_results_20260710.json`, and
  `folded_zca_qmc_v1_stagea_report_20260710.md`.

- **K8 antipodal odd-state Rao--Blackwell closure v1 FAIL.** The frozen
  three-replication Stage-A gate returned `100/100` checksum- and schema-valid
  rows with zero failures, pending shards, or duplicates. Current/candidate
  M1 was `2.752383145e-6` / `2.577790606e-4`, a global current/candidate
  ratio of `0.0106773`; candidate squared bias was `2.562949223e-4`, and its
  projected b27 MSE remained `2.571744117e-4`. All numerical integrity checks
  passed, but the initial rank-one odd-covariance factor residual had mean
  `0.78821` and the propagated closure residual mean was `0.15722`. The
  approximation therefore removed variance at the cost of overwhelming bias:
  increasing the block count cannot rescue this closure. No Stage B or
  estimator change followed. Artifacts:
  `odd_rb_k8_v1_stagea_fly_20260710.jsonl`,
  `odd_rb_k8_v1_stagea_results_20260710.json`, and
  `odd_rb_k8_v1_stagea_report_20260710.md`.

- **Signed low-rank antipodal odd-state transport v1 FAIL.** The frozen
  three-replication Stage-A gate returned `100/100` checksum- and schema-valid
  rows with zero failures, pending shards, or duplicates. Current/candidate M1
  was `2.850595174e-6` / `1.023208499e-4`, a global current/candidate ratio of
  `0.0278594`; candidate squared bias was `8.656223730e-5`, and its projected
  b27 MSE remained `9.590067441e-5`. All pair, subspace, Gram, and
  coordinate-energy-restoration checks passed. The scheduled rank
  `64/32/16/8` carrier captured mean odd-state energy `0.85906`, but restoring
  coordinatewise energy required scales as large as `64.28` and produced
  overwhelming bias. The diagnostic-only exact/candidate block correlation
  averaged `0.52532`, while correction/exact block variance averaged `0.93051`,
  so an unbiased two-level correction is also unattractive. Verdict: **FAIL**;
  no estimator implementation, rerun, or estimator change. Artifacts:
  `odd_lr_transport_v1_20260710_stagea_fly.jsonl`,
  `odd_lr_transport_v1_20260710_stagea_results.json`, and
  `odd_lr_transport_v1_20260710_stagea_report.md`.

- **Final-weight collapse cross-output oracle ceiling CLOSED.** A read-only
  scout reused the completed terminal-mixture Fly vectors and verified all
  `100/100` rebuilt MLP checksums and `300/300` stored baseline MSEs. It gave
  polynomial-ridge, RBF-kernel, and 16-nearest-neighbor smoothers the illegal
  optimistic layer-30 truth mean as a feature anchor, then evaluated a
  truth-oracle global blend back toward the direct estimator. Baseline M1 was
  `2.677520522e-6`; the best oracle blend was only the 16-NN arm at
  `2.650269629e-6`, a `1.01028x` gain, with three-rep-mean MSE
  `1.103269425e-6`. Even the held-out truth-response approximation ceilings
  were orders of magnitude above `1.6e-6`. Verdict: **CLOSE**; final means are
  not sufficiently smooth in these final-weight/penultimate-mean features for
  cross-output denoising, even under information unavailable to a legal
  estimator. No estimator change or new Fly run. Artifacts:
  `final_weight_collapse_scout_20260710.py`,
  `final_weight_collapse_scout_20260710.json`, and
  `final_weight_collapse_scout_20260710.md`.

- **2026-07-20 one-layer analytic-prefix/Hadamard-suffix `hyb1` is not a
  faithful exact-post-ReLU isolation and loses on the current scorer path.**
  Code inspection confirmed that the existing mode
  `hadamard_st3_b16_hyb1` already implements the proposed preactivation
  restart, so no duplicate mode or estimator rewrite was made. It forms the
  exact Gaussian first-preactivation covariance
  `Sigma1 = W0.T @ W0`, adds a ridge of `1e-6` times its mean marginal
  variance for Cholesky stability, and maps randomized Hadamard cubature rows
  through that Cholesky factor before adding their exact antipodes. These rows
  are cubature particles, not Gaussian draws. It then applies ReLU and
  propagates through `W1...` with L3 Strassen plus the current `1.5x`
  first-successor marginal-variance correction. The mode uses only the passed
  MLP, `mlp.seed`, and analytic Gaussian identities; the mode environment
  token is a general diagnostic control, not an evaluation-instance branch.

  The existing implementation does **not** report or enforce the exact
  post-ReLU moments at layer 1. `_zero_mean_relu_mean_cov` computes the exact
  Gaussian mean `sqrt(diag(Sigma1)/(2*pi))` and arc-cosine covariance, but for
  `hyb1` the reporting path discards that analytic row and appends the finite
  Hadamard ensemble mean instead. The propagated layer-1 particles are also
  not recolored, so their mean/covariance are the cubature moments after ReLU,
  not the exact Gaussian ReLU mean/covariance. Only the zero preactivation
  mean and the jittered preactivation covariance are exact for each complete
  Hadamard block. Restarting in Cholesky coordinates also discards the
  input-to-preactivation orthogonal orientation carried by `W0`; that
  orientation is irrelevant for an actual Gaussian but changes Hadamard
  fourth- and higher-order alias structure, which then survives the ReLU and
  deep suffix.

  The requested normal-window command
  `make fly-mode MODE=hadamard_st3_b16_hyb1` used fixed Fly dataset
  fingerprint `50c6efdca4059f0e` but initially produced no aggregate: `0`
  scored rows, `5` Fly Machine failures with return code `408`, and `95`
  pending when the 45-second collection window closed. That first cutoff was
  incorrectly interpreted as a scorer-path wall-time loss. A correction run
  extended only the result-collection window to 90 seconds while leaving the
  worker/scorer wall limit unchanged at 60 seconds. It returned `80` scored
  rows at `8.333e-5` final-layer MSE, `8.422e-6` adjusted score, `2.396e10`
  raw FLOPs, and `2.688e10` effective compute, with no estimator/scorer
  failures among returned rows; there were `4` separate Fly Machine `408`
  failures and `16` pending rows. The same fixed-set canonical `make fly`
  control returned `80` scored rows at `2.667e-6` final-layer MSE,
  `2.829e-7` adjusted score, `2.535e10` raw FLOPs, and `2.799e10` effective
  compute, with no estimator/scorer failures among returned rows; there were
  `2` separate Fly Machine `408` failures and `18` pending rows. Thus the
  hybrid's extra arithmetic is not the issue: it uses about `5.5%` fewer raw
  FLOPs and `4.0%` less effective compute, but its final MSE is about `31.2x`
  worse and adjusted score about `29.8x` worse. This is a decisive statistical
  mechanism loss, so no paired/full-100 or constant-tuning run followed and
  default behavior remains unchanged.

  A separate `hyb1_recolor` would be scientifically justified only as a clean
  mechanism ablation: report the exact analytic layer-1 mean and recolor the
  direct-preactivation cubature particles to the exact Gaussian post-ReLU
  covariance, thereby isolating Cholesky-coordinate higher-moment structure
  from the already-promoted input-space route at matched first two ReLU
  moments. It is not presently justified as an immediate promotion candidate
  because the unrecolored route is roughly `31x` worse in final MSE. Any
  future recolor gate should be preregistered strictly as an isolation
  experiment rather than treated as tuning.

  Follow-up `hyb1_recolor` isolation, requested 2026-07-20: added the
  mode-gated `hadamard_st3_b16_hyb1_recolor` token without changing unforced
  behavior. It keeps the same Cholesky-coordinate preactivation cubature, then
  applies the current default's linear first-layer recolor so the propagated
  ensemble matches the exact Gaussian post-ReLU mean and covariance, reports
  that exact analytic mean for layer 1, and uses fp32 weights/particles for the
  L3 suffix plus the unchanged `1.5x` first-successor variance correction.
  The parser restricts `recolor` to the plain one-layer hybrid so it cannot be
  accidentally composed with the approximate deeper/skew/joint-k3 prefixes.
  `python -m py_compile estimator.py` passed.

  Fly used the same fixed dataset fingerprint and the 90-second collection
  window with the worker/scorer wall still at 60 seconds. The recolored mode
  returned `80` scored rows at `3.480e-6` final-layer MSE, `3.561e-7`
  adjusted score, `2.556e10` raw FLOPs, and `2.788e10` effective compute,
  with no estimator/scorer failures among returned rows; there were `3`
  separate Fly Machine `408` failures and `17` pending rows. Recoloring
  improves MSE by about `24x` versus unrecolored `hyb1` (`8.333e-5`), proving
  that missing post-ReLU moment matching caused most of the catastrophic loss.
  Against the same-day canonical default (`2.667e-6` MSE, `2.829e-7`
  adjusted, `2.535e10` raw, `2.799e10` effective), however, it remains about
  `30.5%` worse in MSE and `25.9%` worse in adjusted score at essentially
  matched compute. This clears the repository's `~15%` summary-noise loss
  threshold. Verdict: the Cholesky-coordinate restart's altered higher-moment
  Hadamard structure is itself harmful after controlling the exact first two
  post-ReLU moments; keep `hyb1_recolor` diagnostic-only, do not promote, and
  do not spend a paired/full-100 follow-up.
