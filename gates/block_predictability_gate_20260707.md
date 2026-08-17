# Block Predictability Gate, 2026-07-07

Purpose: test whether label-free, MLP-derived Hadamard block observables can
predict blockwise final error or useful block covariance strongly enough to
justify unequal block weighting/allocation or paired-block control.

Rules boundary: the submitted estimator may only use the passed MLP object and
MLP-independent randomness. This gate may use the Fly truth bank only as
research labels for measuring whether a legal predictor exists. No local
estimator scoring is performed, and no estimator behavior is changed unless the
gate passes.

Pre-registered design:

- Run one Fly payload shard per truth-bank MLP.
- Rebuild the shard MLP from the truth-bank seed and generate eight independent
  legal 16-block antithetic Hadamard ensembles per MLP, using only MLP weights
  plus independent RNG seeds.
- For every block, propagate the current default first-layer recolor and
  first-successor variance match, then record the final block mean.
- Candidate label-free block features:
  first-layer raw block mean residual, raw covariance trace/diagonal error,
  post-recolor block mean residual, post-recolor covariance trace/diagonal
  error, first-successor variance-match energy, final-layer block radius,
  downstream-weighted final block radius, and final block skew/kurtosis.
- Research labels:
  squared final error per block against the truth-bank final mean, covariance
  with replicate-level final error, and paired/weighted replicate MSE.
- Model class:
  simple ridge/log-linear predictors trained out-of-MLP. Hyperparameters are
  fixed to a small grid before aggregation; selection uses cross-validation
  only inside the training folds.

Pass/fail:

- PASS for weighting/allocation only if cross-validated label-free weights
  imply at least `1.30x` overall variance reduction versus equal block weights,
  with median per-MLP ratio at least `1.20x` and q10 at least `0.95x`.
- PASS for paired-block control only if a fixed label-free pairing/combination
  shows at least `20%` mean final-MSE reduction, median reduction at least
  `10%`, and q10 no worse than `-5%`.
- Otherwise FAIL and treat the shape-preserving sampler path as effectively
  closed pending external code/writeups.

