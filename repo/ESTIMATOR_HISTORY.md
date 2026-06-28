# Estimator History

This is the decision-useful history for the repository-root
[`estimator.py`](estimator.py). It is intentionally compact: keep the current
route, benchmark checkpoints that changed direction, and rejected ideas that
are likely to be retried.

## Current Estimator

The current grader shape is width 256, depth 32, with a `2.72e11` FLOP/MLP
budget and a score-efficient target just under `2.72e10` effective FLOPs.

For depth-32 contest MLPs, unforced `predict()` uses randomized antithetic
Walsh-Hadamard sign cubature with 13 blocks. After the first linear/ReLU layer,
the estimator linearly recolors the first hidden activation ensemble so its
mean and covariance match the exact zero-mean Gaussian ReLU moments for
`W0.T @ W0`. It then propagates the recolored ensemble through the remaining
layers, applying a `1.5x` variance-scale update to only the first subsequent
ReLU ensemble using its Gaussian marginal variance target while preserving its
sample mean. This route uses only the passed MLP object and label-free moment
identities.

For shallower MLPs, the default remains the optimized factorized K=3 cumulant
route with `r=1` degree-4 harmonic tracking, structured third-cumulant factor
groups, and a diagonal-only final-layer ReLU mean shortcut. The K=3 route is
still the relevant fallback and comparison baseline for shallow or diagnostic
runs.

The submission estimator now keeps only the live default route and direct
comparison modes: `r1` for the shallow K=3 path, `hadamard_first_cov` for the
old deep Hadamard route, and `hadamard_var1`/`hadamard_var2` for the first-layer
variance-matching variants, including `hadamard_var1_s<N>` strength sweeps.
Older experimental modes for compressed K=3, K=1/K=2 diagnostics, low-rank
covariance, axis cubature, and sample blends were removed from `estimator.py`
after losing or becoming irrelevant to the current scorer frontier.

## Winning Checkpoints

- **K=2 covariance plus sampling floor.** The early analytical baseline tracked
  full covariance with exact marginal ReLU moments and blended in antithetic
  sampling when it fit below the old 10% score floor. At width-256/depth-8 it
  used about `6.69e9` FLOPs and landed around `2.12e-05` final-layer MSE. It
  was useful as a floor, but left too much accuracy on the table.
- **Factorized K=3 `r1`.** Porting the upstream factorized K=3 path, caching
  repeated diagonal slices, and routing through the `r=1` harmonic subset beat
  K=2 despite higher compute. The final-layer shortcut later reduced the
  width-256/depth-8 route to about `1.63e10` FLOPs with unchanged predictions.
- **Structured grouped `r1`.** Keeping third-cumulant terms as structured
  factor groups reduced analytical FLOPs from about `1.63e10` to `1.25e10` on
  width-256/depth-8 while preserving predictions. Local cached-mini scoring
  with residual-time charging kept exact grouped `r1` ahead of the compression
  variants that were safe to run.
- **Depth-32 retargeting.** Exact grouped `r1` fit analytically at
  width-256/depth-32 but left too little residual-time headroom. A compressed
  K=3 depth route was a temporary bridge, but randomized Hadamard cubature was
  far better for deep networks.
- **First-covariance Hadamard.** Plain Hadamard blocks outperformed compressed
  K=3 on depth-32. Recoloring the first hidden ensemble to the exact
  first-layer Gaussian ReLU mean/covariance improved the route further without
  using labels. Fly EWR 80-result sweeps with corrected residual compute found
  the best adjusted-score frontier at 13 blocks:

  | Blocks | Final-layer MSE | Adjusted score | Effective compute |
  |---:|---:|---:|---:|
  | 11 | `4.473e-6` | `4.473e-7` | `2.583e10` |
  | 12 | `3.664e-6` | `3.786e-7` | `2.811e10` |
  | 13 | `3.068e-6` | `3.430e-7` | `3.041e10` |
  | 14 | `3.458e-6` | `4.150e-7` | `3.267e10` |
  | 16 | `3.230e-6` | `4.423e-7` | `3.724e10` |

- **First-successor variance-scaled Hadamard.** Preserving the first-covariance
  ensemble means while rescaling only the first subsequent post-ReLU marginal
  variance toward a Gaussian ReLU moment target improved returned-set Fly
  comparisons without the instability of broader marginal correction. Exact
  variance matching produced the strongest noisy sample at `3.205e-7`
  adjusted / `2.831e-6` final-layer MSE / `3.082e10` effective compute over 79
  returned MLPs, but retries varied. A `1.5x` scale update gave the best clean
  mode sweep at `3.151e-7` adjusted / `2.783e-6` MSE / `3.080e10` effective
  compute, and the promoted default `make fly` proof scored `3.353e-7`
  adjusted / `2.962e-6` MSE / `3.082e10` effective compute with 80 returned and
  no failures. Rechecking 12 and 14 blocks after the strength update lost:
  12 blocks scored `3.390e-7` adjusted / `3.223e-6` MSE / `2.859e10`
  effective compute, and 14 blocks scored `3.615e-7` adjusted / `2.963e-6`
  MSE / `3.318e10` effective compute.
  Shrinking the per-coordinate variance scale halfway toward one global scalar
  produced one promising first-80 run (`3.069e-7` adjusted / `2.710e-6` MSE),
  but a replicate bounced to `3.634e-7`, and the full-100 check was neutral at
  `3.386e-7` adjusted / `2.989e-6` MSE / `3.085e10` effective compute with no
  failures. Do not promote without a stronger paired win. A positive log-space
  power correction, using `sqrt(target/sample) ** 1.5` instead of the linear
  `1.5x` scale update, also lost at `3.559e-7` adjusted / `3.135e-6` MSE /
  `3.085e10` effective compute over 76 returned MLPs, with four worker
  download failures.

## Rejected Or Guarded Ideas

- **Public-label calibration.** Fitting final ReLU mean coefficients, expanded
  shortcut features, or residual overlays against cached public-mini labels is
  not a legitimate estimator improvement. Those experiments were removed and
  should not be revived.
- **Structured compression.** Whole-group and boundary-group compression looked
  like small local wins, but one structured-cap default was not grader-safe
  because group ordering extracted host Python floats from flopscope remote
  scalars. Compression is still worth revisiting only if ranking stays inside
  flopscope-safe array operations and wins under the residual multiplier.
- **Dense top-k rank caps.** Old flops-only adjusted-score proxies ignored
  residual wall time and overstated the value of dense scheduled compression.
  Current grouped exact `r1` remains the true shallow baseline unless a new
  compression route wins under `make mini`/Fly-style residual scoring.
- **Augmented K=3 suffixes.** Corrected `r1_slices_k211` augmentation improves
  raw MSE, especially in late-layer suffixes, but residual wall time and
  effective compute have beaten the accuracy gain so far. Retry only with a
  concrete residual-time reduction.
- **Alternative sample families.** Ordinary Gaussian sampling, Rademacher
  sampling, axis cubature, low-rank covariance ensembles, Halton/bridge
  samples, `H D H` rotated signs, spherical radial scaling, and fourth-moment
  axis mixes all lost to fixed Hadamard in smokes. Deterministic per-block
  Hadamard column permutations after the `1.5x` first-successor variance update
  also lost at `3.634e-7` adjusted / `3.201e-6` MSE / `3.089e10` effective
  compute, and balancing the random diagonal signs across 13 blocks lost at
  `3.629e-7` adjusted / `3.202e-6` MSE / `3.085e10` effective compute.
  Trimming the highest and lowest final-layer Hadamard block mean per coordinate
  was neutral and stayed inside Fly noise at `3.343e-7` adjusted / `2.952e-6`
  MSE / `3.083e10` effective compute with no failures, so it was not promoted.
- **First-layer moment variants.** Diagonal-only mean/variance matching,
  marginal skew correction, clipping recolored activations back to nonnegative
  support, half-strength first-cov blending, blockwise shrinkage, symmetric
  Gaussian optimal-transport covariance recoloring, blockwise independent
  first-covariance recoloring, covariance-ridge regularization, and final-only
  Gaussian marginal correction did not beat full first-covariance recoloring.
  The OT covariance map raised Fly EWR adjusted score to `3.956e-7` with
  final-layer MSE `3.471e-6` and effective compute `3.103e10`. Recoloring each
  Hadamard block independently scored `4.446e-7` adjusted / `3.864e-6` MSE /
  `3.130e10` effective compute for one-block groups, while a coarse 7+6 block
  split still lost at `3.806e-7` adjusted / `3.400e-6` MSE / `3.046e10`
  effective compute. Increasing the Cholesky covariance ridge from `1e-6` to
  `1e-4` and `1e-3` of the target average variance also lost at `3.497e-7` and
  `3.714e-7` adjusted, respectively.
  Retrying final-only Gaussian marginal mean correction after the `1.5x`
  first-successor variance update as a 50% output blend also lost:
  `3.700e-7` adjusted / `3.262e-6` MSE / `3.085e10` effective compute over
  79 returned MLPs, with one failed worker. Final-layer sample-cumulant
  Edgeworth blends were also not robust: 50% scored `3.504e-7` adjusted /
  `3.082e-6` MSE, 20% initially scored `3.263e-7` adjusted / `2.870e-6` MSE,
  10% bounced to `3.409e-7` adjusted / `2.999e-6` MSE, and a full-100 check
  of the 20% blend settled at `3.458e-7` adjusted / `3.034e-6` MSE /
  `3.102e10` effective compute over 99 returned MLPs with one worker failure.
- **Full per-layer Gaussian marginal correction.** Correcting every layer's
  marginals destroyed useful joint geometry and produced much worse scores.
  Mean-and-variance correction on only the first post-recolor layer also lost
  at `3.767e-7` adjusted, and a 25% first-successor mean pull on top of the
  current `1.5x` variance update stayed neutral at `3.352e-7` adjusted /
  `2.958e-6` MSE / `3.083e10` effective compute over 79 returned MLPs, with
  one worker failure. Variance-only correction of the second successor
  layer alone lost at `3.696e-7`, adding the second successor to the first was
  positive but weaker/noisier (`3.660e-7` clean default proof, `3.676e-7`
  full-100 returned-set comparison), and a decayed `1.5x`/`0.5x` two-layer
  schedule remained behind at `3.510e-7` adjusted / `3.095e-6` MSE /
  `3.085e10` effective compute with no failures. That decayed-schedule delta
  is within the 15% relative Fly-noise band, so it was not promoted but is not
  treated as a decisive mechanism loss. A third variance-only layer lost at
  `4.643e-7`. For the first-successor strength sweep, `0.75x` lost at
  `3.646e-7`, `1.25x` was not enough at `3.400e-7`, and `1.75x` fell back to
  `3.496e-7`. A full first-successor covariance recolor using zero-mean ReLU
  covariance correlations with nonzero marginal variances also lost badly:
  `5.749e-7` adjusted / `4.666e-6` MSE / `3.352e10` effective compute. A
  gain-covariance first-successor recolor with exact marginal variances also
  lost by a large margin: `5.605e-7` adjusted / `4.544e-6` MSE / `3.351e10`
  effective compute over 79 returned MLPs with one clipped worker returncode.
- **Zero-mean arc-cosine and conditional-quadrature K=2 covariance updates.**
  These replaced the simple gain covariance approximation, but nonzero later
  pre-activation means and numerical instability made them worse than the
  original K=2 route.

## Benchmarking Notes

Use current scorer-path comparisons, not stale flops-only proxies. For
estimator changes, follow [`AGENTS.md`](AGENTS.md): compile `estimator.py` and
use the Fly fast runner by default unless the owner asks for a different proof.
For docs-only changes, a link/search check and Markdown sanity are sufficient.
