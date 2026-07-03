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
`W0.T @ W0`. The first layer uses only the positive half of each antithetic
Hadamard block for the matmul, then reconstructs the negative-half ReLU
activations from the negated preactivations. It then propagates the recolored
ensemble through the remaining layers, applying a `1.5x` variance-scale update
to only the first subsequent ReLU ensemble using its Gaussian marginal variance
target while preserving its sample mean. This route uses only the passed MLP
object and label-free moment identities.

For shallower MLPs, the default remains the optimized factorized K=3 cumulant
route with `r=1` degree-4 harmonic tracking, structured third-cumulant factor
groups, and a diagonal-only final-layer ReLU mean shortcut. The K=3 route is
still the relevant fallback and comparison baseline for shallow or diagnostic
runs.

The submission estimator now keeps only the live default route and direct
comparison modes: `r1` for the shallow K=3 path, `hadamard_first_cov` for the
old deep Hadamard route, and `hadamard_var1`/`hadamard_var2` for the first-layer
variance-matching variants, including `hadamard_var1_s<N>` strength sweeps.
`hadamard_chi`, `hadamard_b<N>`, `hadamard_st<L>`, and
`hadamard_st<L>_b<N>` are guarded diagnostics for the same variance route with
chi-radial first-layer scaling, explicit block counts, and Strassen
propagation matmuls, respectively.
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
  Explicitly rechecking the lower block frontier after the `1.5x` strength
  update found that the score floor was not enough to offset MSE: 11 blocks
  reached the `0.1` multiplier floor but scored `3.561e-7` adjusted /
  `3.560e-6` MSE / `2.624e10` effective compute, and 12 blocks scored
  `3.589e-7` adjusted / `3.422e-6` MSE / `2.852e10` effective compute, both
  with no failures.
  Owner AICrowd 50-MLP checks later put `4301cef` ("Use first-layer variance
  match by default") at `3.12e-7` adjusted / `2.77e-6` MSE, while `21128f4`
  ("Variance-match early Hadamard layers") scored `3.44e-7` adjusted /
  `3.07e-6` MSE; treat that as external support for keeping the one-layer
  variance route ahead of broader early-layer correction, while still inside
  the documented Fly noise caveats for small deltas. The later `803d3ad`
  strength-tuned default scored `3.18e-7` adjusted / `2.84e-6` MSE on the same
  owner AICrowd 50-MLP path, supporting the `1.5x` first-successor strength as
  the current default despite noisy Fly replicates.
  Shrinking the per-coordinate variance scale halfway toward one global scalar
  produced one promising first-80 run (`3.069e-7` adjusted / `2.710e-6` MSE),
  but a replicate bounced to `3.634e-7`, and the full-100 check was neutral at
  `3.386e-7` adjusted / `2.989e-6` MSE / `3.085e10` effective compute with no
  failures. Do not promote without a stronger paired win. A positive log-space
  power correction, using `sqrt(target/sample) ** 1.5` instead of the linear
  `1.5x` scale update, also lost at `3.559e-7` adjusted / `3.135e-6` MSE /
  `3.085e10` effective compute over 76 returned MLPs, with four worker
  download failures. A global energy thermostat that scaled the whole
  first-successor centered ensemble by one total variance ratio, preserving
  correlations more aggressively than per-coordinate scaling, was also not
  enough: `1.5x` scored `3.319e-7` adjusted / `2.929e-6` MSE / `3.084e10`
  effective compute with no failures. Clipping the current per-coordinate
  first-successor scale factors around one was close but not actionable:
  a `1.5` cap scored `3.232e-7` adjusted / `2.852e-6` MSE / `3.084e10`
  effective compute, while a looser `2.0` cap scored `3.367e-7` adjusted /
  `2.974e-6` MSE / `3.081e10` effective compute, both with no failures.
  Combining the `1.5` cap with 12 blocks did not recover the lower-compute
  frontier: it scored `3.471e-7` adjusted / `3.301e-6` MSE / `2.856e10`
  effective compute over 79 returned MLPs, with one worker returncode failure.
  An ensemble-derived kurtosis gate for the first-successor variance strength,
  damping coordinates whose preactivation ensemble looked less Gaussian, landed
  near the scale-cap result but was not promotable: damping `0.25` scored
  `3.233e-7` adjusted / `2.852e-6` MSE / `3.084e10` effective compute over
  78 returned MLPs, with two worker returncode failures.
  Weighting the first-successor variance strength by the next layer's outgoing
  weight energy, intended to spend marginal correction only on coordinates with
  high downstream sensitivity, also stayed behind the default frontier:
  `3.394e-7` adjusted / `2.994e-6` MSE / `3.084e10` effective compute with no
  worker failures.
  Restoring the original first-successor correlation geometry after the useful
  marginal variance scale, by mapping the corrected ensemble to the original
  post-ReLU correlation matrix with corrected marginal variances, lost on both
  MSE and compute: `3.883e-7` adjusted / `3.148e-6` MSE / `3.353e10`
  effective compute with no worker failures.
  Clipping negative adjusted activations back to nonnegative support after the
  current first-successor variance scale also lost, indicating that the
  unconstrained centered scale's geometry is more useful than enforcing ReLU
  support at that point: `3.624e-7` adjusted / `3.193e-6` MSE / `3.085e10`
  effective compute with no worker failures.
  Applying the same variance-ratio correction to centered preactivations before
  the first-successor ReLU, instead of to centered post-ReLU activations, was a
  decisive loss at `1.5x`: `5.821e-7` adjusted / `5.130e-6` MSE /
  `3.089e10` effective compute with no failures.
  Replacing the first-layer recolor matrix inverse with `fnp.linalg.solve`
  was semantically equivalent but did not produce a useful scorer-path win:
  `3.520e-7` adjusted / `3.100e-6` MSE / `3.083e10` effective compute with
  no failures, so the established inverse expression was kept.
  Halving the first-layer antithetic Hadamard matmul, by computing only the
  positive block halves and reconstructing `relu(-pre)` for the antithetic
  halves, preserves the layer-0 ensemble up to row order and floating summation
  while removing redundant first-layer FLOPs. The default `make fly` proof
  after this change scored `3.698e-7` adjusted / `3.036e-6` MSE /
  `3.224e10` effective compute over 80 returned MLPs with no failures. Raw
  `flops_mean` dropped to `2.942e10`, about `1.40e9` below the prior
  `3.082e10` effective-compute checkpoint and consistent with the expected
  layer-0 mechanism, but residual wall-time charge made the reported effective
  compute noisy and higher on this first-80 run. The adjusted-score move from
  the documented `3.353e-7` baseline is about 10% worse, inside the repository
  noise caveat, with MSE still near the prior `2.962e-6`.
  A chi-radial variant that stratifies first-layer Hadamard row radii using
  Wilson-Hilferty chi quantiles lost on the first clean Fly check:
  `4.040e-7` adjusted / `3.540e-6` MSE / `3.136e10` effective compute over
  80 returned MLPs with no failures, so it remains only `hadamard_chi`.
  The explicit 24-block diagnostic scored `3.384e-7` adjusted / `1.639e-6`
  MSE / `5.634e10` effective compute over 80 returned MLPs with no failures.
  Since the MSE moved toward the expected variance-halving value rather than
  staying near `2.9e-6`, the current 13-block route appears substantially
  variance-limited even though 24 blocks is not an adjusted-score candidate.
  Replacing only the large ensemble propagation matmuls with plain recursive
  Strassen arithmetic reduces actually executed array arithmetic without
  modifying or bypassing flopscope accounting; keep it flagged for owner
  rules-spirit review before submission use. One Strassen level at 13 blocks
  cut raw `flops_mean` to `2.612e10` and scored `3.343e-7` adjusted /
  `3.141e-6` MSE / `2.910e10` effective compute, then replicated at
  `3.283e-7` adjusted / `2.974e-6` MSE / `3.011e10` effective compute, both
  over 80 returned MLPs with no failures. The adjusted scores are close to but
  not >15% better than the documented `3.353e-7` default, so `hadamard_st1`
  was not promoted. Two Strassen levels reduced raw `flops_mean` further to
  `2.335e10`, but residual wall-time charge rose sharply and the run scored
  `3.482e-7` adjusted / `3.122e-6` MSE / `2.960e10` effective compute with no
  failures. Because L2 effective compute stayed above the `2.72e10` floor and
  residual charge exceeded 15% of raw FLOPs, no block-reinvestment or L3 run
  was taken from this implementation.
  Batching all Strassen leaves into one `fnp.einsum("brk,bkc->brc", ...)` per
  propagation matmul preserved the raw L2 Strassen FLOP count but did not fix
  the residual-time problem. An initial leaf-ordering bug had correct
  `2.335e10` raw FLOPs but broken predictions (`1.190e-1` adjusted /
  `1.119e0` MSE / `2.854e10` effective compute), so it was discarded. After
  fixing the leaf order, `hadamard_st2` scored `3.423e-7` adjusted /
  `3.205e-6` MSE / `2.949e10` effective compute with `2.335e10` raw FLOPs and
  `6.134e9` residual compute, again over 80 returned MLPs with no failures.
  Since effective compute remained above the `2.72e10` floor and residual
  charge stayed about 26% of raw FLOPs, no block reinvestment or L3 run was
  justified. The batched `hadamard_st1` check also regressed to `3.657e-7`
  adjusted / `3.225e-6` MSE / `3.100e10` effective compute with `2.612e10`
  raw FLOPs and `4.883e9` residual compute. Do not promote batched Strassen.
  Promoting the earlier recursive `st1` remains rules-spirit-review material
  because its raw arithmetic saving is deterministic, but its replicated
  adjusted-score edge over the documented default was only about 2%, not a
  >15% scoring win.

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
  Replacing repeated Sylvester-Hadamard blocks with 13 deterministic quadratic
  chirp Hadamard bases, intended to diversify fourth-order aliasing while
  preserving orthogonal sign cubature, also lost: `3.717e-7` adjusted /
  `3.268e-6` MSE / `3.095e10` effective compute with no failures.
  A final-row block jackknife over odd/even Hadamard block halves also lost as
  a finite-block bias correction: +20% scored `3.758e-7` adjusted /
  `3.304e-6` MSE / `3.095e10` effective compute, while -20% scored
  `3.615e-7` adjusted / `3.193e-6` MSE / `3.081e10` effective compute, both
  with no failures.
  A mid-network Gaussian preactivation restart at layer 8, using the current
  ensemble preactivation mean/covariance to regenerate a fresh Hadamard
  Gaussian ensemble before continuing, lost badly at `1.775e-6` adjusted /
  `1.479e-5` MSE / `3.263e10` effective compute.
  Late-layer block pruning, motivated by the score-efficient compute floor and
  late-layer mean-error damping, also lost: keeping 13 blocks through layer 8
  then pruning to 11 blocks scored `3.446e-7` adjusted / `3.378e-6` MSE /
  `2.774e10` effective compute, pruning after layer 12 to 11 blocks scored
  `3.813e-7` adjusted / `3.664e-6` MSE / `2.843e10` effective compute, and
  pruning after layer 8 to 12 blocks scored `3.850e-7` adjusted / `3.580e-6`
  MSE / `2.929e10` effective compute. The lower multiplier did not compensate
  for the final-layer MSE hit.
  Returning cheap placeholder rows for hidden layers while preserving the
  final-layer propagation and final mean was attempted because the leaderboard
  ranks on final-layer adjusted score and hidden rows are diagnostic. It did
  not reduce effective compute and hurt robustness: one 80-return Fly run had
  one worker failure, `3.607e-7` adjusted / `3.183e-6` MSE /
  `3.083e10` effective compute, with all-layer MSE degraded as expected.
  Trimming the highest and lowest final-layer Hadamard block mean per coordinate
  was neutral and stayed inside Fly noise at `3.343e-7` adjusted / `2.952e-6`
  MSE / `3.083e10` effective compute with no failures, so it was not promoted.
  Coordinatewise median aggregation of final-layer block means was much worse,
  suggesting that mean cancellation across Hadamard blocks is important:
  `5.216e-7` adjusted / `4.602e-6` MSE / `3.083e10` effective compute with no
  failures.
  Clipping high-leverage sample rows after the first-successor variance match,
  as a robust-cubature attempt to limit rare large activation radii before they
  propagate through later layers, also hurt raw MSE: a loose `2.0x` RMS-radius
  cap scored `3.578e-7` adjusted / `3.152e-6` MSE / `3.092e10` effective
  compute with no failures.
  Final-row inverse-variance weighting across Hadamard block means also lost:
  a 50% blend scored `3.508e-7` adjusted / `3.089e-6` MSE / `3.088e10`
  effective compute, and a 20% blend scored `3.427e-7` adjusted / `3.024e-6`
  MSE / `3.085e10` effective compute.
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
  Replacing the full first-layer Cholesky covariance transport with diagonal
  coordinate transport reduced compute but lost too much accuracy:
  `3.781e-7` adjusted / `3.494e-6` MSE / `2.945e10` effective compute with no
  failures. A first-successor projected covariance correction in the top 32
  next-weight right-singular directions also lost badly despite being gentler
  than full covariance recoloring: `9.209e-7` adjusted / `8.004e-6` MSE /
  `3.128e10` effective compute with no failures.
  A downstream-aware orthogonal gauge inside the first-layer covariance
  transport, preserving the exact first-layer mean/covariance while rotating
  the whitened ensemble toward the next weight metric, was a decisive loss:
  `8.658e-7` adjusted / `7.571e-6` MSE / `3.110e10` effective compute over
  79 returned MLPs, with one worker returncode failure.
  A radial transport of the centered first-layer recolored ensemble, shrinking
  or expanding each sample toward the exact total covariance trace to reduce
  higher-order row-radius error, also lost at half strength:
  `4.117e-7` adjusted / `3.629e-6` MSE / `3.084e10` effective compute with no
  worker failures.
  A two-radius fourth-moment mixture after the first-layer covariance recolor,
  alternating centered-row scales while recentering to keep the first reported
  mean fixed, was catastrophic even at half strength: `1.353e-5` adjusted /
  `1.191e-4` MSE / `3.090e10` effective compute with no worker failures.
  Retrying final-only Gaussian marginal mean correction after the `1.5x`
  first-successor variance update as a 50% output blend also lost:
  `3.700e-7` adjusted / `3.262e-6` MSE / `3.085e10` effective compute over
  79 returned MLPs, with one failed worker. Smaller final-only Gaussian
  control-variate pulls were not robust either: a 10% blend initially scored
  `3.193e-7` adjusted / `2.818e-6` MSE / `3.083e10` effective compute with one
  failed worker, but a clean replicate bounced to `3.481e-7` adjusted /
  `3.067e-6` MSE / `3.090e10` effective compute, and a 20% blend lost at
  `3.595e-7` adjusted / `3.170e-6` MSE / `3.084e10` effective compute.
  Final-layer sample-cumulant
  Edgeworth blends were also not robust: 50% scored `3.504e-7` adjusted /
  `3.082e-6` MSE, 20% initially scored `3.263e-7` adjusted / `2.870e-6` MSE,
  10% bounced to `3.409e-7` adjusted / `2.999e-6` MSE, and a full-100 check
  of the 20% blend settled at `3.458e-7` adjusted / `3.034e-6` MSE /
  `3.102e10` effective compute over 99 returned MLPs with one worker failure.
  First-successor row-only Gaussian control variates, which changed reported
  rows without mutating the propagated ensemble, did not provide an actionable
  final-layer signal: 10% scored `3.368e-7` adjusted / `2.971e-6` MSE /
  `3.085e10` effective compute, while 20% scored `3.590e-7` adjusted /
  `3.172e-6` MSE / `3.082e10` effective compute.
  A final-row pull toward an independently propagated diagonal Gaussian
  mean-field estimator also lost badly even at 5% blend:
  `5.860e-7` adjusted / `5.148e-6` MSE / `3.096e10` effective compute with no
  failures.
  Blending the Hadamard route with the existing analytical K=3/r=1 propagation
  was not viable as a control variate because the fixed analytical pass
  exhausted the combined budget even at a 5% blend: the Fly run returned
  `9.220e-01` adjusted / `9.220e-01` MSE / `2.769e11` effective compute, with
  all 80 returned MLPs reporting `combined_budget_exhausted`.
  A quadratic Hermite control variate for reported ReLU means, using fitted
  zero-mean `H2` features from each layer's current preactivation ensemble,
  lost on score after residual compute: all-row QCV at 100% scored `3.694e-7`
  adjusted / `3.042e-6` MSE / `3.279e10` effective compute, and final-row-only
  QCV at 100% scored `3.415e-7` adjusted / `3.010e-6` MSE / `3.090e10`
  effective compute, both with no failures.
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
  A third-cumulant Edgeworth target for the first-successor marginal variance
  was tried as a more theoretical replacement for the Gaussian variance target
  while preserving the existing one-layer variance-match hook. Both signs lost:
  50% scored `3.689e-7` adjusted / `3.248e-6` MSE / `3.088e10` effective
  compute, and -50% scored `3.875e-7` adjusted / `3.419e-6` MSE /
  `3.086e10` effective compute, with no worker failures.
  A first-layer cubic marginal transport, matching the exact ReLU half-Gaussian
  third central moment after the first covariance recolor and before later
  propagation, also damaged the useful joint geometry: 50% strength scored
  `5.020e-7` adjusted / `4.419e-6` MSE / `3.090e10` effective compute over
  79 returned MLPs, with one worker returncode failure.
  Pre-ReLU gate-rate calibration on the first successor, shifting each
  coordinate's threshold toward the Gaussian marginal active probability before
  applying the existing post-ReLU variance scale, also lost badly:
  `4.295e-7` adjusted / `3.784e-6` MSE / `3.085e10` effective compute with no
  worker failures.
  Heat-kernel smoothing of the ReLU kink during the Hadamard propagation,
  replacing each hard ReLU with `E[ReLU(pre + noise)]` at 5% of each layer's
  ensemble preactivation standard deviation, also lost and spent extra compute:
  `4.476e-7` adjusted / `3.427e-6` MSE / `3.547e10` effective compute with no
  worker failures. Restricting the same kink smoothing to the final scored
  layer avoided most of the extra compute but still worsened raw MSE: a 2%
  final-layer bandwidth scored `3.678e-7` adjusted / `3.221e-6` MSE /
  `3.111e10` effective compute with no worker failures.
- **Zero-mean arc-cosine and conditional-quadrature K=2 covariance updates.**
  These replaced the simple gain covariance approximation, but nonzero later
  pre-activation means and numerical instability made them worse than the
  original K=2 route. Blending the current Hadamard route 5% toward an
  independently propagated full gain-covariance analytical estimate also lost:
  `3.656e-7` adjusted / `3.048e-6` MSE / `3.266e10` effective compute, with
  the extra covariance pass hurting the score multiplier.

## Benchmarking Notes

Use current scorer-path comparisons, not stale flops-only proxies. For
estimator changes, follow [`AGENTS.md`](AGENTS.md): compile `estimator.py` and
use the Fly fast runner by default unless the owner asks for a different proof.
For docs-only changes, a link/search check and Markdown sanity are sufficient.
