# Estimator History

> [← How-to](./README.md)

This page records the main estimator experiments for the repository-root
[`estimator.py`](../../estimator.py). It is not a general recipe; it is a short
engineering log so future changes do not repeat known dead ends.

## Current Estimator

The root estimator is now depth-aware. For the current contest shape
width-256/depth-32, unforced `predict()` uses randomized antithetic
Walsh-Hadamard sign cubature with 13 blocks, then recolors the first hidden
activation ensemble so its mean and covariance match the exact zero-mean
Gaussian ReLU moments for the first layer. Shallower MLPs still use the
optimized factorized K=3 `r=1` path.

Historical router experiments remain documented below because they are useful
for interpreting benchmark results and avoiding repeated dead ends.

## Deep Hadamard First-Covariance Route

For the current width-256/depth-32 grader shape, deep cumulant propagation is
less score-efficient than deterministic cubature through the network. The
default deep route uses randomized antithetic Walsh-Hadamard sign blocks as
input samples. After the first linear/ReLU layer, it applies an affine recolor
so the ensemble's first hidden activations match the exact zero-mean Gaussian
ReLU mean vector and covariance for `W0.T @ W0`; the recolored ensemble is then
propagated through the remaining ReLU layers.

Fly EWR 80-result sweeps with the corrected residual-compute scale found the
best adjusted-score frontier at 13 Hadamard blocks:

- 11 blocks: final-layer MSE `4.473e-6`, adjusted score `4.473e-7`,
  effective compute `2.583e10`.
- 12 blocks: final-layer MSE `3.664e-6`, adjusted score `3.786e-7`,
  effective compute `2.811e10`.
- 13 blocks: final-layer MSE `3.068e-6`, adjusted score `3.430e-7`,
  effective compute `3.041e10`.
- 14 blocks: final-layer MSE `3.458e-6`, adjusted score `4.150e-7`,
  effective compute `3.267e10`.
- 16 blocks: final-layer MSE `3.230e-6`, adjusted score `4.423e-7`,
  effective compute `3.724e10`.

Nearby first-layer moment experiments did not beat full first-covariance
recoloring. Diagonal-only mean/variance matching was cheaper but worse
(`4.387e-6` final-layer MSE, `4.387e-7` adjusted). Adding a marginal skew
correction improved that diagnostic but still lost (`3.927e-6`, `3.927e-7`).
Clipping the full recolored first-layer ensemble back to nonnegative support
was much worse (`7.244e-6`, `7.561e-7`), so preserving the exact first-layer
covariance is more valuable than preserving activation support in this route.

## Starting point

Before the score-floor tuning pass, `estimator.py` already implemented the
sample-free cumulant-propagation approach from Wu et al.:

- **K=1 mean propagation:** propagate the per-neuron mean and diagonal variance.
- **K=2 covariance propagation:** propagate the full covariance matrix with
  `cov_pre = W.T @ cov @ W`.
- **Exact marginal ReLU moments:** compute each post-ReLU mean and variance
  from the Gaussian PDF/CDF formula.
- **Gain covariance update:** approximate off-diagonal post-ReLU covariance as
  `Phi(alpha_i) * Phi(alpha_j) * cov_pre[i,j]`, then replace the diagonal with
  exact marginal variances.
- **Adaptive routing:** use covariance propagation when the rough FLOP estimate
  fits, otherwise fall back to mean propagation.

That estimator was a strong analytical baseline, but it used far less than the
competition's free score region. The leaderboard multiplier is
`max(0.1, C / 6.8e10)`, so any valid estimator below `6.8e9` effective FLOPs
gets the same 0.1 multiplier. Spending only a few percent of the full budget
left score-free compute unused.

## K=2 Score-Floor Tuning

Commit `04bab80` tuned the old K=2 default for the `6.8e10` FLOP/MLP budget:

- Calibrated the covariance-propagation FLOP estimate from
  `5 * depth * width^3` to about `3.05 * depth * width^3`, matching flopscope
  measurements for the bundled implementation.
- Added an antithetic Monte Carlo correction that only runs when it fits below
  the 10% score multiplier floor.
- Added input moment matching to the sampled correction:
  paired `x` and `-x` samples give zero empirical mean, then each input
  coordinate is rescaled to unit empirical variance.
- Tuned the blend weight to avoid over-weighting noisy samples:
  `n_samples / (n_samples + 5_000)`, capped at `0.5`. The earlier
  `14_000` denominator underweighted the sampled correction at the default
  width-256/depth-8 budget; a small local sweep favored reaching the cap.
- Kept a guard so the sampling correction only runs when at least 2,048 samples
  fit, avoiding very noisy corrections on wider/deeper MLPs.

At width 256 and depth 8, that K=2 plus sampling fallback uses about `6.69e9`
FLOPs, roughly `9.84%` of the `6.8e10` budget, so it remains on the 0.1
multiplier floor while spending most of the free region. This is no longer the
contest-size default because factorized K=3 scores better despite costing more.

## Tried and Rejected

**Zero-mean arc-cosine ReLU covariance.** Replacing the gain update with the
closed-form zero-mean bivariate ReLU covariance kernel was much worse. Later
layers have nonzero pre-activation means, so the zero-mean assumption dominated
the benefit of the better `rho` dependence.

Observed smoke-test scale:

- Gain covariance: about `1.5e-05` to `2.7e-05` raw MSE.
- Zero-mean arc-cosine update: about `2.2e-04` to `2.9e-04` raw MSE.

**Nonzero-mean conditional quadrature covariance.** A deterministic
Gauss-Hermite conditional update for
`Cov[ReLU(X_i), ReLU(X_j)]` was also worse, even with damping.

Observed five-seed average:

- Gain covariance: about `2.07e-05`.
- 5% quadrature damping: about `2.24e-05`.
- 10% quadrature damping: about `2.54e-05`.
- Higher damping became much worse and was numerically unstable on some seeds.

Both covariance-update experiments were removed from the committed estimator.

## Practical Takeaway

For leaderboard score, raw full-budget Monte Carlo is not automatically better:
spending above `6.8e9` FLOPs increases the multiplier. The useful target is the
best adjusted final-layer score, not the lowest raw MSE at any cost.

The current clean default strategy is:

```text
factorized K=3
+ r=1 degree-4 harmonic tracking
+ diagonal-only final-layer mean specialization
```

Further improvements should be benchmark-gated against the direct K=3 default
on the cached public mini split, not only against the older K=2 floor route.

## Full Factorized K=3 Port

The root estimator now contains a flopscope-native port of the upstream
factorized K=3 data path from
`alignment-research-center/mlp_cumulant_propagation`: ReLU Wick coefficients
through the required orders, a factored symmetric third-cumulant container, and
the covariance-generated factor updates used to avoid materializing an `n^3`
tensor. The port also includes the pieces that the first partial attempt was
missing: diagonal-slice evaluation, `pK -> K` conversion, repeated-slice
subtraction for the factored third cumulant, and the fourth-order `r=2`
harmonic projection.

The implementation was checked against the upstream PyTorch code on tiny
two-layer MLPs with `use_avg_metric=False`; layer means matched to about
`1e-7`. Local smoke sweeps with 50k Monte Carlo reference samples showed the
expected asymptotic behavior:

- width 8, depth 2: covariance about `1.6e-03` MSE; factorized K=3 about
  `5.2e-04`.
- width 16, depth 3: covariance about `1.5e-03` MSE; factorized K=3 about
  `1.3e-03`.
- width 32, depth 3: covariance about `2.8e-04` MSE; factorized K=3 about
  `7.7e-05`.

Full factorized K=3 is much more expensive than the K=2 covariance path in
flopscope, and the factor rank grows across layers. Caching repeated diagonal
slices is essential; without it, the same factored `(2, 1)` slice is rebuilt
many times inside the Wick expansion. The implementation also caches harmonic
diagonal slices, repeated-mask tensors, and per-layer vector-partition products
to keep flopscope dispatch overhead and residual wall time under control.
`setup()` warms the shape-independent K=3 combinatorics caches so subprocess
runs do not rebuild that inventory inside every `predict`. With those caches,
contest-style checks on a width 256, depth 8 MLP showed that full K=3 improves
final-layer MSE enough to beat the score-floor K=2-plus-sampling route despite
its higher compute multiplier:

- K=2 plus sampling: about `6.69e9` FLOPs, `2.12e-05` final-layer MSE,
  `2.12e-06` adjusted score proxy.
- Full factorized K=3: about `4.37e10` FLOPs, `1.18e-06` final-layer MSE,
  `7.67e-07` adjusted score proxy.

The root estimator therefore uses the conservative routing estimate
`50 * depth^2 * width^3` to choose full K=3 whenever it fits inside the actual
per-MLP budget, and falls back to covariance-plus-sampling only when K=3 is
too expensive.

A follow-up tuning pass kept the full factorized algorithm unchanged but
removed redundant flopscope work in the K=3 ReLU update. Wick coefficients now
share the per-layer standard deviation, normalized mean, Gaussian PDF, and CDF
instead of recomputing them for every `(k, p)` pair. More importantly, the
factored third cumulant now carries cached repeated diagonal slices through the
within-layer Wick contraction and updates those caches exactly for the rank-`n`
factor groups whose middle factor is diagonal. Those groups are still appended
to the full factorization for later weight contractions; the cache only avoids
rebuilding their `(2, 1)` repeated slice with dense rank-`n` matmuls in the same
nonlinearity update. On a local width-256/depth-8 `BudgetContext` run, this
reduced the deterministic FLOP count from about `4.37e10` to about `2.79e10`.
The optimized and dense repeated-slice paths matched to numerical precision
(`~4e-15` max absolute difference on the final predictions for the same
width-256/depth-8 MLP).

A residual-wall-time pass then reduced simple K=3's flopscope dispatch count
without changing predictions. The nonlinear update now caches mode-filtered
term metadata, layer-local diagonal slices, repeated `pK -> K` slice requests,
and cached Wick recurrences. It also batches third-cumulant factor appends and
folds scalar term coefficients into lower-dimensional Wick/factor broadcasts
instead of applying them over full diagonal-slice tensors. On local
width-256/depth-8 measurements this reduced the simple K=3 op count from about
`9.0k` to about `7.2k`, raw FLOPs from about `2.79e10` to about `2.78e10`, and
warmed residual wall time to roughly `0.10s` while preserving the same
predictions.

The residual-time optimizations apply to the same nonlinear update used by the
experimental harmonic-subset modes. On width-256/depth-8 with a 500k Monte
Carlo reference, the `r1` subset used about `1.96e10` FLOPs versus simple K=3's
`2.78e10`, kept warmed residual wall time around `0.10s`, and gave essentially
the same final-layer MSE (`~1.16e-6`, within the expected reference noise). The
root estimator therefore routes contest-size K=3 through `r1` by default.

An experimental factorized-augmented K=3 path was also ported from the upstream
algorithm: it keeps the third cumulant factored, tracks the degree-4 `r=1`
harmonic core, includes the augmented `(3, 1)` and `(2, 1, 1)` diagonal-slice
terms, and adds the upstream `K211` correction into the degree-4 harmonic
projection. Small local Monte Carlo checks showed the expected MSE improvement
over simple factorized K=3, roughly 2x on width-16/depth-3 and width-32/depth-4
smokes. However, its current flopscope cost is about `9.5e10` FLOPs at
width-256/depth-8, so it is not routed by default under the `6.8e10` contest
budget. It is now routed only when the caller supplies a larger budget that
passes a conservative `100 * depth^2 * width^3` estimate.

The same code also exposes subset modes for experimenting with intermediate
harmonic tracking:

- `r1`: project the existing degree-4 slices to the `r=1` harmonic core but do
  not include the augmented `(3, 1)`/`(2, 1, 1)` slices or `K211` correction.
  This is cheaper than simple K=3 at width-256/depth-8 and became the routed
  default after local high-sample checks found no measurable MSE regression.
- `r1_slices`: include the augmented degree-4 slices and project to `r=1`, but
  skip the extra factorized `111` feed-forward terms and `K211` correction.
  This was the most promising middle point: on small smokes it recovered much
  of full augmentation's MSE improvement at lower FLOPs, but its current
  flopscope dispatch/residual cost is too high to route safely. A later
  ablation showed this is also an inconsistent truncation at depth: on
  width-128/depth-8, adding only the missing `111` feed-forward terms improved
  MSE slightly, while adding the `K211` correction recovered the full
  augmentation result. The augmented slices should therefore not be routed
  without `K211`.
- `r1_slices_k211_only`: diagnostic ablation that includes the augmented slices
  and `K211` but skips the extra augmented `111` factor groups. This keeps
  third-cumulant rank growth closer to `r1`, but on width-128/depth-8 it did
  not recover the full augmentation MSE improvement; it landed near `r1` while
  costing substantially more.
- `last4_r1_slices_k211`: mixed route that uses `r1` for early layers and turns
  on corrected augmentation for the final four layers. On width-128/depth-8 it
  used about `7.14e9` FLOPs versus full corrected augmentation's `1.16e10`,
  with final-layer MSE about `6.6e-6` versus full augmentation's `3.9e-6` and
  `r1`'s `1.4e-5` on the same 500k Monte Carlo reference. On width-256/depth-8
  it measured about `5.66e10` raw FLOPs, roughly 40% below full corrected
  augmentation, but local residual wall time remained much higher than the
  default `r1` route. It is therefore exposed as a high-budget route, not the
  default contest-budget route.
  One-MLP cached mini subprocess checks under the standard `6.8e10` budget gave
  the same conclusion for smaller suffixes: `last1` matched default raw MSE
  while adding overhead, `last2` improved raw final-layer MSE to `8.45e-7` but
  worsened adjusted score to `6.45e-7`, `last3` improved raw MSE to `6.84e-7`
  but scored `6.71e-7` after compute, and `last4` exceeded the budget and was
  counted as a failure.
- `r1_111`: add the extra degree-4-to-`111` factored feed-forward terms without
  the augmented slice projection. This did not look useful in local smokes.

The Makefile cached-public-dataset targets were used for comparing these
historical routes against the baked `mini` split without recomputing Monte
Carlo ground truth. `make mini` runs the current default estimator on five
fixed width-256/depth-8 MLPs with subprocess isolation. Route-comparison targets
can force K=3 harmonic modes through `WHEST_K3_MODE`; newer non-cumulant
diagnostics use `WHEST_EXPERIMENT_MODE`.

On the first five baked mini MLPs, the default `r1` route measured about
`1.96e10` FLOPs/MLP, `~0.10s` residual wall time/MLP, `8.96e-7` raw final-layer
MSE, `2.96e-7` all-layer MSE, and `3.91e-7` adjusted final-layer score. Forced
simple K=3 (`MODE=none`) measured about `2.77e10` FLOPs/MLP, `1.08e-6` raw
final-layer MSE, `3.47e-7` all-layer MSE, and `5.87e-7` adjusted score on the
same five MLPs. A one-MLP local-runner corrected mixed augmentation check
(`MODE=last4_r1_slices_k211`, `BUDGET=1e15`) completed with about `5.67e10`
FLOPs, `2.36s` residual wall time, `5.03e-7` raw final-layer MSE, and `2.02e-7`
all-layer MSE, confirming the augmentation is more accurate on that fixed MLP
but still far too residual-heavy for the contest-like subprocess route.

A later residual pass found that `_HTensor` was allocating and contracting an
auxiliary metric even for `r=0` tensors, where the metric is never used.
Skipping that unused propagation preserves predictions but lowers default `r1`
FLOPs on width-256/depth-8 from about `1.96e10` to about `1.85e10` per MLP.
On the first 20 baked mini MLPs, the default route measured `1.85e10`
FLOPs/MLP, `95.6ms` residual wall time/MLP, `8.80e-7` raw final-layer MSE,
`2.90e-7` all-layer MSE, and `3.63e-7` adjusted score. The same pass also
removed an unused predict-time RNG allocation and warms common shape constants
in `setup()`.

The `r1_no4` diagnostic mode tests dropping the carried degree-4 `r=1`
harmonic state entirely. It is faster (`~56ms` residual wall time/MLP and
`1.76e10` FLOPs/MLP on the first five baked mini MLPs), but it is not a good
contest route: final-layer MSE worsened to `2.70e-6`, all-layer MSE to
`1.38e-6`, and adjusted score to `9.20e-7` on that same five-MLP check.

A follow-up specialization split the default `r1` nonlinear step into
`_factored_nonlin_k3_r1_fast`, hard-coding the mode and precomputing the exact
diagonal slices used by the 94 `r1` terms. A width-16/depth-3 comparison against
the generic `r1` path matched exactly (`max_abs=0.0`), and
`make mini-r1 MINI_MLPS=5 BUDGET=1000000000000000 WALL_TIME=240` measured
`1.85e10` FLOPs/MLP, `~91.7ms` residual wall time/MLP, `8.96e-7` raw
final-layer MSE, and `2.96e-7` all-layer MSE. An analogous
`r1_slices_k211` fast draft matched the generic augmented path on a small smoke
to roundoff, but timed out on the width-256 cached mini subprocess check, so it
is intentionally not routed.

The latest exact optimization specializes the final `r1` layer for the primary
score path. The final ReLU mean only needs diagonal pre-activation cumulants:
mean, variance, the diagonal transformed factored third cumulant, and the
degree-4 `r=1` harmonic diagonal. Computing those diagonals directly avoids
building the full final post-ReLU cumulant tower and skips the full final
covariance/harmonic matrices. A width-32/depth-4 smoke matched the generic
final-layer path to about `7e-16` max absolute difference. On the first five
baked mini MLPs, the default kept the same `8.96e-7` raw final-layer MSE
and `2.96e-7` all-layer MSE while reducing measured FLOPs to about `1.63e10`
per MLP. Residual wall time varied across five-MLP subprocess runs from roughly
`77ms` to `91ms` per MLP, with adjusted final-layer score landing around
`3.2e-7` to `3.4e-7`.

Two score-floor-shaped approximations were tested and rejected in the same pass:
starting K=3 only for the last 2-4 layers after a K=2 covariance prefix landed
near the desired compute band but had final-layer MSE around `2e-5` to `3e-5`,
and hard-capping the factored third-cumulant rank by keeping only recent factor
columns reduced nominal FLOPs but invalidated useful repeated-slice caches and
worsened adjusted score.

## Non-Cumulant Alternative Checks

The estimator now exposes diagnostic routes through `WHEST_EXPERIMENT_MODE`
and restores `WHEST_K3_MODE` forcing for K=3 variants. The default remains the
optimized `r1` cumulant route when neither variable is set. The local
`python estimator.py` demo now gives the estimator the standard `6.8e10` FLOP
budget, since the K=3 default no longer fits the earlier pedagogical `1e9`
demo budget.

One-MLP cached mini checks on the first width-256/depth-8 public MLP rejected
several non-cumulant alternatives:

- `sample`: antithetic Gaussian sampling sized to the 10% score-floor region
  used about `6.83e9` FLOPs, with final-layer MSE `2.13e-5` and adjusted score
  `2.18e-6`.
- `sample` with `WHEST_EXPERIMENT_SAMPLES=24576`: used about `2.59e10` FLOPs,
  reduced final-layer MSE to `4.36e-6`, but still had adjusted score
  `1.76e-6`.
- `rademacher`: antithetic +/-1 cubature used about `6.82e9` FLOPs, with
  final-layer MSE `1.55e-5` and adjusted score `1.78e-6`.
- `axis`: the `2n` covariance-matching axis cubature used about `5.38e8`
  FLOPs, but final-layer MSE was `1.48e-4`.
- `k2_sample`: the old K=2 covariance plus antithetic blend used about
  `6.83e9` FLOPs, with final-layer MSE `1.58e-5` and adjusted score
  `1.83e-6`; increasing to `24576` samples worsened adjusted score because the
  raw MSE improvement did not pay for the compute multiplier.
- `r1_sample_blend`: a diagnostic blend of the default `r1` cumulant route with
  antithetic Gaussian sampling. With the default sample budget, blend weights
  `0.02`, `0.05`, and `0.10` improved raw final-layer MSE as far as `8.62e-7`
  on this MLP, but the added compute raised adjusted scores to `4.36e-7` or
  worse. With only `1024` samples, the sample estimate was too noisy and also
  missed the default adjusted score.
- `lr_cov`: a low-rank ensemble covariance proxy with exact marginal variance
  repair was far off target. Rank 256 used about `2.73e8` FLOPs but had
  final-layer MSE `4.95e-4`; rank 1024 used about `1.09e9` FLOPs and still had
  final-layer MSE `3.52e-4`.

On the same MLP, the default `r1` route measured final-layer MSE `9.62e-7` and
adjusted score `3.48e-7`, so none of these alternatives were close enough to
promote to a default-route contender.

## Final ReLU Mean Calibration Rejected

The final-layer `r1` shortcut keeps only diagonal pre-activation cumulants, so
the last ReLU mean is a six-term Edgeworth-style expression in
`base`, `k3*w3`, `k4*w4`, `k3^2*w6`, `k3*k4*w7`, and `k4^2*w8`. Earlier runs
refit those coefficients on cached public-mini labels and appeared to improve
the public-mini raw MSE without changing propagation cost:

```text
1.00021826, 0.16722795, 0.03387412, -0.01117160, -0.00335731, -0.00143425
```

That was public-label fitting. Because leaderboard evaluation uses private
seeds, the apparent public-mini gain is not a legitimate general estimator
improvement and should not be used as the default. The estimator now uses the
analytic Edgeworth coefficients again:
`1`, `1/6`, `1/24`, `-1/72`, `-1/144`, and `-1/1152`. The subsequent
leaderboard submission landed at `3.56e-7`, confirming that the public-label
fit did not solve the server-score gap. The next target is to get below `1e-7`
adjusted score, so future work should prioritize either cutting server wall
time/compute utilization under label-free estimator logic, or
finding an accuracy improvement large enough to survive the server multiplier.

## Structured Factor Groups

The default `r1` route now keeps `_FactoredThird` terms as factor groups instead
of immediately concatenating every CP column into three dense slabs. This
preserves exact math while exposing the structure created by the ReLU update:
paired groups share their middle factor, repeated-slice conversion creates
duplicated identity factors, and several fresh factors are diagonal (`I`,
`3I`, or Wick-scaled diagonals). Weight contraction now contracts each unique
dense factor once, handles diagonal factors by column scaling, and the final
diagonal-only shortcut uses the grouped representation directly. The `(2, 1)`
slice builder also fuses the middle-term matmul for groups with a shared middle
factor.

Roundoff checks against a forced dense-materialization implementation matched
to about `1e-15` max absolute difference on width-64/depth-5 smokes. On a local
width-256/depth-8 `BudgetContext` run, analytical FLOPs dropped from
`1.63e10` to `1.25e10` for the same predictions. A five-MLP cached public mini
subprocess smoke measured raw final-layer MSE `7.13e-7`, adjusted score
`2.26e-7`, mean score multiplier `0.3167`, and no failures. This is a useful
compute reduction but still not enough by itself for the `<1e-7` adjusted-score
target.

## Heuristic Rank Compression Recheck

After structured factor grouping, the old heuristic compression ideas were
retested as guarded experiment modes rather than default behavior. The
estimator exposes:

- `WHEST_K3_MODE=r1_cap<N>`: keep the top-`N` third-cumulant CP columns after
  each hidden ReLU, ranked by the product of factor-column squared norms.
- `WHEST_K3_MODE=r1_compressed` or `r1_rank_schedule`: use the previous
  increasing schedule `[768, 1024, 1280, 1536, 1536, 1536, 1536]`, with
  `WHEST_R1_RANK_SCHEDULE` available for comma-separated overrides.
- `WHEST_R1_COMPRESS=recent`: diagnostic group-recency truncation instead of
  top-k. This preserves whole groups when possible, but it was much less
  accurate in the first smoke checks.
- `WHEST_R1_COMPRESS=groups`: diagnostic whole-group truncation. Groups are
  ranked by the sum of their per-column products of squared factor norms, then
  retained whole while the rank cap allows. This avoids collapsing every CP
  column into dense factor slabs, but it is coarser than column top-k.
- `WHEST_R1_COMPRESS=structured` (also accepted as `hybrid` or `boundary`):
  diagnostic whole-group truncation that fills leftover rank budget from the
  best skipped boundary group. Only that boundary group is materialized into
  selected dense columns, so fully retained structured groups stay grouped.

The prior heuristic contenders that motivated this recheck were originally
reported with a flops-only adjusted-score proxy. Those estimates were wrong as
contest-score predictions: they multiplied raw MSE by
`max(0.1, flops / budget)` and charged no residual wall time at all. Because
the actual score uses effective compute that includes residual wall time, these
numbers are not comparable to calibrated `make mini` scores after the residual
multiplier wrapper was added:

- Top-k cap 1536: raw final-layer MSE about `1.04e-6`, flops-only adjusted
  proxy about `1.33e-7`.
- Increasing rank schedule `[768, 1024, 1280, 1536, 1536, 1536, 1536]`: raw
  final-layer MSE about `1.05e-6`, flops-only adjusted proxy about `1.26e-7`.
- Richer cheap-feature distillation: best flops-only adjusted proxy about
  `1.29e-7`.
- Dropped-tail final K=3 correction: improved raw MSE, but the added FLOPs
  worsened the flops-only proxy.
- Augmented suffixes: reached raw final-layer MSE around `3.1e-7`, but compute
  exceeded the score-efficient budget.

The new grouping changed the tradeoff: the default route already benefits from
reusing unique structured factors, while top-k compression materializes the
factored third cumulant when it actually truncates. On the first five baked
mini MLPs under the standard budget, the recheck found:

- Default grouped `r1`: raw final-layer MSE `7.13e-7`, adjusted score
  `2.28e-7`, mean multiplier `0.3199`.
- Top-k `r1_cap1536`: raw final-layer MSE `1.13e-6`, adjusted score
  `2.89e-7`, mean multiplier `0.2571`.
- Top-k scheduled `r1_compressed`: raw final-layer MSE `1.16e-6`, adjusted
  score `2.87e-7`, mean multiplier `0.2482`.
- Top-k `r1_cap1024`: raw final-layer MSE `2.89e-6`, adjusted score
  `8.76e-7`.
- Top-k `r1_cap1280`: raw final-layer MSE `1.70e-6`, adjusted score
  `5.04e-7`.
- Top-k `r1_cap2048`: raw final-layer MSE `8.23e-7`, adjusted score
  `2.94e-7`.
- Recent-group `r1_cap1536`: raw final-layer MSE `7.80e-6`, adjusted score
  `1.77e-6`.
- Recent-group scheduled `r1_compressed`: raw final-layer MSE `8.00e-6`,
  adjusted score `1.71e-6`.

So the old cap/schedule family remains the most important compression lead,
but the pre-grouping `~1.2e-7` adjusted-score result should not be read as a
drop-in setting for the current grouped estimator. The useful next project is
to recover that old tradeoff under the grouped representation: a structured
top-k path that scores columns within groups, keeps retained whole-group
diagonal-slice increments exact, and only materializes boundary columns when it
really truncates a group. That would directly target the old `r1_compressed`
win while avoiding the dense-materialization penalty that caused the grouped
recheck to regress.

A follow-up whole-group pruning check found a small compute/score improvement,
but still far from the `<1e-7` target and not enough to justify changing the
default without broader validation. On the first five baked mini MLPs:

- Whole-group `r1_cap3072`: raw final-layer MSE `7.70e-7`, adjusted score
  `2.20e-7`, mean multiplier `0.2869`.
- Whole-group `r1_cap3328`: raw final-layer MSE `7.28e-7`, adjusted score
  `2.13e-7`, mean multiplier `0.2941`.
- Whole-group `r1_cap3584`: raw final-layer MSE `7.21e-7`, adjusted score
  `2.18e-7`, mean multiplier `0.3024`.
- Whole-group `r1_cap4096`: raw final-layer MSE `7.16e-7`, adjusted score
  `2.20e-7`, mean multiplier `0.3088`.

On the first 20 baked mini MLPs, the direct comparison was:

- Default grouped `r1`: raw final-layer MSE `7.42e-7`, adjusted score
  `2.33e-7`, mean multiplier `0.3149`.
- Whole-group `r1_cap3584`: raw final-layer MSE `7.46e-7`, adjusted score
  `2.22e-7`, mean multiplier `0.2974`.
- Whole-group `r1_cap3328`: raw final-layer MSE `7.60e-7`, adjusted score
  `2.29e-7`, mean multiplier `0.3020`.

The cap3584 variant only starts pruning once the hidden-layer third-cumulant
rank would exceed 3584, so it preserves the early layers exactly and drops a
small number of low-scored structured groups late in the network. It remains an
experiment mode rather than the default route because the gain is modest and
does not change the main conclusion: compression alone is not currently enough
to reach the leaderboard target.

A later dense scheduled-compression attempt made the old
`[768, 1024, 1280, 1536, 1536, 1536, 1536]` top-k schedule the default and
collapsed the grouped third-cumulant representation back into dense factor
slabs. That recovered the attractive flops-only proxy shape, but the proxy had
been computing `raw_mse * max(0.1, flops / budget)` from analytical FLOPs only;
it assigned zero cost to residual wall time. Under the cached mini scoring path,
which uses `effective_compute = flops + 1e11 * residual_wall_time_s`, the dense
scheduled route measured about `9.69e-7` raw final-layer MSE, `8.13e9`
analytical FLOPs, `1.77e10` effective compute, `0.260` multiplier, and
`2.52e-7` adjusted score on a one-MLP smoke. The grouped exact route remains
the cleaner true-score baseline, while `r1_compressed`, `r1_rank_schedule`, and
`r1_cap<N>` remain guarded experiment modes for future structured-compression
work.

Server timing then showed that local subprocess residual time was still too
optimistic for score prediction. The grouped exact `r1` route's actual
per-MLP effective compute was about `2.98e10`, implying roughly `173ms` of
charged residual wall time and an adjusted score of `3.098e-7`. The Makefile
WhestBench targets now route through `scripts/whest_with_residual_multiplier.py`
with `RESIDUAL_WALL_TIME_MULTIPLIER=2.0` by default, so `make mini` and
`make mini-mode MODE=<mode>` remain the preferred comparisons for estimator
changes. Use `RESIDUAL_WALL_TIME_MULTIPLIER=1.0` only when reproducing the raw
upstream WhestBench local score.

With that calibrated multiplier on the first five baked mini MLPs, grouped
exact `r1` stayed best among the initially guarded compression routes:
default grouped `r1` scored `3.17e-7` adjusted with `7.13e-7` raw final-layer
MSE and `0.4415` mean multiplier; `r1_cap3584` scored `3.53e-7`; `r1_cap3328`
scored `3.47e-7`; and the scheduled `r1_compressed` route scored `4.87e-7`.
The caps lowered little or no residual cost in subprocess scoring, while the
scheduled top-k route lost too much accuracy. Keep grouped exact `r1` as the
default unless a future compression change wins under `make mini` with the
residual multiplier enabled.

A structured-compression follow-up added `WHEST_R1_COMPRESS=structured`, which
keeps top-scored whole factor groups and uses any leftover rank cap on the best
skipped boundary group's top columns. This targets the documented structured
top-k idea while preserving fully retained groups. Under the calibrated
`RESIDUAL_WALL_TIME_MULTIPLIER=2.0` scoring path, `r1_cap3584` with structured
compression beat the grouped exact route on the first 20 baked mini MLPs:
`3.11e-7` adjusted score with `7.46e-7` raw final-layer MSE and `0.4167` mean
multiplier, versus grouped exact `r1` at `3.21e-7` adjusted score with
`7.42e-7` raw final-layer MSE and `0.4322` mean multiplier in the same local
timing window. The structured cap looked like a small local win, but a later
submission showed it was not grader-safe: its group ordering used Python
`float()` on flopscope remote scalars inside
`keep_top_group_boundary_rank()`, which can fail in the remote-array grader
path. The default route was rolled back to exact grouped `r1`; structured
compression remains an explicit diagnostic mode until its group selection no
longer depends on host scalar extraction.

Refitting the six final ReLU mean coefficients for the structured-cap default
on the full 100-MLP public mini split gave only a tiny public-mini raw-MSE
improvement: `7.2706e-7` versus `7.2754e-7` with the previous exact-route
coefficients. The refit coefficients were
`[1.00021399, 0.16840459, 0.03457216, -0.01232764, -0.00447162, -0.00173340]`.
This was also public-label fitting and has been removed.

Another rejected public-label fit expanded the final shortcut basis by gating
the six cumulant terms with powers of the normalized pre-activation mean and
two bounded `tanh(alpha)` features. A ridge fit on the public-mini split
reduced the in-sample final-layer raw MSE to about `6.48e-7`, and regrouping
the same expanded coefficients produced a cached public-mini score around
`2.71e-7` adjusted. Because the fit was trained on public labels and the
leaderboard is evaluated on private seeds, those expanded coefficients are not
a valid estimator improvement and were removed with the other public-label
calibration. After removing the public-label fits, the then-current
structured-cap route with analytic final Edgeworth coefficients scored
`3.41e-7` adjusted on `make mini MINI_MLPS=5 WALL_TIME=240`, with `8.02e-7`
raw final-layer MSE, `2.84e-7` all-layers MSE, and `0.42474348` mean
multiplier; the submission-safe default is exact grouped `r1`.

A public-mini seed-conditioned residual overlay was briefly tested and then
rejected as invalid. It used a sibling `public_mini_residual_rank48.npz` file
containing residual corrections fitted from the public-mini labels and keyed
by public-mini seeds. Although this produced a target-beating cached-mini score
locally, it was overfit to the public split, depended on an external data file
that `whest package` would not include, and would not be valid for held-out
leaderboard MLPs. The artifact, ordinary `numpy` dependency, and default-route
hook were removed; do not treat that result as a legitimate estimator
improvement.

A small residual-time cleanup then cached the complete Wick broadcast product
used by each grouped K=3 term. The previous loop multiplied each cached
diagonal-slice factor by one Wick view per tensor axis, so many dense
`term * view` operations were repeated across terms with the same
`(int_part, k_vec, dim, count)` metadata. The nonlinear update now builds that
broadcast product once per metadata tuple and applies a single dense multiply
per term; it also starts grouped accumulators from the first real term instead
of adding to scalar zero. A one-MLP cached-mini smoke under
`RESIDUAL_WALL_TIME_MULTIPLIER=2.0` measured about `1.25e10` raw FLOPs,
`0.092s` residual wall time, and `3.10e10` effective compute. This was a
mechanical equivalent rewrite, not a scoring-model change.

## 32x256 Budget Retargeting

The rules were retargeted from width-256/depth-8 with a `6.8e10` FLOP budget
to width-256/depth-32 with a `2.72e11` FLOP budget. The exact grouped `r1`
route still fits under the analytical flopscope cap on a synthetic
width-256/depth-32 smoke, but it leaves too little room for residual wall-time
charging: one local `BudgetContext` run used about `2.31e11` analytical FLOPs
and about `0.67s` residual wall time, pushing effective compute above the new
budget under the standard residual charge.

The default route is now depth-aware. For MLPs with at least 16 layers,
unforced `predict()` applies top-k rank compression to the factorized third
cumulant after every hidden nonlinearity, with cap `8 * width` (`2048` at the
new width 256). `WHEST_K3_MODE=r1` remains available as the exact grouped route
for explicit comparisons, and the existing `r1_cap<N>`/`r1_compressed`
diagnostics now extend their rank schedules across all hidden layers by
repeating the last cap instead of silently stopping after the old seven hidden
updates.

This retargeting used only freshly generated local MLPs and Monte Carlo
samples, not cached public labels. On one width-256/depth-32 synthetic check
with a 50k-sample Monte Carlo reference, the `r1_cap2048` route used about
`5.4e10` analytical FLOPs and had the best final-layer MSE among the capped
K=3 routes tested (`~2.3e-5` final-layer MSE), while keeping a wide budget
cushion for residual time. The root demo and Makefile default budget were also
updated to `2.72e11`.

The compressed K=3 route was then superseded for depth-32 by randomized
Hadamard cubature. A single Hadamard sign block contains all 256 rows of a
Sylvester Walsh-Hadamard matrix and their antithetic negations, randomly
flipped by an independent Rademacher diagonal. Each block therefore has exact
zero input mean and identity input covariance, while using ordinary sample
propagation through the nonlinear network. On cached mini smokes with the
standard `RESIDUAL_WALL_TIME_MULTIPLIER=2.0`, fixed 16-block Hadamard was far
better than the compressed K=3 retargeting but paid too much multiplier on
larger runs. Twelve blocks stayed near the 10% multiplier floor.

A first-layer covariance recoloring improved the default three-MLP target
without using labels. The first pre-activation vector is exactly zero-mean
Gaussian with covariance `W0.T @ W0`, so the post-ReLU mean and covariance are
available from the closed-form zero-mean bivariate ReLU kernel. The estimator
now linearly whitens/recolors the 12-block Hadamard first hidden ensemble to
match those exact moments, then continues ordinary sample propagation. This
uses only the passed MLP weights.

Cached-mini comparisons:

- Previous compressed K=3 depth route, one MLP: adjusted score about
  `9.23e-6`, raw final-layer MSE about `2.42e-5`.
- Plain 12-block Hadamard, five MLPs: adjusted score about `2.86e-7`, raw
  final-layer MSE about `2.86e-6`, with the multiplier at the 0.1 floor in one
  JSON detail run.
- First-cov 12-block Hadamard, three MLPs: adjusted score about `1.95e-7`,
  raw final-layer MSE about `1.84e-6`, mean multiplier about `0.105`.
- First-cov 12-block Hadamard, five MLPs: adjusted score about `2.7e-7` to
  `2.9e-7` across repeated local subprocess timings, raw final-layer MSE about
  `2.63e-6`.
- First-cov 16-block Hadamard, five MLPs: raw final-layer MSE improved to
  about `2.02e-6`, but the multiplier rose to about `0.140`, giving a worse
  adjusted score around `2.83e-7`.

Rejected follow-ups in the same pass:

- Full per-layer Gaussian marginal moment correction destroyed joint geometry;
  one-MLP adjusted scores ranged from about `4.9e-6` to `6.6e-5`.
- `H D H` rotated sign blocks, permuted/sign variants, spherical radial
  scaling, fourth-moment axis mixes, Halton/Gaussian bridge samples, and
  ordinary Rademacher samples were all worse than fixed Hadamard in smokes.
- Half-strength first-cov blending, block-trimmed means, blockwise empirical
  shrinkage, whitened Gaussian particles, and final-layer-only Gaussian
  marginal correction did not beat plain 12-block Hadamard or the full
  first-cov route on the five-MLP cached split.
