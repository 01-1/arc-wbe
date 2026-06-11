# Estimator History

> [← How-to](./README.md)

This page records the main estimator experiments for the repository-root
[`estimator.py`](../../estimator.py). It is not a general recipe; it is a short
engineering log so future changes do not repeat known dead ends.

## Current Estimator

The root estimator now runs the optimized factorized K=3 path directly. It
tracks the symmetric third cumulant plus the `r=1` degree-4 harmonic state and
uses a diagonal-only final-layer mean specialization for the primary score
path. The previous budget router was removed so `predict()` no longer switches
between augmented K=3 variants, covariance-plus-sampling, or mean propagation.

Historical router experiments remain documented below because they are useful
for interpreting benchmark results and avoiding repeated dead ends.

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

## Final ReLU Mean Calibration

The final-layer `r1` shortcut keeps only diagonal pre-activation cumulants, so
the last ReLU mean is a six-term Edgeworth-style expression in
`base`, `k3*w3`, `k4*w4`, `k3^2*w6`, `k3*k4*w7`, and `k4^2*w8`. Refitting only
these six coefficients on the cached public mini labels gave a stable raw-MSE
improvement without changing propagation cost. The calibrated coefficients are:

```text
1.00021826, 0.16722795, 0.03387412, -0.01117160, -0.00335731, -0.00143425
```

On the full 100-MLP public mini split, the fitted raw final-layer MSE was about
`7.22e-7`; leave-one-MLP-out validation was about `7.23e-7`, versus
`8.84e-7` for the analytic coefficients. This is now the default final shortcut
because it improves the accuracy side of the score at essentially unchanged
FLOPs. It does not by itself solve the server-score gap: with the current
server effective-compute multiplier around `0.49`, the adjusted score would
still land near `3.5e-7`.

The subsequent leaderboard submission landed at `3.56e-7`, confirming that the
calibrated final coefficients are active but that server wall time still leaves
the route far above the top-participant band. The next target is to get below
`1e-7` adjusted score, so future work should prioritize either cutting server
wall time/compute utilization without losing the calibrated raw-MSE gain, or
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
