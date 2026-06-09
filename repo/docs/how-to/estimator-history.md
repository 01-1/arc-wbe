# Estimator History

> [← How-to](./README.md)

This page records the main estimator experiments for the repository-root
[`estimator.py`](../../estimator.py). It is not a general recipe; it is a short
engineering log so future changes do not repeat known dead ends.

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

## Current committed change

Commit `04bab80` tuned the root estimator for the `6.8e10` FLOP/MLP budget:

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

At width 256 and depth 8, the current estimator uses about `6.69e9` FLOPs,
roughly `9.84%` of the `6.8e10` budget, so it remains on the 0.1 multiplier
floor while spending most of the free region.

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
spending above `6.8e9` FLOPs increases the multiplier. The useful target is
therefore "best final-layer MSE under the 10% floor", not "lowest raw MSE at
the full budget".

The current clean strategy is:

```text
full covariance propagation
+ score-floor-aware antithetic, moment-matched Monte Carlo blend
```

Further improvements should be benchmark-gated against this baseline under the
same `6.8e9` effective-compute target.

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
- `r1_111`: add the extra degree-4-to-`111` factored feed-forward terms without
  the augmented slice projection. This did not look useful in local smokes.

The Makefile now includes cached-public-dataset targets for comparing these
routes against the baked `mini` split without recomputing Monte Carlo ground
truth. `make mini` runs the default estimator on five fixed width-256/depth-8
MLPs with subprocess isolation. `make mini-r1`, `make mini-simple`, and
`make mini-mode MODE=<mode> BUDGET=<flops>` force a specific K=3 harmonic route
through the same cached dataset. Long corrected-augmentation diagnostics can
use `make mini-mixed-local MINI_MLPS=1` or
`make mini-aug-local MINI_MLPS=1`, which
keeps the same baked ground truth but uses the local runner because the
subprocess harness currently times out before the corrected augmentation route
returns.

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
