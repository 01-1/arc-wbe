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
many times inside the Wick expansion. With that cache, contest-style checks on
a width 256, depth 8 MLP showed that full K=3 improves final-layer MSE enough
to beat the score-floor K=2-plus-sampling route despite its higher compute
multiplier:

- K=2 plus sampling: about `6.69e9` FLOPs, `2.12e-05` final-layer MSE,
  `2.12e-06` adjusted score proxy.
- Full factorized K=3: about `4.41e10` FLOPs, `1.18e-06` final-layer MSE,
  `7.67e-07` adjusted score proxy.

The root estimator therefore uses the conservative routing estimate
`50 * depth^2 * width^3` to choose full K=3 whenever it fits inside the actual
per-MLP budget, and falls back to covariance-plus-sampling only when K=3 is
too expensive.
