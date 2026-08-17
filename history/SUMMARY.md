# Summary

A reading guide to ~240KB of estimator history: how the route got where it is,
what closed along the way, and which numbers are trustworthy.

## The problem

Estimate per-neuron post-ReLU activation means for width-256, depth-32 random
ReLU MLPs, given the weights, without a forward pass on the evaluation inputs.
Score is `MSE x max(0.1, C / 2.72e11)` where `C` is effective compute, so the
compute multiplier bottoms out at `0.1` and a `2.72e10`-FLOP entry pays the
same rate as a free one.

## Where it ended up

Sixteen blocks of randomized antithetic Walsh-Hadamard sign cubature. After the
first linear/ReLU layer the ensemble is linearly recolored so its mean and
covariance match the exact zero-mean Gaussian ReLU moments for `W0.T @ W0`; the
first layer uses only the positive half of each antithetic block and
reconstructs the negative half from negated preactivations. The recolored
ensemble then propagates in fp32 through the remaining layers with three
batched-leaf Strassen levels, with a `1.5x` variance-scale update applied to the
first subsequent ReLU ensemble only.

`2.667e-6` final-layer MSE, `2.74e-7` adjusted, `2.535e10` raw /
`2.832e10` effective compute. Best recorded grader submission `2.411e-7`
adjusted.

## Lineage

Each step replaced the previous one on a measured comparison.

| Step | Why it was replaced |
|---|---|
| K=2 covariance + sampling floor | Analytic-only accuracy floor |
| Factorized K=3 (`r1`) | Third cumulant helped, cost grew |
| Structured grouped `r1` | Kept the gain, cut the cost |
| Depth-32 retargeting, compressed K=3 | Shape moved 256x8 → 256x32, budget `6.8e10` → `2.72e11` |
| First-covariance Hadamard | Plain Hadamard blocks beat compressed K=3 outright |
| First-successor variance-scaled Hadamard (`1.5x`) | Best clean sweep; broader correction was unstable |
| Strassen L3 + block reinvestment | Cheaper matmuls, spent on blocks |
| fp32 propagation | `+0.000348%` MSE, bit-identical decisions |

The cumulant era lives in [`00-warmup-round.md`](00-warmup-round.md) (256x8) and
the transition in [`02-winning-checkpoints.md`](02-winning-checkpoints.md).

## Three results worth knowing

**There is no bias floor.** Across 4,096 → 32,768 samples on 97 paired MLPs,
`MSE ∝ n^-1.039` (bootstrap 90% `[0.840, 1.240]`), with the block-independent
component bounded below `1.5e-7` and consistent with zero. The error is pure
sampling variance. Details and data in
[`analysis/block_ladder/`](../analysis/block_ladder/) and
[`05-block-scaling-and-leaderboard.md`](05-block-scaling-and-leaderboard.md).

**Block count barely moves the score.** `MSE x effective_compute` holds at
`7.5e4 FLOP` to within ~10% across the ladder, so above the multiplier floor the
adjusted score is close to invariant in compute. Sixteen blocks and sixty-four
blocks score `2.75e-7` and `2.74e-7` at a 3.7x compute difference. Every block
sweep in this archive was therefore structurally unable to change the outcome —
which took a fourth ladder point to notice.

**The archive contains no revivable candidate.** A re-audit of every gate under
the corrected scoring model found no archived candidate with a promotion-grade
lower variance constant or a measured exponent above 1. The best was an
IID-sphere recolor control at an optimistic `V* = 7.258e4` against the route's
`7.5e4` — a 3.2% edge, unmetered, with a bank-contaminated bias figure, and at
full budget it tolerates only `b^2 < 8.88e-9`.

## What is closed

[`03-rejected-and-guarded-ideas.md`](03-rejected-and-guarded-ideas.md) indexes
69 lanes. The families, with the number that closed each:

- **Anchored control variates** — ceiling at ~0.5% of final-error variance,
  against the ~40% a competitive mechanism needs.
- **Angular / likelihood-ratio importance sampling** — for a fixed set of
  proposal evaluations, any universally unbiased matrix-weight estimator
  Rao-Blackwellizes to scalar balance weights on the same integrand, so
  output-specific proposals cannot beat the scalar optimum `q ∝ ||g||`.
- **Analytic cumulant ladders** — bias-floor ~`2.4e-5` even extrapolating
  joint-k4; zero sampling variance does not help when `b^2 * C` is orders above
  `7.5e4`.
- **Terminal readouts** — two-Gaussian mixture `0.955x`, one-Gaussian `0.712x`,
  Gaussian/Edgeworth bias-dominated, Haar-sphere fold-CV failed four of five
  gates.
- **Odd-state Rao-Blackwell** — variance `0.564x` baseline, but `b^2 = 2.56e-4`,
  roughly 2,000x its allowance.
- **QMC / LHS closures** ([`04-qmc-lhs-closures.md`](04-qmc-lhs-closures.md)) —
  every successful gate ran at a single sample count, so none can support a
  scaling-exponent argument either way.

## What was never resolved

- The gap to the strongest honest entries, measured at ~5x in variance per FLOP
  at matched compute. No mechanism in the searched families accounts for it.
- Whether `alpha ≈ 1` holds for the methods at the top of the leaderboard. The
  exponent here is measured for one family only.
- Anything grader-side. Every measurement is on the public release dataset;
  submissions closed before a grader-side ladder was possible.

## Reading the numbers

Five label-level errors were found in this archive, in a corpus where the raw
rows were correct throughout. Treat prose figures as unverified until
re-derived.

- Two **wrong-arm attributions**: a bias proxy belonging to a rejected mixture
  readout, and another belonging to a rejected Haar-CV variant, both circulating
  as if they described the default route.
- Three **decomposition bugs**: aggregators labelling `M3` (the MSE of the
  replicate mean) as `bias^2`. Correct form, with `M1` the mean single-replicate
  MSE: `sigma^2 = (3/2)(M1 - M3)` and `b^2 = (3/2)M3 - (1/2)M1`. This inverted
  at least one gate decision.

Two further standing cautions:

- **The truth bank is not for MSE.** `analysis/truth_bank/` bakes truth at
  ~`1.64e6` samples, a `~1.1e-7` truth-side floor — the same order as the
  quantities it would be used to measure. Three gates give the same route
  bias values spanning 9x. Use the `N=1e9` public dataset for anything
  decision-grade.
- **Two-point extrapolations do not identify a floor.** An early `F + bias =
  3.09e-7 ± 2.05e-7` was a 1.5-sigma two-point fit that hardened into a
  constant before being retracted. Three points fitted to three parameters gave
  floors of `1.1e-7`, `4.4e-7` and `6.0e-7` depending on functional form — three
  different verdicts. It took a fourth point to identify the model.

## Files

| File | Contents |
|---|---|
| [`00-warmup-round.md`](00-warmup-round.md) | Recovered 256x8 history: K=2, factorized K=3, factor groups, rank compression |
| [`01-current-estimator.md`](01-current-estimator.md) | The route as it stands |
| [`02-winning-checkpoints.md`](02-winning-checkpoints.md) | Every promotion, with the comparison that justified it |
| [`03-rejected-and-guarded-ideas.md`](03-rejected-and-guarded-ideas.md) | 69 closed lanes, indexed |
| [`04-qmc-lhs-closures.md`](04-qmc-lhs-closures.md) | Sobol, lattice, LHS, folded ZCA |
| [`05-block-scaling-and-leaderboard.md`](05-block-scaling-and-leaderboard.md) | The ladder, the re-audit, the corrections |
| [`06-benchmarking-notes.md`](06-benchmarking-notes.md) | How comparisons were run and what counts as noise |
