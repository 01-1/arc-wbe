# How Ground Truth Is Generated

> [← Documentation](../README.md)

This page explains how the evaluator computes the reference values your estimator is scored against.

## The process

For each MLP in the evaluation:

1. The evaluator generates random inputs from a standard normal distribution: each neuron receives an independent N(0, 1) value.
2. The inputs are propagated through the MLP (matrix multiply + ReLU at each layer).
3. This is repeated for many samples (configured by `ground_truth_samples`, typically 10,000+).
4. The per-neuron mean across all samples is the ground truth for each layer.

The evaluator uses flopscope for this computation, but under a very large FLOP budget (effectively unlimited). Ground truth computation is not constrained by the participant's FLOP budget.

## Ground truth has its own error

Because ground truth is estimated by sampling, it has finite precision. With k samples, the standard error of the mean is approximately:

    standard_error ≈ sigma / sqrt(k)

For typical networks (width=256, depth=8), 10,000 samples (the `dataset bake` default) give standard errors around 0.005–0.01. The grader's `whest run` default is much higher — 2,560,000 samples — shrinking the standard error by ~16× (to roughly 3e-4–6e-4). Either way, your MSE will never reach exactly zero because the target itself is approximate.

## What this means for your estimator

- A "perfect" estimator that exactly matches the theoretical means would still show nonzero MSE due to ground truth sampling noise.
- At the grader's 2,560,000-sample default the ground-truth noise floor is very low — the bundled covariance estimator already reaches ~4e-5 and is still above it — so a sub-`0.001` MSE does **not** mean you have hit the floor. Treat your score as noise-limited only when it stops improving across seeds.
- Different evaluation runs with different random seeds will produce slightly different ground truth values and slightly different scores.

## Configuration

The number of ground truth samples is set in the contest configuration (`ContestSpec`), which defines all evaluation parameters: width, depth, FLOP budget, number of MLPs, and ground truth sample count. You can override some of these via CLI flags (e.g., `--n-mlps`, `--flop-budget`, `--n-samples`).

- `ground_truth_samples`: number of forward passes used to estimate ground truth (default: `100 * 100 * 256 = 2,560,000`)

Higher values produce more accurate ground truth but take longer to compute. The contest organizer balances this tradeoff.

## ➡️ Next step

- [Scoring Model](./scoring-model.md)
- [Problem Setup](./problem-setup.md)
