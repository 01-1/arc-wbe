# Scoring Model

> [← Documentation](../README.md)

## When to use this page

Use this page to understand how the leaderboard score is computed from your estimator's predictions.

## Pipeline at a glance

```
   ┌─────────────────────┐
   │  random MLP_m       │   one of M MLPs (default M=10)
   │  flop_budget        │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────────────────┐
   │  your predict(mlp_m, budget)    │   runs inside flopscope.BudgetContext
   │  (flopscope counts every op)    │
   └──────────┬──────────────────────┘
              │
              ▼
        flops_used > flop_budget ?
         /                     \
     yes /                       \ no
        ▼                         ▼
   ┌─────────────┐         ┌─────────────────────┐
   │ pred_m :=   │         │ pred_m := your      │
   │   zeros     │         │   returned array    │
   └──────┬──────┘         └──────────┬──────────┘
          │                           │
          └────────────┬──────────────┘
                       ▼
          ┌─────────────────────────────────┐
          │ truth_m  =  Monte-Carlo means   │
          │ MSE_m    =  mean((pred_m -      │
          │              truth_m)²)         │
          └────────────┬────────────────────┘
                       │
            (repeat for every MLP)
                       │
                       ▼
            ┌────────────────────────────────────┐
            │ primary_score = mean over m of     │
            │  the FINAL-LAYER MSE (length n)    │
            │                                    │
            │ secondary_score = mean over m of   │
            │  the ALL-LAYER MSE (d × n cells)   │
            └────────────────────────────────────┘

                       lower is better
```

## TL;DR

- Lower score is better.
- Score is pure MSE under a FLOP budget constraint.
- If your estimator exceeds the FLOP budget, all predictions for that MLP are zeroed.
- Your score is pure MSE — the closer to zero, the better.

## The core idea

The scoring model answers a specific question: **how accurately can your estimator predict expected neuron values within a fixed analytical compute budget?**

Each estimator call is given a `flop_budget` — a cap on the number of floating-point operations it may perform, tracked analytically by flopscope. If the estimator stays within budget, its predictions are scored by MSE against Monte Carlo ground truth. If it exceeds the budget, all predictions for that MLP are replaced with zeros.

## How scoring works

For the configured FLOP budget:

1. **Your estimator runs.** Your `predict(mlp, budget)` is called. flopscope tracks all FLOP usage analytically — no wall-clock measurement.
2. **Budget is checked.** If the total FLOPs used exceed `flop_budget`, all predictions for this MLP are replaced with zero vectors.
3. **Accuracy is measured.** Per-depth mean squared error (MSE) between your predictions and Monte Carlo ground truth is computed.
4. **Score is MSE.** Your score is pure MSE — the closer to zero, the better.

Final score is the MSE averaged across MLPs (zeroed where budget was exceeded).

## The formula

Two headline metrics. Both are means of squared errors; they differ only in
which cells they average over. **Lower is better** for each.

```
                   1   M    1   n
primary_score   = ─── ∑   ─── ∑  ( pred_m[d-1, i] − truth_m[d-1, i] )²
                   M  m=1   n  i=1
                            └──────── final-layer cells only ────────┘

                   1   M    1     d-1   n
secondary_score = ─── ∑   ─── ∑     ∑   ( pred_m[k, i] − truth_m[k, i] )²
                   M  m=1  d·n k=0   i=1
                            └────── all (depth × width) cells ───────┘

  M       = number of MLPs in the suite (default 10; --n-mlps overrides)
  d       = mlp.depth, n = mlp.width
  pred_m  = (depth, width) array your predict() returned for MLP m
  truth_m = Monte-Carlo ground-truth means for MLP m
            (replaced with zeros if your call exceeded flop_budget)
```

`primary_score` is what the leaderboard ranks on. `secondary_score` is
useful for diagnosing whether your error concentrates in the final layer
or accumulates earlier — see also `best_mlp_score` and `worst_mlp_score`
in the [score report](../reference/score-report-fields.md).

## Budget behavior

Your estimator receives a `budget` argument (the FLOP budget). You may use it to route between cheap and expensive algorithms — the combined estimator example does this. But you are not required to. Fixed-strategy estimators that always use the same approach work fine, as long as they stay within budget.

## Budget enforcement rules

flopscope enforces the FLOP budget analytically:

- **Exceeded budget.** If your estimator's total FLOPs exceed `flop_budget`, **all** predictions for that MLP are replaced with zeros. This is a hard cutoff, not per-depth.
- **Under budget.** Predictions are used as-is. There is no bonus for using fewer FLOPs than the cap — accuracy is what matters.
- **No floor or clamping.** Predictions are scored as-is — there is no minimum fraction or penalty floor.

## What a good score looks like

A score near zero means your predictions are highly accurate for those MLPs. A score well above zero means prediction error is high — either because your method is inaccurate, or because it exceeded the FLOP cap and was zeroed.

Scores below what sampling would achieve at that budget indicate your structural approach is genuinely better than brute-force Monte Carlo. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows and stays within budget.
- Use `flop_budget` to gate whether to run more expensive methods.
- Tune switching behavior using local reports across budgets.
- Compare `final_mse` and `all_layer_mse` in your reports to diagnose which depths are hurting your score.
- Use [evaluation datasets](../how-to/use-evaluation-datasets.md) to fix networks and ground truth across runs — this makes score comparisons meaningful and skips repeated sampling.

## Worked example

Suppose ground truth for a 3-neuron final layer is `[0.42, 0.38, 0.51]` and your estimator predicts `[0.40, 0.35, 0.55]`.

    final_mse = mean([(0.40 - 0.42)^2, (0.35 - 0.38)^2, (0.55 - 0.51)^2])
              = mean([0.0004, 0.0009, 0.0016])
              = (0.0004 + 0.0009 + 0.0016) / 3
              = 0.000967

That `0.000967` is this MLP's per-MLP `final_mse`. The leaderboard `primary_score` is the **mean of per-MLP `final_mse` values across all MLPs** in the evaluation — a mean of means, not a sum.

## Example estimator benchmarks

The table below shows real scores from the four bundled example estimators, run with default settings (width=100, depth=16, 10 MLPs, 100M FLOP budget). Use these as calibration points for your own estimator.

| Estimator | Final MSE | All-Layer MSE | Approach |
|-----------|-----------|---------------|----------|
| `random_estimator` | ~0.50 | ~0.48 | Returns random values — the interface walkthrough. The bundled [`estimator.py`](../../estimator.py) at the repo root is the true (all-zeros) baseline; running `uv run whest init <dir>` in a fresh directory produces the same template. |
| `mean_propagation` | ~0.004 | ~0.002 | Diagonal variance, O(depth x width^2). ~100x better than baseline. |
| `covariance_propagation` | ~0.0003 | ~0.0002 | Full covariance, O(depth x width^3). ~1000x better than baseline. |
| `combined_estimator` | ~0.0002 | ~0.0002 | Routes to covariance when budget allows, otherwise mean propagation. |

**How to read these numbers:**

- The **random estimator** (zeros) gives you the "doing nothing" baseline. Its MSE reflects the natural scale of the ground truth activations.
- **Mean propagation** is ~100x more accurate than zeros — a huge improvement from a simple analytical formula with negligible FLOP cost.
- **Covariance propagation** is another ~10x better, but costs O(width^3) per layer. At width=100, this is affordable; at width=1000, it would exhaust the budget.
- The **combined estimator** gets the best of both worlds by checking the budget before deciding which algorithm to run.

To reproduce: `uv run whest run --estimator examples/<NN>_<name>.py --n-mlps 10` (e.g. `examples/02_mean_propagation.py`)

Scores vary slightly between runs due to random MLP generation and Monte Carlo ground truth noise.

## Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
