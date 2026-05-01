# Write an Estimator

> [← Documentation](../README.md)

## When to use this page

Use this page when implementing your custom participant estimator.

## Do this now

Start from [`examples/01_random.py`](../../examples/01_random.py), then replace the prediction logic.

Minimal structure:

```python
from __future__ import annotations

import flopscope.numpy as fnp

from whestbench import BaseEstimator, MLP


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))
```

## Expected outcome

Your estimator implements `predict(mlp, budget)` and returns a `(depth, width)` array of predicted neuron values.

## MLP traversal starter

If you need exact `MLP` field semantics or weight matrices, use:

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)

## Contract checklist

- return a `(mlp.depth, mlp.width)` array,
- all values must be finite.

## Common first failure

Symptom: estimator returns wrong shape.

Fix: ensure `predict` returns a 2D array with shape `(mlp.depth, mlp.width)` and all finite values.

---

## Building your first estimator

### Step 1: The zeros baseline

The template estimator returns all zeros. Run it to see what a bad score looks like:

```bash
uv run whest run --estimator estimator.py --n-mlps 3
```

Look at `primary_score` — this is the MSE of predicting all zeros. It is your floor.

### Step 2: Mean propagation

Copy the mean propagation example — it uses the ReLU expectation formula:

```bash
cp examples/02_mean_propagation.py estimator.py
uv run whest run --estimator estimator.py --n-mlps 3
```

Compare `primary_score` to the zeros baseline. Mean propagation uses the network's weights to make informed predictions, so it should score significantly better.

### Step 3: Understand the score report

The report shows per-MLP results:
- `final_mse`: your accuracy on the final layer (primary ranking metric)
- `flops_used`: how many FLOPs your estimator consumed
- `budget_exhausted`: whether you exceeded the budget (predictions zeroed if true)

### Step 4: Try the combined estimator

The combined estimator routes between cheap and expensive algorithms based on budget:

```bash
cp examples/04_combined.py estimator.py
uv run whest run --estimator estimator.py --n-mlps 3
```

This demonstrates the budget-aware routing pattern — a common design for production estimators.

---

## Recommended learning path

1. [`examples/01_random.py`](../../examples/01_random.py) — the interface
2. [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) — simplest real algorithm
3. [`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py) — more accurate, more expensive
4. [`examples/04_combined.py`](../../examples/04_combined.py) — budget-aware routing
5. [`estimator.py`](../../estimator.py) — the repo-root template, runnable two ways: `uv run python estimator.py` for the pure-local pedagogical loop (see [Stage 1](../getting-started/stage-1-standalone.md)) and `uv run whest run --estimator estimator.py` for the harness path. Copy when you want a minimal iteration loop.
6. [Algorithm Ideas](./algorithm-ideas.md) — full survey of strategies
7. [Performance Tips](./performance-tips.md) — FLOP optimization patterns

## Next step

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)
- [Algorithm Ideas](./algorithm-ideas.md)
- [Manage FLOP Budget](./manage-flop-budget.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
