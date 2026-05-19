# Estimator Contract

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, mlp: MLP, budget: int) -> fnp.ndarray`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### Lifecycle

```
  Estimator()           ──▶  __init__         (cheap; no I/O, no compute)
       │
       ▼
  setup(context)        ──▶  one call before any predict()
       │                     • runs OUTSIDE any BudgetContext (off-budget)
       │                     • bounded by setup_timeout_s (default ~5s)
       │                     • good for: lookup tables, config loads,
       │                                  shape-independent precompute
       ▼
  predict(mlp_1, b)     ──▶  one call per MLP
  predict(mlp_2, b)            • runs INSIDE a BudgetContext
  ...                          • bounded by --flop-budget and (optionally)
  predict(mlp_M, b)              --wall-time-limit / --residual-wall-time-limit
       │
       ▼
  teardown()            ──▶  one call after all predict() calls
                             • cleanup of resources opened in setup()
```

`setup()` and `teardown()` are entirely optional — `examples/02_*` and
`examples/04_*` skip both. Define them when you have shape-agnostic
precompute that's expensive enough to be worth doing once. See
[FAQ: Can I precompute things in setup()?](../troubleshooting/faq.md#can-i-precompute-things-in-setup)
for budget rules.

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs |
| `depth` | `int` | Number of layers per MLP |
| `flop_budget` | `int` | FLOP cap for the estimator |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching across calls (subprocess and Docker runners; otherwise typically `None`) |

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer |
| `MLP` | `depth` | Number of weight matrices (layers) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |
| `MLP` | `seed` | Per-MLP grader-supplied seed; use this to seed estimator-internal randomness for reproducibility under regrade. See [Reproducibility under the grader seed](#reproducibility-under-the-grader-seed) below. |

For traversal examples, see [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` |
| Numeric validity | Every value is finite |

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops` and `import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

## Failure semantics

The harness never crashes on a bad estimator. Every failure mode is
surfaced as report data so that one bad MLP doesn't take down the run.

| Failure | Behavior | Report field(s) surfacing it | Stage that catches it first |
|---|---|---|---|
| Wrong return shape (not `(mlp.depth, mlp.width)`) | predictions for this MLP zeroed | `per_mlp[i].error.details.{expected_shape, got_shape}` | Stage 2 (`whest validate`) |
| Wrong dtype (not a `flopscope.numpy.ndarray`) | predictions for this MLP zeroed | `per_mlp[i].error` with hint | Stage 2 |
| Non-finite values (NaN, Inf) | predictions for this MLP zeroed | `per_mlp[i].error.details.cause_hints` | Stage 2 |
| `predict()` raised an exception | predictions for this MLP zeroed; harness continues to the next MLP; CLI exits `1` and prints an "Estimator Errors" panel | `per_mlp[i].{error, error_code, traceback}`; `error_code` is the Python exception class name | Stage 3 (`whest run`) |
| Exceeded `flop_budget` | flopscope raises `BudgetExhaustedError` *before* the over-budget op runs; predictions zeroed | `per_mlp[i].budget_exhausted: true` | Stage 3 |
| Exceeded `--wall-time-limit` (`wall_time_limit_s`) | flopscope raises `TimeExhaustedError`; predictions zeroed | `per_mlp[i].time_exhausted: true` | Stage 3 (with `--wall-time-limit`) |
| Exceeded `--residual-wall-time-limit` | scoring layer (not flopscope) zeroes the predictions after `predict()` returns | `per_mlp[i].residual_wall_time_exhausted: true` | Stage 3 (with `--residual-wall-time-limit`) |

When `predict()` raises, the runner captures the exception, records the
class name in `error_code`, and forwards a formatted `traceback` (subprocess
runs forward it across the worker boundary). Use `--debug` to see
tracebacks inline; `--fail-fast` to halt at the first failure.

Predictions for the failed MLP are scored against zeros AND the per-MLP multiplier is forced to **1.0** (no compute discount), so the per-MLP `adjusted_final_layer_score_m = MSE(0, Y_m) × 1.0`. This is strictly worse than a trivial-zero submission that succeeds, which receives the 0.1 multiplier floor — a factor-of-ten cap on the discount. The suite mean stays finite either way; the `failure_breakdown` and `n_failed_mlps` aggregates surface how many MLPs hit which failure path. If you want the run to stop at the first problem rather than score-against-zeros, use `--fail-fast`.

For the structured `error.details` schema, see [score-report-fields.md](score-report-fields.md#per-mlp-fields).

## Reproducibility under the grader seed

If your estimator uses randomness — Monte Carlo sampling, randomized hashing, random projections, etc. — seed it from `mlp.seed`. The grader supplies a fixed per-MLP seed that is identical across all submissions for a given MLP, derived deterministically from the suite seed. **Submissions that use unseeded randomness or their own per-MLP seeds are NOT guaranteed to reproduce under regrade and may be disqualified for prize eligibility.**

```python
import flopscope.numpy as fnp

def predict(self, mlp, budget):
    rng = fnp.random.default_rng(mlp.seed)
    # ... use rng for any internal randomness
```

If your estimator is deterministic (no internal randomness), you can ignore `mlp.seed`.

`setup()` runs before any MLP is seen, so `mlp.seed` is not yet available there. Setup-time precompute must be deterministic — use a hard-coded constant (e.g. a class-level `SETUP_SEED`) to seed any submission-level random precompute. The resulting state is identical across all MLPs in the suite and across regrades; that is the right behavior for "precompute it once, reuse it" patterns like fixed random projections. All four bundled examples (`examples/0[1-4]_*.py`) carry the scaffold side-by-side so the pattern is visible whether you start from the random baseline or one of the deterministic propagators.

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
