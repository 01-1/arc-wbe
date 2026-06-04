# Stage 3: Run Locally (In-Process Harness)

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · **3** · [4](stage-4-run-subprocess.md) · [5](stage-5-package.md)

Stage 2 confirms the contract. Stage 3 runs the **real scoring pipeline** (the same one the grader uses) — in-process, so you can drop `import pdb; pdb.set_trace()` anywhere in `predict()` and step through it. By default it scores a fresh random MLP suite; pass `--dataset` to score the exact public-eval MLPs behind the leaderboard.

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --runner local
```

The default runner is `local` — you can omit `--runner local`. Defaults: `--n-mlps 10`, `--flop-budget 6.8e10` (68B), `--n-samples 2,560,000` (`= 100*100*256`). The MLP shape is fixed at the competition size (width=256, depth=8); `whest run` does not take `--width`/`--depth`.

You'll see a Rich-rendered report with five panels:

1. **Run Context** — estimator class, path, timestamps, `n_mlps`, `width`, `depth`, `flop_budget`.
2. **Hardware & Runtime** — host, OS, CPU, RAM, Python and NumPy versions (so a leaderboard score is reproducible across machines).
3. **Sampling Budget Breakdown (Ground Truth)** — total FLOPs and time spent generating Monte-Carlo ground truth.
4. **Estimator Budget Breakdown** — same fields for your `predict()` call(s).
5. **Final Score** — the headline metrics:

```
╭──────────────────────── Final Score ────────────────────────╮
│ Adjusted Final-Layer Score  [adjusted_final_layer_score]     │
│    ≈ 0.077   ← primary score (what the leaderboard ranks on) │
│ Raw Final-Layer MSE         [final_layer_mse]       ≈ 0.77   │
│ All-Layers MSE              [all_layers_mse]        ≈ 0.65   │
│ ───────                                                      │
│ Best MLP   [best_mlp_adjusted_final_layer_score]    ≈ 0.049  │
│ Worst MLP  [worst_mlp_adjusted_final_layer_score]   ≈ 0.13   │
│ ───────                                                      │
│ Mean Score Multiplier     [mean_score_multiplier]   ≈ 0.10   │
│ Mean Compute Utilization  [mean_compute_utilization] ≈ 6e-6  │
│ Failed MLPs               [n_failed_mlps]          0 of 10    │
╰ per-MLP score = final_layer_mse × max(0.1, C_m / flop_budget) ╯
```

With the zeros template, the **raw** MSE rows (`final_layer_mse` ≈ 0.77, `all_layers_mse` ≈ 0.65) reflect the natural variance of the ReLU activations. But the metric you are ranked on is `adjusted_final_layer_score`: because the zeros template spends almost no compute (~6e-6 of the budget), its multiplier sits at the 0.1 floor, so the leaderboard score is about `0.77 × 0.1 ≈ 0.077`. The numbers above use `--seed 42` for reproducibility. (`adjusted_final_layer_score` is the mean across MLPs of `final_layer_mse × max(0.1, C_m / flop_budget)`; the raw `final_layer_mse` / `all_layers_mse` carry no multiplier.) See [score-report-fields.md](../reference/score-report-fields.md) for the full schema.

## FLOP-budget callout: Stage 1 vs Stage 3

Stage 1's `local_engine.compare_against_monte_carlo` runs your `predict()` under `estimator_budget=1e9`. Stage 3's `whest run` uses the grader default `flop_budget=6.8e10` — about **68× larger**. So Stage 1 is the *tighter* budget here: if your estimator fits in Stage 1, it has ample headroom at the grader budget, and budget exhaustion is unlikely to be why a Stage-1-good estimator scores differently in Stage 3.

## Why a different score than Stage 1?

Both stages use the same MLP shape (width=256, depth=8). The numbers still differ because:

- **Stage 1** scores your estimator against **one fixed MLP** (`build_mlp(width=256, depth=8, seed=0)`) and prints **raw MSE** as Monte-Carlo ground truth converges (10 → 100,000 samples).
- **Stage 3** scores a **suite of 10 random MLPs** (seeded by `--seed`, or loaded via `--dataset`), uses far more ground-truth samples (2,560,000), and reports the **budget-adjusted `adjusted_final_layer_score`** averaged across the suite — not raw MSE.

So Stage 3's headline number is averaged over more MLPs *and* scaled by the compute multiplier; expect it to differ from the single-MLP raw MSE you saw in Stage 1.

## Debugging

Because `--runner local` runs in-process, `pdb` works:

```python
def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
    import pdb; pdb.set_trace()
    ...
```

## ✅ Expected outcome

| Estimator | Typical raw `final_layer_mse` (default settings, `--seed 42`) |
|---|---|
| Zeros template | ~0.77 (the all-zeros accuracy floor) |
| `02_mean_propagation` | ~7.7e-04 |
| `03_covariance_propagation` | ~3.7e-05 |

These are the **raw** final-layer MSEs (the accuracy signal). Your leaderboard `adjusted_final_layer_score` scales each by the compute multiplier `max(0.1, C_m / flop_budget)` — and since these all use <1% of the budget, the ranked number is exactly one-tenth of the value shown (the 0.1 floor).

(Same ballpark as the Stage 1 table because the math and shape are the same
(width=256, depth=8); they differ because Stage 3 averages a suite of 10 random
MLPs while Stage 1 scores one fixed MLP.) Full benchmark methodology in
[scoring-model.md](../concepts/scoring-model.md#example-estimator-benchmarks).

## ✅ When you're ready

Move on to [Stage 4: subprocess runner](stage-4-run-subprocess.md) for grader parity.
