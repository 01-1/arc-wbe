# Stage 1: Iterate Locally (Just `flopscope`)

You don't need the `whest` CLI to start. The `local_engine.py` in this repo gives you the same MLP factory and Monte-Carlo helpers the harness uses internally — wired up so you can iterate on `predict()` and see convergence in seconds.

## Run it

```bash
uv run python estimator.py
```

You should see a table like:

```
--- Your estimator ---
MLP: width=32 depth=6 seed=0

 n_samples | sampling_flops | estimator_flops |        MSE
----------------------------------------------------------
        10 |         65,792 |               0 |   0.605731
       100 |        656,192 |               0 |   0.520049
     1,000 |      6,560,192 |               0 |   0.466525
    10,000 |     65,600,192 |               0 |   0.464739
   100,000 |    656,000,192 |               0 |   0.465598
```

The stub `predict()` returns all zeros, so `estimator_flops` is `0` and the MSE
plateaus at the variance of the true outputs — once you put real math in
`predict()`, both columns come alive and the MSE should shrink roughly as
`1/sqrt(n_samples)` (Monte Carlo converging to your estimator's answer).

## Edit `predict()`

Open [estimator.py](../../estimator.py). The body of `predict()` returns all zeros — replace it with your idea. The template already imports `flopscope as flops` and `flopscope.numpy as fnp`, so any array op you write through `fnp` (or via Python operators on `fnp` arrays) is FLOP-counted automatically. Re-run; the MSE column tells you how close you are, and `estimator_flops` shows what your math cost.

## Compare against a baseline

```bash
uv run python estimator.py --baseline mean_propagation
```

This loads `examples/02_mean_propagation.py` and runs both estimators on the same MLP.

## When you're ready

Move on to [Stage 2: validate the contract](stage-2-validate.md).
