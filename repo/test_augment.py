"""Compare K=3 factorized propagation: augmented vs non-augmented."""

import time
import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp, monte_carlo_layer_means
from estimator import _factorized_k3_propagation

def run_one(mlp, augment, mc_samples=100_000):
    label = "augment=True" if augment else "augment=False"
    rng = fnp.random.default_rng(mlp.seed)
    _ = rng

    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = _factorized_k3_propagation(mlp, augment=augment)
    t1 = time.perf_counter()

    sampled = monte_carlo_layer_means(mlp, mc_samples, seed=0)
    mse = float(fnp.mean((pred - sampled) ** 2))

    return {
        "label": label,
        "flops": ctx.flops_used,
        "time_sec": t1 - t0,
        "mse": mse,
    }


def main():
    for width, depth in [(64, 4), (128, 8), (256, 8)]:
        mlp = build_mlp(width=width, depth=depth, seed=0)
        print(f"\n=== MLP width={width} depth={depth} ===")
        results = []
        for aug in [False, True]:
            r = run_one(mlp, aug)
            results.append(r)
            print(
                f"{r['label']:16s}  flops={r['flops']:>14,}  time={r['time_sec']:.4f}s  MSE={r['mse']:.8f}"
            )
        if len(results) == 2:
            flop_ratio = results[1]["flops"] / max(results[0]["flops"], 1)
            mse_ratio = results[1]["mse"] / max(results[0]["mse"], 1e-30)
            print(
                f"  ratio (aug/bare)  flops={flop_ratio:.2f}x  MSE={mse_ratio:.4f}x"
            )


if __name__ == "__main__":
    main()
