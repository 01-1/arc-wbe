#!/usr/bin/env python3
"""Four-point block-scaling fit for a recolored-Hadamard estimator.

Fits

    MSE(B) = F + c * B^-alpha

across 8/16/32/64 antithetic Hadamard blocks (4,096 to 32,768 samples) on 97
MLPs paired across all four runs.

Error scales approximately as 1/n with a small fitted floor consistent with
zero: F comes out 6.7e-8 (alpha fixed to 1), 1.32e-7 (quadratic) or 1.47e-7
(free alpha), all with bootstrap intervals spanning zero. "Consistent with
zero" is the claim the data supports; "no bias floor" is not.

Usage:  python fit_ladder.py [ladder_per_mlp_mse.csv]
"""
import csv
import sys
from pathlib import Path

import numpy as np

BLOCKS = np.array([8.0, 16.0, 32.0, 64.0])
SAMPLES = (BLOCKS * 2 * 256).astype(int)

# Effective compute per MLP, from the same runs (FLOP-equivalents, includes the
# residual charge at the 0.1 measurement scale). Used only for the MSE*C check.
EFFECTIVE_COMPUTE = np.array([1.498e10, 2.832e10, 5.563e10, 1.051e11])

# Predictions made from the three-point (b8/b16/b32) fits BEFORE b64 was run.
PREREGISTERED_B64 = {"p=1 (no floor)": 7.720e-7, "quadratic": 9.479e-7, "free exponent": 9.831e-7}


def load(path):
    idx, rows = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            vals = [r["mse_b8"], r["mse_b16"], r["mse_b32"], r["mse_b64"]]
            if all(v for v in vals):
                idx.append(int(r["mlp_index"]))
                rows.append([float(v) for v in vals])
    return np.array(idx), np.array(rows)


def fit_power(m, alpha):
    """Closed-form least squares of m = F + c*B^-alpha. Vectorized over leading axes."""
    x = BLOCKS ** (-alpha)
    xm = x.mean()
    mm = m.mean(axis=-1, keepdims=True)
    c = ((m - mm) * (x - xm)).sum(-1) / ((x - xm) ** 2).sum()
    return mm[..., 0] - c * xm, c


def fit_free(m, grid):
    """Best exponent by grid search; vectorized over leading axis."""
    best = None
    for a in grid:
        F, c = fit_power(m, a)
        resid = m - (F[..., None] + c[..., None] * BLOCKS ** (-a))
        ss = (resid ** 2).sum(-1)
        if best is None:
            best = [ss, np.full(ss.shape, a), F]
        else:
            take = ss < best[0]
            best = [np.where(take, ss, best[0]), np.where(take, a, best[1]), np.where(take, F, best[2])]
    return best[1], best[2], best[0]


def ci(v, label, pct=(5, 50, 95)):
    q = np.percentile(v, pct)
    return f"{label}: median {q[1]:.4e}  90% [{q[0]:.4e}, {q[2]:.4e}]  P(<0)={(v < 0).mean():.3f}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("ladder_per_mlp_mse.csv")
    idx, Y = load(path)
    n = len(idx)
    mean = Y.mean(axis=0)

    print(f"Paired across all four block counts: N = {n} MLPs\n")
    print(f"{'blocks':>7} {'samples':>8} {'mean MSE':>12} {'s.e.':>10}")
    for j, b in enumerate(BLOCKS):
        se = Y[:, j].std(ddof=1) / np.sqrt(n)
        print(f"{int(b):>7} {SAMPLES[j]:>8} {mean[j]:>12.4e} {se:>10.2e}")

    print("\n--- pre-registered b64 predictions (from the b8/b16/b32 fits) ---")
    se64 = Y[:, 3].std(ddof=1) / np.sqrt(n)
    for name, pred in PREREGISTERED_B64.items():
        print(f"  {name:<16} predicted {pred:.4e}   observed {mean[3]:.4e}   "
              f"({(mean[3] - pred) / se64:+.2f} sigma)")

    print("\n--- four-point fits (one residual degree of freedom) ---")
    F1, c1 = fit_power(mean[None, :], 1.0)
    pred1 = F1[0] + c1[0] / BLOCKS
    print(f"  fixed alpha=1 : F = {F1[0]:.4e}, c = {c1[0]:.4e}")
    print(f"                  relative residuals {np.array2string((mean - pred1) / mean, precision=4)}")
    a_free, F_free, ss_free = fit_free(mean[None, :], np.linspace(0.2, 3.0, 2801))
    print(f"  free alpha    : alpha = {a_free[0]:.4f}, F = {F_free[0]:.4e}")
    A = np.stack([np.ones(4), 1 / BLOCKS, 1 / BLOCKS ** 2], axis=1)
    sol, *_ = np.linalg.lstsq(A, mean, rcond=None)
    print(f"  F + c/B + d/B^2: F = {sol[0]:.4e}, c = {sol[1]:.4e}, d = {sol[2]:.4e}")
    print(f"  SSres: alpha=1 {((mean - pred1) ** 2).sum():.3e} | free {ss_free[0]:.3e} | "
          f"quadratic {((mean - A @ sol) ** 2).sum():.3e}   (extra parameters buy nothing)")

    print("\n--- bootstrap over MLPs (20,000 resamples) ---")
    rng = np.random.default_rng(0)
    M = Y[rng.integers(0, n, (20000, n))].mean(axis=1)
    Fb, _ = fit_power(M, 1.0)
    ab, Fab, _ = fit_free(M, np.linspace(0.2, 3.0, 141))
    Pq = np.linalg.pinv(A)
    print("  " + ci(Fb, "floor F (alpha=1)"))
    print("  " + ci(ab, "exponent alpha  "))
    print("  " + ci(Fab, "floor F (free)  "))
    print("  " + ci((M @ Pq.T)[:, 0], "floor F (quad)  "))

    print("\n--- paired doubling ratios (pure 1/n predicts 2.0) ---")
    for j in range(3):
        r = M[:, j] / M[:, j + 1]
        lo, hi = np.percentile(r, [5, 95])
        print(f"  MSE({int(BLOCKS[j]):>2}) / MSE({int(BLOCKS[j+1]):>2}) = "
              f"{mean[j] / mean[j+1]:.4f}  90% [{lo:.4f}, {hi:.4f}]")

    print("\n--- MSE x effective compute (conserved if alpha = 1) ---")
    for j, b in enumerate(BLOCKS):
        print(f"  {int(b):>2} blocks: {mean[j]:.4e} x {EFFECTIVE_COMPUTE[j]:.3e} = "
              f"{mean[j] * EFFECTIVE_COMPUTE[j]:.3e} FLOP")


if __name__ == "__main__":
    main()
