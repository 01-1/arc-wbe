"""Cumulant-propagation estimator for ReLU MLPs.

Implements the K=1 (mean propagation) and K=2 (covariance propagation)
algorithms from Wu et al., "Estimating the expected output of wide random
MLPs more efficiently than sampling" (arXiv:2605.05179).

The estimator automatically selects the highest-order algorithm that fits
comfortably inside the per-MLP FLOP budget:

* Covariance propagation (K=2) – full covariance matrix, O(depth·width³) FLOPs.
* Mean propagation (K=1)       – diagonal variance only, O(depth·width²) FLOPs.

Both variants use the exact ReLU Gaussian-moment formulas (power cumulants)
described in the paper.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


def _relu_moments(mu: fnp.ndarray, var: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Return (mean, second_moment, gain) of ReLU(N(mu, var)).

    Uses the standard rectified-Gaussian formulas:

        alpha   = mu / sigma
        phi     = pdf(alpha)
        Phi     = cdf(alpha)
        E[z]    = mu * Phi + sigma * phi
        E[z²]   = (mu² + var) * Phi + mu * sigma * phi
        gain    = Phi(alpha)          # first Hermite coefficient b̃_1

    The ``gain`` is the c_i factor from the paper's Algorithm 2 (covariance
    propagation) and is used to approximate off-diagonal post-ReLU covariances.
    """
    sigma = fnp.sqrt(fnp.maximum(var, 1e-30))
    alpha = mu / sigma

    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    mean = mu * Phi + sigma * phi
    ez2 = (mu * mu + var) * Phi + mu * sigma * phi
    gain = Phi

    return mean, ez2, gain


def _mean_propagation(mlp: MLP) -> fnp.ndarray:
    """K=1 mean propagation (Algorithm 1 from the paper).

    Tracks the mean vector and a scalar average variance (trace of covariance)
    through each layer.  In practice we keep the full diagonal variance because
    it has the same asymptotic cost and strictly lower error than tracking only
    the trace.
    """
    width = mlp.width
    mu = fnp.zeros(width)
    var = fnp.ones(width)

    rows = []
    for w in mlp.weights:
        # Linear layer
        mu_pre = w.T @ mu
        var_pre = (w * w).T @ var

        # ReLU
        mu, ez2, _ = _relu_moments(mu_pre, var_pre)
        var = fnp.maximum(ez2 - mu * mu, 0.0)

        rows.append(mu)

    return fnp.stack(rows, axis=0)


def _covariance_propagation(mlp: MLP) -> fnp.ndarray:
    """K=2 covariance propagation (Algorithm 2 from the paper).

    Tracks the full mean vector and covariance matrix through each layer.
    The post-ReLU covariance is approximated with the leading-order Hermite
    term (gain method) while the diagonal is replaced by the exact marginal
    variance computed from power cumulants.
    """
    width = mlp.width
    mu = fnp.zeros(width)
    cov = fnp.eye(width)

    rows = []
    for w in mlp.weights:
        # Linear layer
        mu_pre = w.T @ mu
        # Use einsum so flopscope sees the symmetry of the two w operands.
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

        # Extract marginal variances
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-30)

        # ReLU moments
        mu_post, ez2, gain = _relu_moments(mu_pre, var_pre)
        var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)

        # Off-diagonal approximation: cov_post[i,j] ≈ gain[i]*gain[j]*cov_pre[i,j]
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)

        mu = mu_post
        rows.append(mu)

    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Adaptive cumulant-propagation estimator.

    Chooses between covariance propagation (K=2) and mean propagation (K=1)
    based on the FLOP budget.  Both are sample-free estimators from the paper.
    """

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer mean activations.

        Returns an array of shape ``(depth, width)``.
        """
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng

        width = mlp.width
        depth = mlp.depth

        # Rough FLOP estimates (empirically calibrated against the bundled
        # examples on width=256, depth=8).
        mean_flops = 3 * depth * width * width
        cov_flops = 5 * depth * width * width * width

        # Use covariance propagation if it fits inside the budget with some
        # headroom for overhead.  Otherwise fall back to mean propagation.
        if cov_flops < budget * 0.8:
            return _covariance_propagation(mlp)
        else:
            return _mean_propagation(mlp)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
