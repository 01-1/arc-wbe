"""Random-baseline estimator.

Demonstrates the canonical contract surface (``setup`` / ``predict`` /
``teardown``) *and* the whestbench RNG-seeding contract:

* ``SETUP_SEED`` / ``self._init_rng`` -- hard-coded submission-level seed.
  Used for random precompute that should be deterministic across MLPs and
  across regrades (here: nothing -- this baseline has no setup-time
  precompute, but the scaffold is present so every bundled example
  demonstrates the pattern).
* ``rng = fnp.random.default_rng(mlp.seed)`` inside ``predict`` -- per-MLP
  RNG seeded from the grader-supplied ``mlp.seed``. This is the seed
  whose determinism the grader checks under regrade. Submissions that
  use their own per-MLP seeds (or unseeded randomness) may be
  disqualified -- see
  ``docs/reference/estimator-contract.md``
  ("Reproducibility under the grader seed") for the contract.
"""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


class Estimator(BaseEstimator):
    SETUP_SEED = 0xC0FFEE  # any fixed constant; identifies this submission's setup state

    def __init__(self) -> None:
        self._context = None
        self._init_rng = fnp.random.default_rng(self.SETUP_SEED)

    def setup(self, context: SetupContext) -> None:
        self._context = context

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)
        return fnp.asarray(
            rng.uniform(0.0, 1.0, size=(mlp.depth, mlp.width)).astype(fnp.float32)
        )

    def teardown(self) -> None:
        self._context = None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=32, depth=6, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)
