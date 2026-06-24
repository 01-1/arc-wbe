# AGENTS.md

Guidance for coding agents working in this repository:

- Use `rg`/`rg --files` for search.
- Do not cheat under the ARC White-Box Estimation Challenge rules. Treat the
  MLP object passed to `predict()` as the only legitimate input: do not read or
  use ground-truth Monte-Carlo samples, private test suites, private seeds,
  reference outputs, or grader-internal state; do not modify or circumvent
  `flopscope` accounting or FLOP-budget enforcement; and do not rely on network
  access during evaluation. Do not fit, tune, branch on, memorize, special-case,
  or otherwise optimize estimator behavior for public test MLPs, public seeds,
  public leaderboard cases, or any other identifiable evaluation instances.
- Keep estimator changes focused in `estimator.py` unless the user asks for
  docs, examples, or harness changes.
- After changing estimator behavior, update relevant docs in the same turn;
  for `estimator.py` changes, update
  [`docs/how-to/estimator-history.md`](docs/how-to/estimator-history.md).
- Run tests with a workspace-local uv cache:
  `UV_CACHE_DIR=/i/e/.uv-cache uv run pytest -q`.
- Prefer the Makefile for WhestBench runs. Use `make mini` and its variants
  for cached public-mini scoring with the repo's residual-wall-time multiplier
  applied; run benchmarking/comparison jobs sequentially rather than in
  parallel so residual-time comparisons stay meaningful. Vary `MINI_MLPS`,
  `BUDGET`, `WALL_TIME`, and `RESIDUAL_WALL_TIME_MULTIPLIER` as needed. Use
  `make mini-mode MODE=<mode>` for forced K=3 route comparisons, and reserve
  `make mini-*-local` targets for long augmentation diagnostics that time out
  under subprocess isolation. The default residual multiplier is `2.0`,
  calibrated from observed server timing.
- The current grader shape is width 256, depth 32. The grader budget is
  `2.72e11` FLOPs/MLP. The score multiplier is
  `max(0.1, C / 2.72e11)`, so the score-efficient target is just under
  `2.72e10` effective FLOPs.
- Before changing the estimator, read
  [`docs/how-to/estimator-history.md`](docs/how-to/estimator-history.md).
