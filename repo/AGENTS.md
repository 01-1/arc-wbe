# AGENTS.md

Guidance for coding agents working in this repository:

- Use `rg`/`rg --files` for search.
- Keep estimator changes focused in `estimator.py` unless the user asks for
  docs, examples, or harness changes.
- Run tests with a workspace-local uv cache:
  `UV_CACHE_DIR=/i/e/.uv-cache uv run pytest -q`.
- The grader budget is `6.8e10` FLOPs/MLP. The score multiplier is
  `max(0.1, C / 6.8e10)`, so the score-efficient target is just under
  `6.8e9` effective FLOPs.
- Before changing the root estimator, read
  [`docs/how-to/estimator-history.md`](docs/how-to/estimator-history.md). It
  records the current covariance-plus-sampling strategy, the previous
  covariance-propagation baseline, and covariance-update experiments that were
  tried and rejected.

