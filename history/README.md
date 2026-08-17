# Estimator history

The decision-useful history for the estimator: the current route, benchmark
checkpoints that changed direction, and rejected ideas likely to be retried.
The estimator source itself is under [`repo/estimator.py`](../repo/estimator.py).

## Contents

- **[Summary](SUMMARY.md)** — start here: the arc, the results, and which numbers to trust
- [Warmup round (256x8) — recovered](00-warmup-round.md)
- [Current Estimator](01-current-estimator.md)
- [Winning Checkpoints](02-winning-checkpoints.md)
- [Rejected Or Guarded Ideas](03-rejected-and-guarded-ideas.md)
- [2026-07-10 Gaussian QMC/LHS closures](04-qmc-lhs-closures.md)
- [2026-08-10 Block-scaling closure and leaderboard calibration](05-block-scaling-and-leaderboard.md)
- [Benchmarking Notes](06-benchmarking-notes.md)

Generated from `ESTIMATOR_HISTORY.md` by `scripts/build_history_split.py`.
Edit that file, not these.

## Path conventions

These files were written against the private working repository, so artifact
paths in the prose do not all match the published layout:

- `paired_fly_logs/fingerprint_theory/...` is [`gates/`](../gates/) there.
- `paired_fly_logs/*.log` are the raw Fly run logs, which are **not** published:
  they carry presigned object-store URLs and machine identifiers. The
  measurements taken from them are in
  [`analysis/block_ladder/ladder_per_mlp_mse.csv`](../analysis/block_ladder/ladder_per_mlp_mse.csv).
- `estimator.py`, `AGENTS.md` and the rest of the working repository are under
  [`repo/`](../repo/).
