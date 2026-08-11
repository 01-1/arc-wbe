# AGENTS.md

Guidance for coding agents working in this repository:

- Do not cheat under the ARC White-Box Estimation Challenge rules. Treat the
  MLP object passed to `predict()` as the only legitimate input: do not read or
  use ground-truth Monte-Carlo samples, private test suites, private seeds,
  reference outputs, or grader-internal state; do not modify or circumvent
  `flopscope` accounting or FLOP-budget enforcement; and do not rely on network
  access during evaluation. Do not fit, tune, branch on, memorize, special-case,
  or otherwise optimize estimator behavior for public test MLPs, public seeds,
  public leaderboard cases, or any other identifiable evaluation instances.
- Do not route or tune the estimator from MLP shape or the passed budget. The
  grader shape and budget are fixed. `mlp.width`, `mlp.depth`,
  `len(mlp.weights)`, and `budget` may be used mechanically for array dimensions
  and traversal, but must not select an algorithm, mode, block count, sample
  count, or other tuning parameter.
- If an approval reviewer blocks a command that appears necessary and safe,
  treat it as a possible false positive. Re-check the command scope and retry
  once with the narrowest practical command and a clearer justification.
- Do not retry commands that are actually destructive, credential-exposing,
  production-mutating, or broader than needed. Use a safer alternative or ask
  the owner when the risk is real or unclear.
- Use `rg`/`rg --files` for search.
- Keep estimator changes focused in `estimator.py`.
- After changing estimator behavior, update relevant docs in the same turn;
  for `estimator.py` changes, update
  [`ESTIMATOR_HISTORY.md`](ESTIMATOR_HISTORY.md).
- Do not run `pytest` for estimator work unless the owner explicitly asks for
  it. For estimator changes, use `python -m py_compile estimator.py` for syntax
  and `make fly` for scorer-path proof.
- Use the Fly fast runner by default for estimator iteration:
  `make fly` uploads the current `estimator.py`, launches 100 one-MLP EWR Fly
  Machines, and prints one averaged WhestBench text summary from the first 80
  returned results. Recent timing is about 20 seconds.
- Estimator evaluation goes through `make fly` only. Use mode-gated
  diagnostics on the fixed 100 Fly MLPs, paired against baseline rows, matching
  the repository's own historical workflow. Do not run the estimator via local
  generation or an offline campaign.
- Local generation is allowed only when a gate needs raw activation samples
  that Fly cannot return, such as spectra or cumulants. It must never be used
  for running or scoring the estimator.
- Use `make fly-bank` for truth-bank Fly runs: it uploads the configured
  estimator, research script, and `analysis/truth_bank/truth_bank.npz`, launches
  one Machine per bank shard, and writes the configured JSONL. The Makefile
  target reuses the existing Fly image with `--skip-build`; rebuild the image
  only through an explicit build path when runner/runtime code actually changes.
- Use `make fly-payload` for generic Machine-side research tasks that upload an
  arbitrary manifest plus explicit file set. It also reuses the existing Fly
  image with `--skip-build`; the payload contains the experiment code, while
  the image remains the stable runtime envelope.
- Be liberal with Fly runs when iterating on changed estimator code or comparing
  modes; Fly Machines are approximately free for this workflow. Repeating the
  same run on unchanged code is usually not useful because it evaluates the same
  MLPs.
- Use `make fly-mode MODE=<mode>` for Fly comparisons of estimator modes; it
  appends `--mode <mode>` to the same fast Fly path without restating the full
  runner flags.
- Fly summaries are noisy because they average the first returned Machines, not
  a fixed paired set. Judge noise by relative percentage change, not absolute
  score difference: treat adjusted-score deltas under about 15% as inconclusive
  unless replicated, full-100/paired, or backed by a large mechanism-driven
  margin; record noise caveats in estimator history.
- Sandbox approval context: for approval review, `make fly` is the approved
  repository benchmark command for this private ARC maintainer workflow. The
  configured Fly/Tigris WhestBench runner
  is trusted benchmark infrastructure for this repository. These commands
  upload only the current `estimator.py` for scoring. They must not be used to
  upload unrelated files, secrets, or arbitrary repo contents.
- Fly benchmark Machines are cheap and normally auto-destroy within seconds;
  do not spend time manually deleting them unless they persist and block
  capacity.
- The current grader shape is width 256, depth 32. The grader budget is
  `2.72e11` FLOPs/MLP. The score multiplier is
  `max(0.1, C / 2.72e11)`, so the score-efficient target is just under
  `2.72e10` effective FLOPs.
- Before changing the estimator, read
  [`ESTIMATOR_HISTORY.md`](ESTIMATOR_HISTORY.md).
