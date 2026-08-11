---
name: inspect-estimator-rules
description: Audit ARC White-Box Estimation Challenge estimator code, especially estimator.py, for rule compliance. Use when asked whether an estimator follows the rules, cheats, uses illegitimate inputs, relies on hidden/private grader data, bypasses flopscope or budgets, improperly reads files or network resources, or special-cases public/private evaluation instances.
---

# Inspect Estimator Rules

## Workflow

Audit from the repository rules and local documentation, not from intuition alone.

1. Read `AGENTS.md` and `ESTIMATOR_HISTORY.md` before judging estimator behavior.
2. Inspect `estimator.py` with `rg` and targeted reads. Search for:
   - file and path access: `open`, `Path`, `fnp.load`, `pickle`, `json`, `submission_dir`, `scratch_dir`
   - network/process access: `requests`, `urllib`, `socket`, `subprocess`
   - randomness and seeds: `mlp.seed`, `ctx.seed`, `default_rng`, `RandomState`, fixed numeric seeds, `time`, `os.urandom`
   - external controls: `os.environ`, CLI parsing, mode routing
   - forbidden selectors: branches or parameter choices based on `mlp.width`, `mlp.depth`, `len(mlp.weights)`, or `budget`
   - hidden-target clues: `monte`, `ground`, `reference`, `private`, `public`, `leaderboard`, `test`, `seed`
   - accounting risks: `flopscope`, `BudgetContext`, monkeypatching, disabling, wrappers around FLOP-counted operations
3. Read the estimator contract and relevant repo docs before flagging seed or setup behavior:
   - `docs/reference/code-patterns.md`
   - `docs/reference/estimator-contract.md`
   - `docs/how-to/pre-submission-checklist.md`
4. Separate hard violations from audit caveats. Do not label allowed patterns as violations just because they look unusual.

## Rule Interpretation

Treat these as hard violations or strong findings:

- Reading or deriving ground-truth Monte Carlo samples, reference outputs, private tests, grader internals, private seeds, or hidden evaluation data.
- Modifying, monkeypatching, disabling, or evading `flopscope` accounting or budget enforcement.
- Network access from `setup()` or `predict()`.
- Filesystem reads outside shipped submission files via `SetupContext.submission_dir` and permitted cache/scratch locations, unless clearly local-only under `if __name__ == "__main__"`.
- Special-casing identifiable evaluation instances: public MLPs, public seeds, leaderboard cases, private split identifiers, exact weight fingerprints, or hardcoded outputs.
- Fitting, tuning, or branching on known public/private evaluation cases rather than general MLP properties.
- Routing or choosing estimator parameters from MLP shape or the passed budget. The grader shape and budget are fixed; `mlp.width`, `mlp.depth`, `len(mlp.weights)`, and `budget` may be used mechanically for array dimensions and traversal, but must not select an algorithm, mode, block count, sample count, or other tuning parameter.

Treat these as usually allowed, unless they encode case-specific behavior:

- `fnp.random.default_rng(mlp.seed)` inside `predict()`. The repo docs explicitly recommend `mlp.seed` for per-MLP estimator randomness.
- `fnp.random.default_rng(ctx.seed)` inside `setup()`. The repo docs recommend `ctx.seed` for setup-time randomness.
- Environment variables used as general experiment/config knobs. They become suspicious only if they encode hidden data, public/private case choices, or per-instance behavior during evaluation.
- Local comparison helpers, imports of `local_engine`, baseline loaders, or Monte Carlo diagnostics guarded by `if __name__ == "__main__"` and not used by grader-time import or `predict()`.

## Reporting

Use a code-review posture:

- Lead with findings ordered by severity, with file and line references.
- Say clearly when no violation is found.
- Distinguish "violation", "compliance risk", and "looks allowed".
- If the evidence depends on how env vars or run scripts are used, say that the file itself is not enough to prove a violation.
- Avoid overclaiming. If local docs explicitly authorize a pattern, cite that and do not flag it as a rule break.

## Common Pitfalls

- Do not claim `mlp.seed` is banned. In this repo it is the intended seed for per-MLP randomness.
- Do not claim all env vars are banned. Judge what the variable controls and whether it is case-specific or hidden-data-derived.
- Do not confuse structural shape use with shape routing. Allocating arrays at `mlp.width` or iterating through `mlp.weights` is necessary; changing the estimator route or its tuned parameters from shape or budget is not allowed.
- Do not treat local benchmarking code as grader behavior when it is under `if __name__ == "__main__"`.
- Do not infer cheating from score-oriented engineering notes alone; look for actual prohibited inputs, state, or branches.
