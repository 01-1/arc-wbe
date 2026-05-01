# Troubleshooting — When something breaks

> [← Documentation](../index.md)

If your run errored, your score regressed, or your local and remote scores disagree, start here.

| Doc | When to read |
|---|---|
| [common-participant-errors.md](common-participant-errors.md) | Symptom → cause → fix-now → verify, for the most common failures (wrong shape, NaN/Inf, exceeded budget, signature mismatches, import errors, numeric blow-up in deep networks). |
| [faq.md](faq.md) | Quick answers — can I use scipy? what is `untracked_time_limit`? why does my submission score worse than my local run? |

## Related

- [How-to: debugging checklist](../how-to/debugging-checklist.md) — the tiered procedure for "estimator runs but something feels wrong" (Tier 0 → Tier 3).
- [Reference: `whest doctor`](../reference/whest-doctor.md) — environment / install diagnostics. Reach for it when you suspect drift between machines.
- [Reference: score report fields](../reference/score-report-fields.md) — interpret the `per_mlp[i].error`, `budget_exhausted`, `time_exhausted` fields the harness emits.
