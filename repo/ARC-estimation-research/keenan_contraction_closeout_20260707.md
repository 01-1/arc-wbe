# Keenan Contraction Closeout, 2026-07-07

Scope: focused follow-up on Keenan's hidden-profile contraction clue and the
unexplained terminal drop. This pass was read-only over estimator behavior: no
new mode was added, no local estimator scoring was run, and no Fly scorer run
was launched because no pre-registered gate survived to implementation.

## Inputs Read

- `AGENTS.md` and the current `ESTIMATOR_HISTORY.md` entries around the
  2026-07-06/07 leaderboard forensics, contraction, collapse/filament,
  terminal-refinement, block-predictability, final-PC reflection, low-rank
  projection, H2 control-variate, and spline conditional-readout gates.
- `ARC-estimation-research/terminal_mechanism_closeout_20260707.md`.
- `estimator.py` around the current Hadamard route, terminal propagation loop,
  final-layer posthoc controls, mirror path, and mode parser.

Several older Keenan/readout/collapse gate artifacts are gitignored in this
worktree, so the durable history entries were treated as the source of truth.

## What Remains True

The Keenan gate found one real physical signal: injected mean-relevant hidden
errors contract at about `0.943` per layer, matching Keenan's public hidden
profile slope (`e^-0.0609 ~= 0.941`). That validates the contraction clue as a
description of deep MLP propagation.

The same gate did not identify an estimator mechanism. The only propagated toy
inside the slope band was plain-particle sampling, with profile correlation
`0.995` to the sampler reference. The shape-distinct state-reprojection toy
missed the slope band. Therefore the hidden profile is compatible with ordinary
sampling plus propagation contraction; it does not distinguish a legal
state-propagation estimator from the existing Hadamard sampler family.

The remaining anomaly is terminal, not hidden: Keenan's profile shows an
additional final-layer drop of roughly `25x` relative to the hidden trend. A
follow-up is only actionable if it explains that terminal discontinuity by a
label-free rule derivable from the passed MLP and the current-route particles.

## Gate Triage

No distinct gate was launched because every plausible terminal-only route
available from the current code and durable measurements reduces to a killed
family:

- A state-propagation/refinement switch that changes only the terminal layer
  without changing the hidden particle law becomes extra terminal allocation.
  Existing block-count economics, placeholder/late pruning history,
  penultimate `mirror30`, and rank-limited final-PC reflection all show that
  spending terminal work on the current particle law does not approach the
  needed `>=1.35x` mean final-MSE reduction.
- A conditional law over the collapsed penultimate state was directly tested
  in hard-cell conditional-Gaussian form and then in smooth spline/polynomial
  cross-fitted form. The smooth gate's best aggregate was neutral:
  mean ratio `0.99965x`, median `1.00067x`, q10 `0.98663x`, versus the
  preregistered `>=1.35x` / `>=1.20x` / `>=0.90x` pass bar.
- Terminal Gaussian/readout smoothing and final Gaussian/Edgeworth pulls are
  bias-dominated on the non-Gaussian collapsed terminal marginals. The readout
  gate measured layer-31 plug-in bias squared around `1.090e-6`, already a
  large fraction of the current route's final MSE.
- Reflection/mirroring variants impose empirical symmetries rather than a
  proven unbiased terminal law. Final-preactivation reflection was
  catastrophically biased, full penultimate mirroring was too noisy, and
  rank-limited final-PC reflection increased final MSE.
- Label-free block weighting/allocation observables are too weak to make the
  terminal drop. The block predictability gate found visible difficulty
  correlation but no useful variance reduction: weighting ratio mean/median
  near `1.001`/`1.000`, and paired-block reductions only `2.9%` mean.
- Downstream projection and tail-kernel gates found only small ranking signals,
  not robust target-scale correction selectors.

Given those measurements, a new Fly-bank gate would have been a resampling or
renaming of a killed lane unless it supplied a new legal conditional statistic.
No such statistic is present in the current estimator path.

## Proof-Style Conclusion

Keenan's hidden contraction clue remains real but non-discriminating. It
explains why many sampler-like hidden profiles decay with depth, not how to
obtain the terminal discontinuity. The terminal drop necessarily requires one
of three things:

1. a better shape-preserving sampler constant, already suggested by thylinao
   and ionel/mliston but not recoverable from Keenan's hidden profile alone;
2. a final-layer allocation/refinement mechanism that generates new useful
   terminal samples from a lower-variance conditional law; or
3. an external implementation detail or participant writeup not present in the
   repository.

The in-repo legal variants of (2) have been killed or shown neutral. The
contraction clue by itself supplies no label-free terminal/refinement rule and
does not justify an estimator change. Default estimator behavior is unchanged.

## Status

- Pre-registered promotion threshold considered:
  `>=1.35x` mean final-MSE reduction, median `>=1.20x`, q10 `>=0.90x`, no
  tail blowups.
- New gate run: none, because no novel legal gate survived triage.
- `estimator.py` changed: no.
- `make fly` / `make fly-mode`: not run.
- Target crossed: no evidence; no final scorer run was warranted.
