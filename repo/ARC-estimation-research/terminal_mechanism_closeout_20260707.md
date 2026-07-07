# Terminal Mechanism Closeout, 2026-07-07

Scope: read-only closeout of remaining terminal/final-readout mechanism space
for the current depth-32 Hadamard route. The active spline conditional readout
gate is owned by another worker and was not duplicated here. No estimator mode
was added and no Fly run was launched.

## Context Read

- `AGENTS.md` and the 2026-07-06/07 `ESTIMATOR_HISTORY.md` entries.
- Durable history records for leaderboard profile forensics, readout smoothing,
  Keenan contraction, collapse/filament gates, terminal refinement probes,
  final-PC reflection, H2/anchored CV, augmented readout shortcut, and block
  predictability.
- `ARC-estimation-research/external_sampler_writeup_scout_20260707.md`.
- `estimator.py` around the Hadamard final-layer path and posthoc tokens.

Several requested gitignored gate files were not present in this worktree:
`profile_forensics_v2_20260706.md`, `readout_smoothing_gate_20260706.md`,
`keenan_contraction_gate_20260706.md`, `collapse_gate_20260706.md`, and
`filament_stage1_20260706.md`. Their durable numerical outcomes are recorded
in `ESTIMATOR_HISTORY.md` and were used as the source of truth.

## Mechanism Inventory

- **Terminal Gaussian/readout smoothing.** The truth-bank gate directly tested
  Gaussian plug-in `E[ReLU]` from terminal `(mu, sigma)` against sample
  averaging. It failed all preregistered premises: non-Gaussian terminal
  marginals, smoothed/direct MSE ratio above one, and plug-in bias squared
  around `1.09e-6`. No smaller variant has target-scale prior.

- **Final Gaussian/Edgeworth pulls and sample-cumulant blends.** Scorer-path
  pulls bounced or lost, and the finite-sample Gram-Charlier family is already
  explained by variance in estimated higher moments. Public k4 extrapolation
  writeups rely on truth-labeled coefficient fitting, which is not a legal
  submitted-estimator mechanism here.

- **Empirical-Bayes/readout shrinkage using within-predict sample structure.**
  The legal within-predict versions already tested are final-row trimming,
  inverse-variance block weighting, H2/CV3 scalar controls, and anchored CV
  screens. The generous offline anchored-CV ceiling was only about `0.5%` of
  final-error variance, versus `~40%` needed for target-scale movement. The
  later block-predictability gate also found no label-free block observable
  strong enough for weighting/allocation.

- **Terminal reflections and mirroring.** Full final-preactivation reflection
  was catastrophically biased (`8.4e-3` class MSE). Penultimate full mirroring
  (`mirror30`) was clean but far too noisy (`6.36e-7` adjusted), and
  rank-limited final-PC reflection also lost (`3.43e-7` / `3.82e-7` adjusted
  for one/two PCs). These close the obvious collapsed-latent reflection route
  unless a future proposal proves an unbiased symmetry rather than imposing an
  empirical one.

- **Low-dimensional collapsed-latent conditional readout.** Collapse itself is
  real, but the sampled-latent conditional-Gaussian gate was `~1.0x` by
  construction and deterministic filament grids hit the exponential-nodes
  curse. The only still-live distinct conditional-readout idea is the active
  spline conditional readout gate in another worker. Duplicating it here would
  risk conflicting edits and duplicate Fly spend.

- **Final-layer effort allocation with extra independent terminal samples.**
  Placeholder hidden rows, late pruning, block reinvestment, mirror30, final-PC
  reflection, and L4/b17 economics already cover the accessible legal
  allocation variants in this codebase. Extra independent terminal samples are
  only useful if they come from a lower-variance conditional law; otherwise
  they are just block-count reinvestment, which is noise-limited and does not
  approach `<1.6e-7`.

- **Profile-forensics-derived terminal mechanism.** Public profiles still
  suggest that some competitors have strong final-layer-specific effort or a
  better sampler constant, but no implementable legal algorithm follows from
  the profiles alone. The external-source scout found no public writeup/code
  explaining a distinct terminal mechanism. Using profile shapes to tune
  coefficients or branch behavior would be public-instance forensics, not a
  legal general estimator.

## Verdict

No distinct terminal/final-readout gate remains with target-scale prior outside
the active spline conditional readout worker. The remaining terminal ideas are
either direct duplicates of killed lanes, require truth-labeled fitting, or
collapse to generic extra sampling/allocation without a new lower-variance
conditional law. Default estimator unchanged; no mode-gated candidate, syntax
check, `make fly-mode`, or final `make fly` was warranted.
