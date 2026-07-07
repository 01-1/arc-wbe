# Remaining Lane Audit, 2026-07-07

Scope: final independent read-only audit for autonomous legal estimator lanes
after the L2 sketching, Keenan contraction, large-kernel L4, paired-block CRN,
external telemetry, terminal mechanism, sampler forensics, and public-source
closeouts. No estimator behavior was changed and no Fly run was launched.

## Bottom Line

No autonomous estimator or gate lane remains with a plausible target-scale
prior. I found no legal, non-duplicate mode candidate likely to deliver the
needed `~1.5x` variance-per-FLOP improvement or cross below `1.6e-7`.

The current practical frontier is still the depth-32 Hadamard route with exact
first-layer ReLU mean/covariance recolor, first-successor variance matching,
antithetic structure, and L3 Strassen/block reinvestment. The best
grader-confirmed point remains about `2.4e-7`, and paired 16/32-block probes
show the gap is genuine estimator variance, not a hidden truth-floor artifact.
Crossing `1.6e-7` needs a new variance mechanism, not another block-count,
strength, radial, final-pull, or wall-trim knob.

## Candidate Audit

- **Shape-preserving sampler constant:** closed autonomously. Public telemetry
  still suggests a floor-group sampler edge, but public code/writeups expose
  only folded-whitening antithetic MC, which is weaker than the current route.
  Internal point-design, BQ, alias/sign schedules, split blocks, radial
  schemes, robust aggregation, block weighting, paired-block controls, and
  block-predictability/CRN screens do not show target-scale signal.
- **Terminal/refinement mechanisms:** closed autonomously. Gaussian and
  Edgeworth final pulls are bias-dominated, reflection/mirroring imposes
  unsupported symmetries, conditional Gaussian/spline collapsed-latent gates
  are neutral, and terminal effort allocation collapses to extra sampling
  without a lower-variance conditional law.
- **Mechanistic `L^2` / downstream projection:** closed autonomously.
  Diagonal tail ranking and low-rank full-kernel ranking were essentially tied
  with local error ranking. Subspace and ridge variants still need a legal
  correction vector; with only current-route sample/moment residuals they
  reduce to killed coordinate, H2/CV3, anchored-control, block, or final-pull
  families.
- **Keenan contraction clue:** not actionable. The hidden contraction physics
  matches ordinary sampler-profile decay but does not supply the roughly `25x`
  terminal discontinuity. The remaining explanations require a better sampler
  constant, a new terminal conditional law, or external participant details.
- **Large-kernel L4 / wall economics:** closed autonomously. L4's raw-FLOP
  saving is structurally tied to many narrow leaves under legal dense-array
  operations. Grader A/B showed the raw savings transfer but residual charge
  overwhelms the benefit; dense large-kernel reformulations either lose the
  Strassen cross-product saving or require unavailable sparse/batched
  primitives.
- **Higher cumulant / analytic-prefix routes:** closed for current autonomous
  action. Augmented K3 state is material but the straightforward port times
  out; exact K3 depth-32 exhausts budget; Gaussian and Edgeworth analytic
  prefixes lose too much non-Gaussian structure; low-rank joint-k3 transports
  hit covariance-feasibility and quality walls. A future route would need a
  genuinely cheaper low-covariance carrier, not another truncation of the same
  quadratic transport.

## Graveyard Coverage

The audit treats the following as covered and not worth reopening without a
new mechanism: small variance-strength knobs, lower block counts, 32-block
score-flatness probes, L4 residual trims, split rows/blocks, chirp or permuted
Hadamards, balanced signs, chi/radial/row-normalized samples, first-layer
transport variants, broader marginal/covariance corrections, exact layer-2
Gaussian recolor, H2/CV3/anchored controls, final Gaussian or Edgeworth pulls,
block trimming/median/weighting, mirror/reflection paths, analytic-prefix
Gaussian or joint-k3 transports, augmented K3 full propagation, downstream
projection ranking, conditional readout splines, and public-source sampler
clues.

## Blocker

The precise blocker is absence of an implementable legal conditional law or
sampler construction that reduces the per-block variance constant by about
`1.5x` while staying near floor compute. The remaining routes require at least
one of:

- external participant information, code, writeup, or all-layer disclosure;
- a genuinely new label-free conditional law for the collapsed terminal state;
- a new low-covariance higher-cumulant carrier that preserves non-Gaussian
  structure at depth 32 without the current compute/residual wall;
- owner-provided telemetry or idea that is not already in the documented
  graveyard.

## Status

- Estimator changed: no.
- New estimator mode: no.
- `make fly`: not run, because no estimator candidate was implemented.
- Final `make fly` crossed `1.6e-7`: no.
- Files changed: this report and `ESTIMATOR_HISTORY.md`.
