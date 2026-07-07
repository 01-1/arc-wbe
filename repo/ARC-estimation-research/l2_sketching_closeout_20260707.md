# L2 Sketching Closeout, 2026-07-07

Scope: read-only audit of mechanistic `L^2` / deduction-projection ideas not
already killed by the diagonal tail-coordinate gate or the low-rank full-kernel
downstream projection gate. No estimator mode was added and no Fly run was
launched.

## Context Read

- `AGENTS.md` and the latest `ESTIMATOR_HISTORY.md`.
- `ARC-estimation-research/estimator-useful-extract.md`, especially the
  deduction-projection and mechanistic `L^2` sketching sections.
- Durable history entries for the 2026-07-07 tail-aware projection proxy gate,
  low-rank full-kernel downstream projection gate, downstream-weighted H2
  control variate, block predictability gate, exact layer-2 recolor, and
  terminal mechanism closeout.
- `ARC-estimation-research/terminal_mechanism_closeout_20260707.md`.
- `estimator.py` around the Hadamard route and moment/posthoc correction
  hooks.

The requested `paired_fly_logs/fingerprint_theory/tail_projection_proxy_gate_20260707.md`
and low-rank full-kernel artifact files were not present in this worktree,
apparently because they are gitignored. Their numerical outcomes are recorded
in `ESTIMATOR_HISTORY.md` and are treated here as the durable source of truth.

## Mechanisms Audited

- **Diagonal tail-aware coordinate ranking.** Already killed. The Fly-bank
  gate computed a whole-suffix Hutchinson diagonal kernel and ranked top-32
  layer-mean correction coordinates by `e^2 diag(K_tail)`. It was essentially
  tied with local error ranking: median tail/local reduction ratio `1.009`,
  win fraction `0.555`, and only `+0.011` median Spearman over local score.
  This is below any target-scale threshold and does not justify a mode.

- **Low-rank full-kernel coordinate ranking.** Already killed as the distinct
  follow-up to the diagonal proxy. The Fly-payload gate propagated sampled
  Rademacher final probes backward through downstream ReLU-mask chains and
  ranked coordinates by `e_j^2 ||S J e_j||^2`. It again failed versus local
  error ranking: median low-rank/local reduction ratio `1.006`, win fraction
  `0.508`, and best-layer median only `1.084` versus the `1.35` trigger.
  Merely increasing sketch rank, changing probe count, or resizing the same
  statistic is not a new mechanism.

- **Low-dimensional subspace correction rather than coordinate selection.**
  A true subspace correction needs both a downstream metric and a correction
  vector in that subspace. The downstream metric side was exactly what the
  low-rank full-kernel gate supplied; the missing correction vector cannot be
  truth-labeled residuals under the challenge rules. If it is estimated from
  current-route samples, it reduces to the same sample/moment residuals already
  tested by local error ranking, H2/CV3/anchored controls, block
  predictability, and final/readout pulls. If it is an analytic moment vector,
  nearby legal variants are the first-successor variance match, exact layer-2
  recolor, K=3/hybrid joint transport, and augmented K3; those are already
  either live defaults, negative, or killed by scorer-path economics.

- **Ridge-projected mean correction under a downstream metric.** Ridge changes
  how aggressively one applies a correction vector after choosing a metric.
  It does not supply the correction direction. With truth-bank labels it could
  fit a useful shrinkage coefficient, but that would be a research measurement
  only, not a legal estimator. With label-free shrinkage, the closest tested
  families are anchored CV, downstream-weighted H2, block weighting, and
  projection ranking; measured ceilings were far below the `~1.3x` to `1.5x`
  variance-per-FLOP bar.

- **Label-free correction vector from current-route samples plus analytic
  identities.** The code already exploits the strong legal identities: exact
  first-layer ReLU mean/covariance and first-successor Gaussian marginal
  variance. Broader analytic corrections lose because they damage the useful
  finite-ensemble geometry or add biased final pulls: exact layer-2 covariance
  anchoring, downstream covariance gauges, Gaussian/Edgeworth final pulls,
  H2/CV3 controls, and block-observable weighting all failed to produce stable
  target-scale gains.

- **Proof-style deduction-projection framing.** The useful abstraction remains
  valid: deduce state, then project to the state most valuable under the
  remaining tail. But the available cheap projections for the current route
  have now been tested at the relevant boundaries: coordinate, diagonal-tail,
  low-rank full-kernel, scalar Hermite/control, block-observable, and
  terminal/collapse projections. The surviving untested ideas would need a
  new low-variance conditional law or higher-order propagated state, not just
  another `L^2` weighting of the existing sample residuals.

## Verdict

No distinct mechanistic `L^2` sketch gate remains with target-scale prior.
Subspace and ridge variants either require truth-labeled error directions, or
collapse to the killed coordinate/downstream-kernel/control-variate/block
families once restricted to legal MLP/sample-derived quantities. Default
estimator unchanged; no mode-gated candidate, `python -m py_compile`,
`make fly-mode`, or final `make fly` was warranted.
