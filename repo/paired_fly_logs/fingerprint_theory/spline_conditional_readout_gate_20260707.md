# Spline Conditional Readout Gate, 2026-07-07

Purpose: test the follow-up left open by the collapse gate: replace hard-cell
latent conditional-Gaussian readout with cross-fitted smooth local
polynomial/spline conditional readouts before touching estimator behavior.

Rules boundary: candidate estimator behavior may use only the passed MLP plus
MLP-independent randomness. This gate uses truth-bank final means only after
the label-free readout rule is fixed, to measure final MSE. The conditional
models are trained only from sampled latent/final-activation rows generated
from the MLP; truth means are never used to fit, select rows, or construct
features.

Pre-registered design:

- Run one Fly payload shard per truth-bank MLP.
- For each MLP, generate three independent legal current-route
  `hadamard_st3_b16` ensembles, including first-layer covariance recolor and
  first-successor variance match.
- Save row activations at layers 24, 28, and 30, plus final layer activations.
- For each late layer, form MLP/sample-derived PCA latents with ranks 1, 2,
  and 4.
- Fit four-fold row-cross-fitted conditional readouts of final activations as
  functions of those latents. Families are quadratic polynomial, cubic
  polynomial, and an 8-knot piecewise-constant spline surrogate, with fixed
  ridge values `1e-3` and `1e-1`.
- Compare the averaged held-out conditional predictions to the plain equal-row
  current-route final mean at matched generated rows. Report per-MLP ratios
  after averaging the three independent replicates.

Pass/fail:

- PASS only if the best pre-registered family has mean final-MSE reduction at
  least `1.35x`, median per-MLP ratio at least `1.20x`, q10 at least `0.90x`,
  and no obvious tail blowup (`ratio < 0.80` on any MLP).
- A mode-gated estimator candidate is justified only if the above PASS is
  target-scale after considering charged compute. Otherwise `estimator.py`
  remains unchanged.

Results:

- Packaging note: the first Fly payload launch omitted `estimator.py` from the
  generic payload file list and failed uniformly with `ModuleNotFoundError`.
  This was a packaging-only failure before any gate measurement. The corrected
  launch included `estimator.py` and returned `100/100` shards with zero
  failures.
- Best family: layer 28, rank 4 PCA latent, quadratic polynomial readout,
  ridge `0.1`.
- Best mean equal-row final MSE: `5.3007998195629885e-06`.
- Best mean conditional final MSE: `5.302666041212421e-06`.
- Best mean ratio equal/conditional: `0.9996480597429807`.
- Best median per-MLP ratio: `1.0006698630637465`.
- Best q10/q90 per-MLP ratio: `0.9866317766327262` /
  `1.0091794083690275`.
- Tail blowups by the registered `ratio < 0.80` rule: `0`.

Verdict: FAIL. The smooth cross-fitted conditional readouts are neutral,
slightly worse on mean MSE, and nowhere near the `1.35x` mean / `1.20x`
median target-scale pass bar. The result matches the collapse-gate lesson:
late collapsed latents describe row variation, but modeling final activations
conditional on sampled latents does not reduce the mean-estimation variance.
No estimator mode was implemented, no default changed, and no final `make fly`
was run.
