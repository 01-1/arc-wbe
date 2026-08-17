# Fingerprint Theory Pass, 2026-07-05

Scope: analysis only. No estimator or tracked-file edits. I read
`ESTIMATOR_HISTORY.md` in full and used only local self-generated MLPs plus
local Monte-Carlo truth. Scratch experiment code/results are in this directory:
`fingerprint_experiments.py` and `fingerprint_experiment_results.json`.

## Premise

The public cluster appears to have a scale-free variance-per-FLOP advantage:
about `1.5x` to `2x` both near the score floor and at about `47%` budget. Since
our real-grader final-layer MSE is `2.293e-6` and the independent MC floor is
about `0.31e-6`, our estimator variance component is about
`1.98e-6`. A floor-cluster total MSE of `1.25e-6` to `1.6e-6` implies an
estimator component of `0.94e-6` to `1.29e-6`, i.e. a variance factor of
`1.98 / 1.29 = 1.53` to `1.98 / 0.94 = 2.11`. The target for `1.6e-7` adjusted
is therefore roughly a `1.55x` variance reduction, not a small arithmetic or
bias correction.

## Candidate Mechanisms

### 1. Common-randomness randomized-smoothing readout with exact debiasing

Mechanism: replace hard ReLU at selected late layers or final readout with an
unbiased conditional estimator
`relu(z + sigma eps) - (E_eps relu(z + sigma eps) - relu(z))`, preferably with
shared antithetic/common noise across rows and coordinates so the kink is
sampled by a smoother family while preserving the original target in
expectation.

Why scale-free: it changes each propagated sample's per-row variance by a
constant factor. If it worked, doubling rows would halve both baseline and
smoothed variance, preserving the ratio.

Why not already excluded: the history bullet is "Heat-kernel smoothing ... lost"
for biased fixed-bandwidth smoothing at all layers or final layer. It did not
test an exactly unbiased or bias-corrected smoothing construction.

Arithmetic: exact debiasing conditional on current preactivations gives
`Var[relu(z + sigma eps) - correction | z] = Var[relu(z + sigma eps) | z] >= 0`.
Unless the changed downstream distribution reduces the much larger row-to-row
integrand variance by more than this injected conditional variance, the net
factor cannot beat 1. A small `sigma = 0.05 * std(z)` has injected noise on
only kink-near rows, so the plausible upside is at most order `5%` to `10%`,
not the required `35%` to `50%`.

Cheap test: compare baseline vs exact-unbiased late smoothing on one local MLP
over paired estimator seeds, measuring final seed variance and MSE against
local MC truth.

Cost/risk: extra random tensor draws, normal CDF/PDF, and correction arrays per
smoothed layer. Under flopscope this likely adds meaningful residual wall and
some counted elementwise work. Rule risk is low if using only MLP weights and
estimator randomness.

Screen result: dead. On MLP seed 11, `R=12`, `120k` truth samples,
`smooth_unbiased_0.05` had seed-variance ratio `1.063x` and MSE ratio `3.56x`
vs baseline. The large MSE is probably from propagation-distribution damage;
even the variance component moved the wrong way.

### 2. Late deterministic mean-map contraction / ensemble shrinkage

Mechanism: exploit the observed late-layer contraction by shrinking centered
row deviations after some depth, or replacing late layers by Gaussian marginal
mean-map updates from ensemble mean/variance. The hope is that middle-layer
errors have already mixed into a stable mean trajectory, so late row noise can
be damped without paying for more samples.

Why scale-free: a fixed late shrink or conditional-expectation suffix would
multiply row variance by a roughly fixed factor independent of row count.

Why not already excluded: related failures exist, but not this exact
conditional expectation framing. "Late-layer block pruning" removed rows and
lost. "mid-network Gaussian preactivation restart" regenerated a fresh
Gaussian ensemble and lost. "mirror<K>" forced symmetric odd moments and lost.
This candidate instead asks whether a weak late shrink can trade late row
variance for negligible bias.

Arithmetic: if late layers contributed fraction `q` of final estimator
variance and shrink factor is `a`, variance becomes
`(1 - q + q a^2)`. To get `1/1.55 = 0.645`, even eliminating all late noise
requires `q >= 0.355`. The layerwise profile says MSE peaks at layers 9-16 and
decays, final MSE `2.35e-6`; it does not show a dominant independent late
component. A moderate `a=0.85` with `q=0.4` gives only
`0.6 + 0.4*0.7225 = 0.889`, a `1.12x` factor before bias.

Cheap test: shrink centered activations by `0.90` from layer 20 or `0.85` from
layer 24 in the offline harness and compare paired final variance/MSE.

Cost/risk: cheap elementwise operations, little counted FLOP risk. High
statistical risk because ReLU mean is nonlinear in the row distribution and
history shows post-ReLU skew matters.

Screen result: dead in this naive form. `late_shrink_20_0.90` had variance
ratio `1.053x` and MSE ratio `543x`; `late_shrink_24_0.85` had variance ratio
`1.079x` and MSE ratio `241x`. The bias is catastrophic, consistent with the
mirror/restart history: late skew is not disposable.

### 3. Learned-free high-order-even cubature: orthogonal-array blocks targeting
degree-4/6 aliases

Mechanism: replace repeated independently signed Sylvester blocks with a
deterministic or randomized orthogonal-array construction whose fourth and
possibly sixth sign monomials cancel across blocks. The current error is
dominated by high-order even interactions because "noanti" moved MSE only
about `4%`; low-order mean/cov/skew refinements do little.

Why scale-free: an array that reduces the per-block fourth/sixth alias
constant reduces variance per row by a constant. More rows preserve the same
ratio.

Why not already excluded: "Alternative sample families" excludes Gaussian,
Rademacher, Halton, axis/spherical, chi-radial, column permutations, balanced
signs, split blocks, and chirp Hadamard bases. But those probes did not
guarantee cancellation of the specific high-order even monomials induced by
the weighted deep ReLU composition; chirp bases were generic, not
downstream-weighted.

Arithmetic: with current estimator component `1.98e-6`, a `1.55x` win needs
to remove `0.70e-6`. If high-order-even aliases are `>=70%` of estimator
variance, a construction cutting those aliases in half gives
`0.30 + 0.70/2 = 0.65`, exactly the target. This is the only route whose
arithmetic naturally matches the fingerprint. However, the failed chirp and
split probes are strong negative priors.

Cheap test: on local MLPs, estimate block-level error covariance and regress it
against degree-4 sign alias sketches such as
`sum_j w_next[j]^2 h_i h_j h_k h_l` sampled over random coordinate tuples.
Then compare a small library of candidate OA/sign schedules on the same MLP
with paired seeds. A useful pre-test is whether final block errors have
predictable projections on any degree-4 alias sketch; if not, do not implement.

Cost/risk: implementation can be cheap if it only changes diagonal sign
schedules; no extra matmuls. Risk is moderate: history says many sign-family
variants were neutral/losses, and constructing true width-256,
block-count-16 high-strength arrays may be awkward.

### 4. Downstream-weighted multi-level control variate from a cheap suffix

Mechanism: split each block into a small number of rows that run a cheaper
approximate suffix or coarser width projection, and use the paired difference
between full and cheap suffix as a control variate. Unlike reported-row
anchored controls, this would attack the propagated high-order-even residual
at the same random rows.

Why scale-free: with fixed allocation ratio, the control-variate correlation
and overhead are constants. Variance per FLOP is multiplied by
`(1 - rho^2) / (1 + overhead)` independent of total rows.

Why not already excluded: "Corrections" excludes reported-row Gaussian pulls,
Edgeworth, trimming, jackknife, QCV, `cv3`, and the anchored-control family
ceiling. Those use summary/reporting features. This candidate uses paired
pathwise suffix residuals, not a pooled exactified statistic. "Blending with
K=3" exhausted budget, but a much cheaper local suffix/projection CV is a
different object.

Arithmetic: the target `1.55x` at `10%` overhead requires
`(1 - rho^2) * 1.10 <= 0.645`, so `rho^2 >= 0.414`. At `20%` overhead it
needs `rho^2 >= 0.462`. This is plausible only if a cheap suffix captures at
least half of final-row random error. The anchored-CV screen's honest
`R^2=0.005` says global summaries do not; pathwise paired suffixes might.

Cheap test: for local MLPs, save baseline activations at layers 16/20/24,
propagate both full suffix and a cheap diagonal-Gaussian or low-rank projected
suffix from the same activations, then compute cross-seed `R^2` of final
errors. Reject unless paired suffix `rho^2 >= 0.45` after realistic overhead.

Cost/risk: potentially high. Any second suffix propagation spends matmuls or
heavy projections under flopscope. Low-rank projections may be flopscope-safe
but residual wall can erase wins. Rule risk low if no labels and no public
case fitting.

### 5. Online Rao-Blackwellization over last-layer gates

Mechanism: condition on penultimate activations and analytically integrate the
last ReLU's Gaussian perturbation induced by uncertainty in the final
preactivation, using an unbiased or bias-corrected estimate of that
uncertainty. More ambitiously, apply to the last 2-3 layers with a small
conditional Gaussian state.

Why scale-free: it replaces a fixed portion of rowwise Bernoulli gate noise by
conditional expectation, so the variance ratio is independent of sample count.

Why not already excluded: final-only biased heat smoothing and Gaussian
marginal pulls lost, but an exactly conditional last-gate Rao-Blackwellization
from the current row ensemble was not directly tested. It is also narrower
than "Full per-layer Gaussian marginal correction", which damaged joint
geometry.

Arithmetic: final-layer-only cannot win enough unless final gates explain
`>=35%` of estimator variance. Given the layerwise profile peaks mid-depth and
then decays, final-only likely explains much less. A 3-layer version could
matter, but then approximating joint gate geometry risks the same failure mode
as Gaussian restart. A realistic final-only `q=0.15` with perfect
Rao-Blackwellization gives only `1 / 0.85 = 1.18x`.

Cheap test: local harness final-layer replacement:
`E[relu(N(mu_j, sigma_j^2))]` with `mu/sigma` inferred from rowwise
preactivation neighborhoods or block residuals, with leave-one-block-out to
avoid self-bias. Measure MSE and seed variance.

Cost/risk: moderate elementwise cost plus reductions. Bias risk high because
final preactivation uncertainty is not independent Gaussian noise around each
row; history's final-only Gaussian pulls and heat smoothing are bad signs.

## Executed Screens

Command:

```bash
/i/e/.venv/bin/python paired_fly_logs/fingerprint_theory/fingerprint_experiments.py
```

Configuration: MLP seed `11`; `R=12` paired estimator seeds; `120,000`
antithetic MC truth samples; baseline route copied from
`paired_fly_logs/offline_screen/screen.py` without Strassen timing concerns.

| Variant | Final MSE | Seed variance | Variance ratio | MSE ratio | Bias MSE |
|---|---:|---:|---:|---:|---:|
| baseline | `3.043e-6` | `2.968e-6` | `1.000` | `1.000` | `3.229e-7` |
| exact-unbiased smoothing, layers 16+, `sigma=0.05 std` | `1.083e-5` | `3.154e-6` | `1.063` | `3.558` | `7.938e-6` |
| late shrink from layer 20, `alpha=0.90` | `1.651e-3` | `3.125e-6` | `1.053` | `542.6` | `1.648e-3` |
| late shrink from layer 24, `alpha=0.85` | `7.333e-4` | `3.201e-6` | `1.079` | `240.9` | `7.303e-4` |

These are small screens, but both tested mechanisms fail in the first
directional statistic: variance ratio is above 1 before even accounting for
bias/cost.

## Ranked Verdict

1. **Best candidate for an implementation cycle: high-order-even cubature/OA
   alias control, but only after an offline alias-correlation pre-test.** It is
   the only mechanism whose arithmetic naturally reaches `1.55x` without
   fixed-cost amortization or low-order corrections. It directly targets the
   "noanti only 4%" fingerprint. The prior is still weak because chirp,
   split-block, balanced signs, and permutations already lost.

2. **Second candidate: paired cheap-suffix control variate.** It survives the
   anchored-CV rejection because it would be pathwise, not a reported-row
   summary. It needs `rho^2 ~= 0.42-0.46` after overhead, so the first offline
   test should be correlation-only. Do not implement in `estimator.py` unless
   local pathwise suffix residuals clear that threshold.

3. **Speculative but probably too small: last-gate Rao-Blackwellization.**
   Final-only arithmetic likely caps below `1.2x`; multi-layer versions risk
   becoming the already-failed Gaussian restart/marginal-correction family.

4. **Dead: exact-unbiased randomized smoothing.** The conditional variance
   arithmetic predicts no free lunch, and the local screen gave
   `1.063x` worse seed variance plus severe MSE damage.

5. **Dead: late deterministic contraction/shrink.** The local screen produced
   catastrophic bias and worse variance, matching history's warning that
   post-ReLU skew remains essential deep into the network.

## Recommended Next Action

Run one more offline-only pre-test, not a Fly run: build degree-4/6 sign-alias
sketches for current Hadamard blocks and measure whether they explain at least
`35%` to `50%` of final block/seed error on local MLPs. If that alias
correlation is absent, the fingerprint probably requires external information
to resolve: leaderboard metadata such as row count, reported FLOPs, whether the
cluster uses sampling or an analytic/sampling hybrid, and whether their
variance scales exactly as `1/n` across budgets would reduce uncertainty most.
