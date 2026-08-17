# Positive-homogeneity angular-importance gate (pre-registration)

Status before launch: **PRE-REGISTERED; no result has been observed.**

## Question

Can a legal, label-free importance proposal on input angle reduce the trace
variance of the final-layer antithetic mean enough to explain the missing
variance-per-FLOP factor, before attempting any change to `estimator.py`?

This is a research measurement, not an estimator evaluation. It uses the 100
research MLP seeds in the Fly truth bank only to select/rebuild one MLP per
shard. Ground-truth means are never loaded, fitted, or scored. All particle
generation and propagation runs machine-side through `make fly-payload`;
aggregation is local.

## Exact identity and proposal

For every bias-free ReLU layer output, positive homogeneity gives

`f_l(r u) = r f_l(u)`, for `r >= 0`.

If `X = R U ~ N(0, I_d)`, with `R ~ chi_d` independent of uniform-sphere
`U`, and `m_d = E[R]`, then

`E[f_l(X)] = m_d E_U[g_l(U)]`,

where `g_l(U) = (f_l(U) + f_l(-U))/2`. The fitted nonnegative folded
surrogate is

`s_a(x) = a_0 ||x|| + sum_i a_i |w_i^T x|`, with all `a_i >= 0`,

where `w_i` is column `i` of the first weight matrix. Its exact Gaussian
normalizer is

`Z_a = a_0 m_d + sqrt(2/pi) sum_i a_i ||w_i||`.

Equivalently, on the sphere, with

`c_d = E|U_1| = sqrt(2/pi)/m_d`,

the proposal likelihood ratio is

`r_a(U) = q(U)/p(U) = s_a(U)/A_a`,

`A_a = a_0 + c_d sum_i a_i ||w_i|| = Z_a/m_d`.

The proposal is exactly samplable as a mixture. The `a_0` component is
uniform on the sphere. Component `i` has angular density proportional to
`|e_i^T U|`, `e_i = w_i/||w_i||`: draw a fair sign,
`Y = (e_i^T U)^2 ~ Beta(1, (d-1)/2)`, draw `V` uniformly on the orthogonal
`S^(d-2)`, and return

`U = sign sqrt(Y) e_i + sqrt(1-Y) V`.

Every component has the same radius-tilted chi law in the full Gaussian
proposal. Homogeneity cancels that radius exactly, so for any fixed `r_0 > 0`
the unbiased antithetic contribution is

`Z_a g_l(r_0 U) / s_a(r_0 U) = m_d g_l(U) / r_a(U)`.

For `h(U) = m_d g_31(U)`, the exact proposal trace-second-moment identity is

`E_q ||h(U)/r_a(U)||^2 = E_p ||h(U)||^2 / r_a(U)`.

The oracle angular proposal `q*(U) proportional ||h(U)||` has second moment
`(E_p ||h(U)||)^2`; it is reported only as an unattainable ceiling.

## Fixed design

- Population: all 100 width-256/depth-32 research truth-bank MLP seeds, one
  Fly payload shard per MLP, with stored weight checksums verified. Truth
  arrays are not accessed.
- Randomness: a new gate-specific RNG stream derived from the research MLP
  seed, independent of truth-bank generation and prior gates.
- Particle law: i.i.d. uniform-sphere directions evaluated in legal
  antithetic pairs `(U, -U)` at fixed `r_0 = sqrt(256)`. No MLP seed, public
  grader seed, or evaluation instance is special-cased.
- Pilot: 512 pairs, exactly `alpha = 1/8` of the current 4096-pair/8192-row
  budget. Fit `s_a` to the pilot terminal folded norm `||g_31(r_0 U)||_2` by
  nonnegative projected ridge least squares. Feature and target scales are
  normalized, ridge is fixed at `1e-3` on slope coefficients (not the
  intercept), and the optimizer uses 400 deterministic projected-gradient
  iterations.
- Primary proposal: mix the fitted normalized proposal with 10% uniform
  angular mass. This remains exactly in the `s_a` family by increasing the
  `a_0 ||x||` coefficient, and guarantees `r(U) >= 0.1` and importance
  weight `1/r(U) <= 10`. The unfloored fitted proposal is diagnostic only.
- Cross-fitted measurement: 2048 new uniform antithetic pairs, independent of
  the pilot, estimate baseline, fitted-proposal, and oracle trace variances by
  the exact likelihood-ratio identity. The squared mean norm is estimated by
  a cross-product of the two held-out halves.
- Direct validation: 1024 additional pairs are sampled directly from the
  primary mixture. They validate mixture normalization, unbiasedness at Monte
  Carlo precision, and agreement between direct and identity-based proposal
  second moments/variances.
- No first-layer covariance recolor or first-successor variance correction is
  applied in this gate. Those are heuristic particle transforms, not the
  original homogeneous integrand, so exact likelihood-ratio identities do not
  pass through them automatically. Any passing result therefore requires a
  separate implementation design for their interaction.

For baseline trace variance `V_p` and proposal trace variance `V_q`, define
`R = V_p/V_q`. Combining the independent pilot baseline estimate and main
proposal estimate with fixed row-count weights has projected total variance
fraction

`F_total = alpha + (1-alpha)/R`, with `alpha = 1/8`.

## Pre-registered decision rule

The primary 10%-uniform proposal passes only if all of the following hold:

1. proposal-only pooled `R >= 2.1`, median per-MLP `R >= 1.8`, and per-MLP
   q10 `R >= 1.05`;
2. pooled projected `F_total <= 0.58`, sufficient to move a fixed-compute
   final MSE to at most `0.58x` if the mechanism transfers;
3. no severe importance tail: direct-sample maximum weight is at most
   `10.000001`, median per-MLP importance-weight ESS fraction is at least
   `0.50`, and q10 ESS fraction is at least `0.25`;
4. direct validation does not contradict the identity: pooled direct versus
   identity proposal second moment and variance agree within 20%, and the
   pooled normalized mean-difference statistic is in `[0.5, 1.5]` (the
   statistic has expectation near one under unbiased independent means).

The oracle ceiling, raw fitted proposal, surrogate correlations, mixture
masses, likelihood-ratio normalization, and per-MLP tails are descriptive and
cannot rescue a failed primary gate. If the gate passes, report a precise
mode-gated implementation plan, including the recolor interaction, but do not
edit `estimator.py` until reassigned. If it fails, document and close the
lane.

## Result (100/100 Fly shards): FAIL; close

The preregistered `make fly-payload` run returned all 100 shards with no
failures and verified all 100 stored MLP weight checksums. Each MLP used 512
pilot pairs, 2048 independent held-out uniform pairs, and 1024 direct proposal
pairs. The complete machine rows and local aggregation are in
`angular_importance_gate_20260709_fly.jsonl` and
`angular_importance_gate_20260709_results.json`.

Primary 10%-uniform fitted proposal:

- pooled proposal-only variance ratio: `1.14483x` (required `>=2.1x`);
- per-MLP ratio median `1.11882x`, q10 `1.04196x`, q90 `1.23925x`
  (required median `>=1.8x`, q10 `>=1.05x`);
- pilot-inclusive projected total variance fraction: `0.88931x`, or only
  `1.12447x` total gain (required `<=0.58x` fraction).

The proposal mathematics and implementation validated cleanly:

- held-out likelihood-ratio mean across MLPs was `0.999934` (median
  `0.999983`);
- direct/identity pooled second-moment ratio was `0.999918`;
- direct/identity pooled variance ratio was `0.992466`;
- the pooled normalized independent-mean difference statistic was `0.999164`,
  essentially its unbiased expectation of one;
- importance-weight ESS fraction was median `0.996874`, q10 `0.996551`, and
  the largest observed weight was only `1.28767`, comfortably inside the
  hard bound of 10.

The unattainable oracle `q*(U) proportional ||g_31(U)||` confirms that angular
importance is not intrinsically powerless: it reached `2.42675x` pooled, with
per-MLP median `2.20160x`, q10 `1.73188x`, and a pilot-inclusive projected
fraction `0.48556x`. The specific legal first-layer folded surrogate cannot
capture that opportunity. Despite pilot train R2 median `0.5030` and held-out
terminal-norm Pearson correlation median `0.4690`, its likelihood ratios are
too concentrated near one to materially approach the oracle. Removing the
10% safety mixture did not help: the raw fitted proposal reached only
`1.14089x` pooled.

This gate is deliberately more favorable to the importance mechanism than a
drop-in comparison with the current estimator. It measures exact i.i.d.
angular antithetic integration of the original homogeneous MLP, while the live
route uses orthogonal Hadamard blocks, exact first-layer covariance recolor,
and first-successor variance matching. Those heuristic sample transforms do
not automatically preserve the likelihood-ratio identity, and the projected
`0.88931x` also excludes fit/sampling overhead. Therefore a mere transfer to
the current route cannot plausibly turn this measured `1.145x` effect into the
required `>=2.1x` proposal gain. Verdict: **performance FAIL, tail PASS,
direct-validation PASS, overall FAIL**. Do not implement an estimator mode;
close this folded first-layer angular-importance lane unless a new surrogate
can approximate the oracle norm with substantially greater amplitude while
remaining exactly normalized and cheaply samplable.
