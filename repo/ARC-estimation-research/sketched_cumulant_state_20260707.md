# Sketched Cumulant State No-Build, 2026-07-07

Scope: theory/economics pass for a fundamentally different low-covariance
higher-cumulant carrier, focused on TensorSketch/CountSketch degree-2 features,
randomized low-rank third-cumulant propagation, signed-measure particles, and
orthogonal-polynomial control variates. This pass used only repository history,
local estimator code, and label-free algebra. No estimator behavior changed and
no truth labels or Fly scoring were used.

## Executive Verdict

No mode-gated estimator candidate survives the math gate.

The central obstruction is not the storage format for the 2304 quadratic
columns. In whitened layer-2 coordinates, any exact third-cumulant transport
through a Gaussian base has a unique degree-2 Hermite projection. Its covariance
is the minimum covariance load required by that matched kappa3 tensor. The
existing full joint-K3 transport already instantiates that projection, and prior
M2/M2c measurements found the undamped load above the available layer-2
covariance, with full-column routes costing about `6.75e10` raw FLOPs and
middle-rank routes losing the diagonal/repeated cumulant mass that matters.

CountSketch/TensorSketch changes how the same degree-2 Hermite projection is
stored or multiplied. It cannot lower the minimum covariance required for the
same matched kappa3. A sketch that preserves the target adds collision variance
or recovers the same covariance load in expectation; a sketch that lowers load
does so by projecting away cumulant mass, which is exactly the killed M2b/M2c
failure mode. The split-base `ts` carrier was already the genuine algebraic
covariance change in this family, because it used independent bases to drop the
same-base cross term. Its clean middle-rank result still missed badly
(`5.286e-6` final MSE at `3.224e10` raw FLOPs for `b15/k512/ts`), so hashing
the split carrier is not a target-scale candidate either.

Therefore the pre-registered build gate fails analytically. This is a no-build
closeout, not an estimator change.

## Carrier Model

Let `g ~ N(0, I_d)` be the whitened Gaussian base at an analytic-prefix
preactivation, with `d = 256`. A quadratic cumulant carrier has the perturbative
form

```text
z = mu + L g + Q(g)
E[Q] = 0
E[g Q^T] = 0
Q_k(g) in the second Hermite chaos
```

Write the centered degree-2 Hermite basis as

```text
H_ij(g) = g_i g_j - delta_ij.
```

For a symmetric target third cumulant `T_ijk`, the first-order contribution of
`Q` is

```text
T_ijk =
  E[g_i g_j Q_k]
+ E[g_i Q_j g_k]
+ E[Q_i g_j g_k].
```

The existing implementation stores `Q` through factor columns:

```text
Q_o(g) = gamma_o * ((u . g) * (v . g) - (u . v))
```

with cyclic role assignments from a factored `_FactoredThird`. This is not just
one arbitrary parameterization. In whitened coordinates, the target tensor fixes
the Hermite-2 projection of `Q`.

For any symmetric matrix `M`,

```text
E[(g^T M g - tr(M)) Q_k] = 2 <M, B_k>
```

where `B_k` is the symmetric matrix of Hermite-2 coefficients for `Q_k`.
Matching all entries of `T` fixes these inner products for every `M`, so it
fixes `B_k` by the Riesz representation theorem. Any other square-integrable
carrier with the same third cumulant decomposes as

```text
Q'_k = B_k : H_2(g) + R_k
```

where `R_k` is orthogonal to the entire degree-2 Hermite space. Hence

```text
Cov(Q') = Cov(B : H_2(g)) + Cov(R)
```

with `Cov(R)` positive semidefinite. The full quadratic carrier is therefore
the minimum-covariance carrier for that exact kappa3 match inside a positive
Gaussian-base transport.

This is the covariance lower bound that the M2 family was running into. The
history measured full undamped same-base load around `3.67x` to `3.70x` of the
available `S2` covariance, with even low ranks non-PD before damping. The
observed `~0.516` damping constant is exactly the scale implied by
`1 / sqrt(load)`.

## TensorSketch/CountSketch Economics

There are two natural sketch constructions.

### Column CountSketch

Hash factor columns `r` into `m` buckets with signs `sigma_r`:

```text
q_b(g) = sum_{r: h(r)=b} sigma_r phi_r(g)
phi_r(g) = (u_r . g)(v_r . g) - u_r . v_r
Q_sk(g) = sum_r gamma_r sigma_r q_{h(r)}(g)
```

Equivalently, if `S` is the CountSketch matrix over columns,

```text
Q_sk = Gamma S^T S phi.
```

Averaged over sketch randomness, `S^T S` is an unbiased identity estimator, so
the target Hermite projection is recovered only in expectation over the MLP-
independent sketch. For any fixed sketch, collisions either:

- reconstruct the same target plus extra collision components, increasing
  expected covariance by the sketch reconstruction error; or
- accidentally cancel some components, which lowers covariance only by losing
  target cumulant mass for that fixed MLP.

Thus CountSketch is a multiplication/storage trick, not a new covariance
geometry. To preserve the M2c kappa3 gate (`>=0.9` diagonal and `>=0.85`
repeated/distinct correlation), the sketch dimension must be comparable to the
stable dimension of the target Hermite projection. Prior rank evidence says the
mass is spread thin: best rank 64 retained only about `18%` sampler-relevant
trace mass, and best rank 128 only about `33%`, far below the old `~70%`
gate. Hashing cannot make a spread tensor concentrated.

### TensorSketch Of Products

TensorSketch can approximate products

```text
(u . g)(v . g)
```

through convolutional sketches of `u` and `v`. This reduces the cost of forming
many product features when the downstream operation only needs inner products
between degree-2 tensors. Here, however, the estimator needs actual output
particles or an output correction `Q_o(g)` for all rows before a ReLU suffix.
The output covariance and matched cumulant are still those of the same Hermite
projection `B : H_2(g)`.

TensorSketch therefore has the same trilemma:

- exact reconstruction: same minimum covariance load as full M2;
- approximate reconstruction: lower compute but reduced kappa3 correlation;
- unbiased randomized reconstruction: same mean target over sketches plus
  additional sketch variance, with fixed-seed conditional risk.

At width 256 and depth 32, the killed M2 ladder shows that this trilemma is
already fatal. The `k512` routes were too weak or too expensive, and full
routes were artifact-prone and above the score-efficient compute floor.

## Relation To Split-Base `ts`

The split-base transport was a real algebraic change:

```text
linear = L (g + h) / sqrt(2)
Q = gamma * (u . g) * (v . h)
```

with independent bases `g` and `h`. Its covariance kernel drops the same-base
term:

```text
same-base:  (U^T U) * (V^T V) + (U^T V) * (U^T V)^T
split-base: (U^T U) * (V^T V)
```

This is the kind of change a sketch would need to beat: not just fewer columns,
but a different covariance kernel. The clean split-base results did not reach
promotion scale:

```text
k128/ts:     9.391e-6 final MSE, 2.900e10 raw FLOPs
k512/ts b15: 5.286e-6 final MSE, 3.224e10 raw FLOPs
```

Full split-base columns retained the expensive/artifact-prone footprint. A
CountSketch over split-base columns can only choose between the same split
kernel with collision noise and a lower-rank projected target. It does not
create a second covariance drop beyond the independent-base construction that
has already been measured.

## Other Carrier Ideas

### Randomized Low-Rank Cumulant Propagation

Random low-rank propagation of the third tensor is another projection of the
same Hermite coefficient matrices `B_k`. If it is used as a particle transport,
the covariance lower bound applies to the retained projection. If it is used as
an analytic state, exact ReLU propagation still needs the repeated/power slices
and factor growth audited in upstream K3. The prior low-rank gate measured the
relevant spread directly and failed: low rank retained too little trace mass
and did not improve covariance feasibility.

### Signed-Measure Particles

Signed weights can formally evade the positive-covariance residual constraint,
but only by moving cancellation into particle weights. For a signed measure,
the relevant stability cost is total variation, not covariance alone. Matching
a third cumulant whose positive carrier has load `Lambda > 1` while keeping the
ordinary second moment inside `S2` forces large positive/negative cancellations
under the absolute measure. That total-variation load would be paid after every
ReLU as cancellation-sensitive bias/variance and extra bookkeeping. Nearby
signed or Edgeworth-style final corrections have already been bias-dominated.
No concrete signed propagation rule here gives a legal, stable, lower-FLOP
state through 30 more ReLUs.

### Orthogonal Polynomial Control Variates

A polynomial-chaos control variate avoids modifying the particle covariance:
carry a known zero-mean Hermite feature and subtract a downstream coefficient.
This is attractive in principle, but the coefficient for a deep ReLU suffix is
a contraction of the cumulant tensor with a third derivative or boundary
measure of the suffix. Computing that coefficient exactly is as hard as the
analytic cumulant route; estimating it from particles collapses to the killed
H2/CV3/block-predictability/downstream-projection families. A TensorSketch of
the controls only reduces compute by projecting the same spread Hermite mass.

The antithetic Hadamard default also cancels odd input-chaos contributions
strongly; the remaining variance is not known to be dominated by a cheap
degree-3 control. Prior degree-4 alias/sketch screens and scalar H2 control
variates were negative, so a high-dimensional chaos CV would need a new legal
coefficient law before it deserved estimator edits.

## Pre-Registered Gate

Before seeing any truth labels or direct Fly scores, a sketched cumulant carrier
would need to pass one of these gates:

1. Algebraic kappa3/covariance gate:
   - at layer 2, diagonal/repeated/distinct third-cumulant correlations at
     least `0.90` / `0.85` / `0.85` against the exact factored target;
   - maximum covariance load
     `lambda_max(S2^-1/2 Cov(Q) S2^-1/2) <= 0.80` before damping, or an
     equivalent proof that the residual covariance can stay positive without
     shrinking the matched cumulant below the M2c quality bar;
   - projected raw compute for prefix plus normal suffix near the score floor,
     not the old full-column `~6.75e10` family.

2. Measurement gate, only if the algebraic gate survives:
   - truth-bank/Fly-payload or direct fixed-100 Fly evidence of at least
     `1.35x` mean final-MSE/adjusted-score reduction with safe tails, compared
     against the current default on paired rows.

The TensorSketch/CountSketch family fails Gate 1 before implementation. Exact
matching inherits the full quadratic covariance lower bound; reducing the
sketch dimension projects away target mass; unbiased sketching adds collision
variance. The split-base exception already tested the main covariance-kernel
change and remained far from target.

## Decision

- Estimator changed: no.
- New mode: no.
- `python -m py_compile estimator.py`: not run because `estimator.py` was not
  changed.
- `make fly`: not run because no estimator candidate survived the pre-scoring
  gate.
- Default changed: no.
- Final `make fly` crossed `1.6e-7`: no final Fly run was warranted.
