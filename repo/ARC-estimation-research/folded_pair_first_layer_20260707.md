# Folded-Pair First-Layer Transport

Date: 2026-07-07

## Rule

For an antithetic first-layer preactivation pair `+z` and `-z`, the two ReLU
rows can be written as

```text
relu(z)  = (|z| + z) / 2
relu(-z) = (|z| - z) / 2
```

The experiment preserves the odd component `z` exactly and recolors only the
folded even component `s = |z|`. For zero-mean Gaussian preactivations
`Z ~ N(0, C)`, with `sigma_i = sqrt(C_ii)` and
`rho_ij = C_ij / (sigma_i sigma_j)`,

```text
E|Z_i| = sqrt(2/pi) sigma_i
E|Z_i||Z_j| =
    sigma_i sigma_j (2/pi) (sqrt(1 - rho_ij^2) + rho_ij asin(rho_ij))
Cov(|Z|) = E|Z||Z|^T - E|Z| E|Z|^T
```

For the reconstructed antithetic ReLU rows, the even/odd cross terms cancel
pairwise, so the row covariance target is

```text
Cov(ReLU(Z)) = (Cov(|Z|) + C) / 4
```

`foldx` adds a finite-ensemble orthogonalization step before recoloring:
project the centered folded samples off the preactivation rows so that their
sample cross-covariance with `z` is zero, then recolor that residual folded
component to the same folded-normal target.

## Legality And Distinction

The transport uses only the passed MLP weights, the first-layer preactivation
ensemble already generated inside `predict()`, and closed-form Gaussian
folded-normal moment identities. It uses no labels, no public/private seeds,
no grader state, no ground-truth Monte Carlo samples, and no network during
estimator evaluation.

This is not the existing full first-layer covariance recolor: the current
default recolors individual ReLU rows after concatenating the two antithetic
halves, while this keeps the odd preactivation component fixed and transforms
only `|z|` before reconstructing the pair. It is also distinct from marginal
skew/cubic transports, support clipping, `anti<N>`/`noanti`, and radial
transport: it changes neither radial row scale nor antithetic composition and
does not apply a coordinatewise post-ReLU marginal map or nonnegativity clip.

Economically, the folded recolor works on the half-sized antithetic folded
ensemble, then reconstructs full rows, so it can save first-layer recolor and
raw propagation cost relative to the current full-row first-covariance path.
The risk is that a linear folded transport can damage higher-order dependence
between `|z|` and `z`, even when first folded moments and pairwise ReLU
covariance identities are respected.

## Fly Results

Both runs used the normal mode-gated fast Fly path, `make fly-mode`, with
80 returned EWR MLPs and no failures.

| Mode | Adjusted score | Final MSE | Raw FLOPs | Effective compute | Failures |
|---|---:|---:|---:|---:|---:|
| `hadamard_st3_b16_fold` | `3.060e-7` | `2.998e-6` | `2.460e10` | `2.707e10` | 0 |
| `hadamard_st3_b16_foldx` | `2.968e-7` | `2.798e-6` | `2.580e10` | `2.888e10` | 0 |

Verdict: killed. The raw FLOP savings are real for `fold`, and `foldx`
recovers some MSE, but neither is a mechanism-scale improvement over the
current `hadamard_st3_b16` frontier. Default behavior remains unchanged.
