# Active-subspace reflection covariance gate (pre-registration)

Status: **PRE-REGISTERED BEFORE FLY**. This is a research-only measurement;
it does not modify estimator behavior.

## Question

Can an MLP-derived orthogonal reflection make the already-antithetic angular
integrand negatively correlated with its reflected copy? For

`g(x) = (f(x) + f(-x)) / 2`,

the equal-cost average of `g(x)` and `g(Tx)` has trace-variance gain

`G_T = 1 / (1 + rho_T)`,

where `rho_T` is the trace covariance correlation between the two centered
vector integrands. A gain of `1.7x` requires `rho_T <= -0.4118`.

## Legitimacy and data boundary

- Run machine-side through `make fly-payload` on all 100 research truth-bank
  MLP seeds, with stored weight checksums verified.
- Candidate projectors use only the rebuilt MLP weights and fresh, unlabeled
  reference inputs. Truth means and truth-bank activation arrays are never
  read. No estimator is run or scored locally.
- No public/private grader data, seeds, reference outputs, network input to
  candidate logic, flopscope/budget bypass, or instance special-casing.
- The truth bank supplies only research MLP seeds/weights; it is not used as
  a fitting target.

## Projector families

For each MLP and each of two independent reference replications:

1. `input_gram`: eigenspace of `W0 @ W0.T`.
2. `downstream_gram`: eigenspace of the normalized linearized input-to-output
   product `W0 @ W1 @ ... @ W31` and its Gram matrix.
3. `reference_jacobian`: average Gram of exact ReLU-gated Jacobians at fresh
   spherical reference points; gates are obtained only from the MLP weights
   and those reference points.

For each family, form nested rank-1/2/4/8 projectors from the leading
eigenvectors. The reflection is `T = I - 2 Q Q.T`, which preserves spherical
measure and norm exactly in real arithmetic.

## Held-out orbit measurement

Each reference replication uses fresh held-out direction batches from two
independent laws: direct normalized Gaussian sphere points (`sphere`) and a
Haar-rotated normalized Gaussian batch (`haar`). For each direction `X`, the
payload evaluates the four-point orbit `{X, -X, TX, -TX}`, forms `g(X)` and
`g(TX)` from the antipodal pairs, and reports final-layer trace variances,
trace covariance, `rho_T`, `G_T`, and a normalized mean-difference sanity
statistic. Input batches and reference points are independent.

## Frozen decision rule

The gate passes only if a single family/rank/law satisfies all conditions in
both fresh reference replications:

1. pooled gain `G_T >= 1.7x`;
2. median per-MLP gain `>= 1.5x`;
3. q10 per-MLP gain `>= 1.0x`;
4. the same plan passes on the second fresh-reference replication, with no
   sign reversal and no pathological trace-variance denominator;
5. the result is based on all 100 MLPs with valid checksums.

Any failure closes this covariance-reflection lane without estimator code.
Descriptive oracle/reference-family comparisons cannot rescue a failed primary
gate.

