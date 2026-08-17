# Cross-output empirical-Bayes final-mean gate (pre-registration)

This is a research-only final-layer measurement. It does not change or score
the estimator. No cross-output functional regression with this fixed feature,
fold, ridge, and shrink construction was found in the existing estimator
history or fingerprint-theory artifacts.

## Frozen route and data boundary

Each truth-bank shard rebuilds its MLP and verifies the stored weight checksum.
For each replication it generates 16 genuinely independent positive randomized
Hadamard bases (one full 256-row basis per block), recovers exact antipodes,
applies the current exact first-layer global ReLU mean/covariance recolor, uses
fp32 propagation with Strassen level 3, and applies the fp32 centered first
successor variance match at strength 1.5. The final preactivation matrix `Z`
and direct mean `y=mean(ReLU(Z), axis=0)` are retained.

Candidate construction is label-free. For each final neuron, compute `a`,
variance/std with `MIN_VARIANCE`, clipped standardized `alpha`, standardized
skew and excess kurtosis, and repository-helper Gaussian ReLU mean `g`.
Define `t=(y-g)/std` and `X=[1,alpha,alpha^2,alpha^3,skew,skew*alpha,excess,excess*alpha]`.
For each fold `j mod 4`, standardize only the seven non-intercept columns from
the other 192 outputs, fit centered ridge with
`lambda=0.1*trace(Xt.T@Xt)/8`, and predict the held-out 64 outputs. Set
`p=g+std*t_pred`. Estimate per-coordinate direct-mean noise as sample variance
of `ReLU(Z)` divided by row count, then set
`lambda_EB=clip(mean(noise)/max(mean((p-y)^2),MIN_VARIANCE),0,1)` and
`y_EB=(1-lambda_EB)*y+lambda_EB*p`.

Truth is read only after all candidate computations are fixed. Metrics are
exact vector MSEs against the truth-bank final mean, with checksum and failure
counts reported.

## Frozen gates

Stage A uses one replication on all 100 shards and passes only if all checksums
are valid, candidate MSE `<=1.8e-6`, global current/candidate ratio `>=1.25`,
per-MLP median ratio `>=1.10`, q10 `>=0.85`, minimum `>=0.65`, and mean
`lambda_EB` is in `[0.02,0.95]`. Failure stops the lane.

Only if Stage A passes, Stage B repeats the same construction for three fixed
independent route replications and requires candidate MSE `<=1.6e-6`, global
ratio `>=1.35`, median `>=1.20`, q10 `>=0.90`, minimum `>=0.70`, and
three-rep candidate squared-bias proxy `<=1e-6`.

No features, folds, clipping, ridge, or shrink caps are swept. No SciPy or
truth data enters candidate logic. Runs use `make fly-payload` only.
