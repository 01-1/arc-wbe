# Cross-output row-cross-fitted James–Stein gate (pre-registration)

Status: **PRE-REGISTERED BEFORE FLY**. Research-only; no estimator or history
files are modified. This gate uses unique `cross_output_rowcf_*_20260710`
artifacts and no SciPy.

## Frozen Stage-A construction

For every truth-bank MLP, generate the corrected 16-independent-block current
ensemble: exact antipodes, global first-layer ReLU mean/covariance recolor, fp32
centered application of the strength-1.5 first-successor match, and L3
propagation. Retain final preactivations `Z`. Positive blocks are contiguous
0:16 and their matching negative blocks are the second half. Fold A is
positive blocks 0:8 plus matching negatives; fold B is positive blocks 8:16
plus matching negatives. Each fold has 4096 rows and eight independent
antithetic blocks. The direct current mean is exactly `0.5*(yA+yB)`.

For each row fold, compute per-output `a`, `std`, clipped `alpha`, skew, excess,
and the analytic Gaussian ReLU mean `g`. Define
`t=(y-g)/std` and
`X=[1,alpha,alpha^2,alpha^3,skew,skew*alpha,excess,excess*alpha]`.
Cross-fit output folds `j mod 4`: standardize only the seven non-intercept
features using training outputs, fit centered ridge
`0.1*trace(Xt.T@Xt)/8`, and predict held-out outputs. No output trains itself.

Use the opposite row-fold predictor as the independent target. With
`noiseA=sample_var(ReLU(ZA))/4096` and the analogous `noiseB`, use

`lambdaA=clip((1-2/256)*mean(noiseA)/max(mean((yA-pB)^2),MIN_VARIANCE),0,1)`

and the analogous `lambdaB` toward `pA`. The candidate is
`0.5*((1-lambdaA)*yA+lambdaA*pB + (1-lambdaB)*yB+lambdaB*pA)`.

Truth is read only after current and candidate vectors are fixed. Report both
lambdas and predictor discrepancies. No feature, ridge, fold, or cap sweep is
allowed.

## Frozen gates

Stage A uses one replication on all 100 shards and passes only with valid
checksums, candidate MSE `<=1.8e-6`, global current/candidate ratio `>=1.25`,
per-MLP median `>=1.10`, q10 `>=0.85`, minimum `>=0.65`, and mean of both
lambdas in `[0.02,0.95]`. Failure stops the lane.

Only if Stage A passes, Stage B repeats the unchanged construction for three
fixed replications and requires candidate MSE `<=1.6e-6`, ratio `>=1.35`,
median `>=1.20`, q10 `>=0.90`, minimum `>=0.70`, and candidate bias proxy
`<=1e-6`. Runs use `make fly-payload` only and package `estimator.py`,
`local_engine.py`, and the truth bank explicitly.
