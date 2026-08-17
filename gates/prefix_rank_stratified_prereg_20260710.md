# Prefix-ranked stratified suffix gate v2 (corrected pre-registration)

Research-only counterfactual measurement; no estimator behavior changes.

## Frozen route and candidate pool

Every one of 100 truth-bank MLPs is rebuilt and its stored weight checksum is
verified. The candidate pool has exactly 32 independent full positive
randomized Hadamard bases, exact antipodes (8192 pairs / 16384 rows), current
global exact first-layer ReLU mean/covariance recolor, fp32 centered first
successor variance match strength 1.5, and L3 fp32 propagation. Pair-folded
post-ReLU prefix activations are retained after K in `{2,4,8,12}` and
pair-folded final activations are used only after the pilot/selection boundary.

The first two blocks (512 pairs) are a terminal pilot; the main pool is 30
blocks / 7680 pairs. For each K, pilot final
pair means define `u = mean(final_pilot)/max(norm,MIN_VARIANCE)`, and pilot
response is `t=final_pilot dot u`. The 256 pair-folded prefix
activation columns are standardized on the pilot only. A centered ridge fit
uses `lambda=0.1*trace(Xt.T@Xt)/256`, then predicts main-pool scalar scores.

Only the remaining 30 blocks (7680 pairs) are eligible for selection. Main
pairs are stable-sorted by predicted score and partitioned into contiguous
near-equal strata. One pair is sampled per stratum with an independent
gate/K-specific RNG; its inclusion weight is stratum size divided by 7680.
Frozen terminal budgets include the two pilot blocks: K2=13, K4=11, K8=8,
K12=3 block-equivalents, giving selected-main counts 2816, 2304, 1536, 256.
The primary candidate is scalar-GREG: the known full main predictor mean plus
the stratum-size-weighted residual mean. Pilot/main population weights are
fixed at `2/32` and `30/32`. Diagnostics are same-count random GREG,
rank-stratified raw Y, and the counterfactual full-32-pool mean ceiling.

Prefix indices are materialized before any selected-main suffix propagation.
Unselected main suffix outputs are not read until every candidate index,
predictor mean, stratum, and inclusion weight is fixed; then they are read only
for the frozen full-main correlation and full-32-pool ceiling diagnostics.
Truth is read only after all
candidate and control vectors are fixed. A separately seeded corrected current
b16 route (16 genuine bases, exact antipodes, recolor, strength-1.5 match,
L3 propagation) is paired for comparison.

## Frozen compute and gates

The corrected 32-block ceiling rationale is that the 27-block pool has a hard
full-pool variance ceiling around `1.7e-6`, so it is superseded and must not be
launched. Projected work is `32*K + terminal_blocks*(32-K)` versus current
`16*32`, with doubled first-layer/recolor overhead accounted conservatively.
Raw-FLOP projections are anchored to current `2.535e10`; any K above that
anchor fails. No parameter sweep is allowed.

Stage A is one replication. A K passes only with 100/100 valid checksums,
candidate MSE `<=1.8e-6`, global current/candidate `>=1.25`, per-MLP median
`>=1.10`, q10 `>=0.85`, minimum `>=0.65`, primary GREG globally better than
same-count random GREG and rank-stratified raw Y, projected raw FLOPs
`<=2.535e10`, and median pilot-to-main held-out scalar Pearson correlation
`>=0.90`. If no K passes, stop.

Only if a K passes, Stage B repeats that K for three independent fixed route
streams and requires MSE `<=1.6e-6`, global ratio `>=1.35`, median `>=1.20`,
q10 `>=0.90`, minimum `>=0.70`, and three-rep squared-bias proxy `<=1e-6`.
