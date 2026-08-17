# Prefix vector-GREG v3 (pre-registration)

Research-only measurement; no estimator/history changes. This v3 follows the
corrected 32-block design and does not reuse or overwrite v2 artifacts.

Each of 100 truth-bank MLPs is checksum-verified. The pool is exactly 32
independent full positive randomized Hadamard blocks with exact antipodes
(8192 pairs / 16384 rows), pool-global exact first-layer ReLU recolor, fp32
strength-1.5 first-successor match, and L3 fp32 propagation. K is frozen to
`{4,6,8}`. The first two blocks (1024 individual rows / 512 pairs) are pilot;
the main pool is 30 blocks (7680 pairs). Terminal budgets include the pilot:
K4=11, K6=9, K8=8, so selected-main counts are 2304, 1792, 1536.

For each K, pilot individual rows supply X and final activation vectors Y.
Columns are standardized on pilot only. Centered multi-output ridge is frozen
to `B=(Xc.T@Xc + lambda I)^-1 Xc.T@Yc`, with
`lambda=0.1*trace(Xc.T@Xc)/256`. Main individual predictors are the pilot
Y mean plus standardized-main X centered at pilot times B; matching positive
and negative rows are pair-folded. Primary ranking uses the centered projection
of pair predictors onto the pilot pair-final top principal direction (SVD sign
normalized). Stable-sort, contiguous near-equal strata, and one independent
uniform pair per stratum with exact stratum-size/7680 weights are frozen.

Primary main estimate is known full-main pair-predictor mean plus weighted
selected residual vector `(Y-predicted Y)`, combined with pilot/main population
weights 2/32 and 30/32. Same-count simple-random full-vector GREG and raw
ranked-Y are frozen controls. Candidate indices, predictor means, strata, and
weights are fixed before any unselected main suffix output is read. Only then
are full-main vector R2/residual fraction, top-PC share, and full-b32 ceiling
computed. Truth is read last. An independently seeded corrected b16 reference
route is paired.

Compute projection includes dense `32*K + terminal_blocks*(32-K)` plus pilot
XTX/XTY/256-RHS solve, main predictor matmul, standardization/PCA/sort/reduction;
any K above current raw `2.535e10` fails. No BudgetContext nesting, sweeps, or
post-result changes are allowed.

Stage-A PASS requires all 100 checksums, candidate MSE `<=1.8e-6`, global
current/candidate `>=1.25`, median/q10/min ratios `>=1.10/0.85/0.65`, primary
better than random-vector-GREG and raw-rank globally, projected raw
`<=2.535e10`, and full-main vector R2 median/q10 `>=0.90/0.75`.

Only a passing K may advance to a separately preregistered three-rep Stage B;
then require MSE `<=1.6e-6`, ratio `>=1.35`, median/q10/min
`>=1.20/0.90/0.70`, and bias proxy `<=1e-6`.
