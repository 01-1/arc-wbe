# Current Estimator

The current grader shape is width 256, depth 32, with a `2.72e11` FLOP/MLP
budget and a score-efficient target just under `2.72e10` effective FLOPs.

Unforced `predict()` always uses 16 blocks of randomized antithetic
Walsh-Hadamard sign cubature. There is deliberately no shape/depth route
selector and no budget-derived block selector: the contest shape and budget
are fixed, and altered metadata in a diagnostic must not silently select a
different estimator configuration. After the first linear/ReLU layer, the
estimator linearly recolors the first hidden activation ensemble so its
mean and covariance match the exact zero-mean Gaussian ReLU moments for
`W0.T @ W0`. The first layer uses only the positive half of each antithetic
Hadamard block for the matmul, then reconstructs the negative-half ReLU
activations from the negated preactivations. It then propagates the recolored
ensemble through the remaining layers with three batched-leaf Strassen levels
for the large propagation matmuls, applying a `1.5x` variance-scale update to
only the first subsequent ReLU ensemble using its Gaussian marginal variance
target while preserving its sample mean. This route uses only the passed MLP
object and label-free moment identities; L4 remains diagnostic because the
best clean L4 measurements were either weaker than `st3_b16` or too close to
the combined-budget edge after widening the Fly collection window.

The optimized factorized K=3 cumulant route with `r=1` degree-4 harmonic
tracking, structured third-cumulant factor groups, and a diagonal-only
final-layer ReLU mean shortcut remains available only through the explicit
`r1` diagnostic mode; it is not an automatic shallow-network fallback.

**2026-07-20 fixed-route robustness correction.** Removed the default depth
selector and every automatic shape/budget-derived Hadamard block or sample
selector. Unforced prediction is now unconditional L3/16-block Hadamard, and
unnumbered Hadamard diagnostics also fall back to 16 blocks; explicit
experiment mode or environment overrides remain explicit. The fixed grader
computation is unchanged. `python -m py_compile estimator.py` passed. The
first Fly proof was contaminated by one residual-time combined-budget failure
among 80 returned rows. A confirmation returned 80 scored rows with zero
estimator/scorer failures at `2.804e-6` final-layer MSE, `2.894e-7` adjusted
score, `2.535e10` raw FLOPs, and `2.815e10` effective compute. Five separate
Fly Machines returned HTTP 408 and 15 remained pending, so the wrapper exited
124 after printing the clean aggregate; this is an infrastructure-completion
caveat, not an estimator failure.

The submission estimator now keeps only the live default route and direct
comparison modes: `r1` for the shallow K=3 path, `hadamard_first_cov` for the
old deep Hadamard route, and `hadamard_var1`/`hadamard_var2` for the first-layer
variance-matching variants, including `hadamard_var1_s<N>` strength sweeps.
`hadamard_chi`, `hadamard_b<N>`, and composable
`hadamard[_st<L>][_b<N>][_split<F>]` modes remain diagnostics for the same
variance route with chi-radial first-layer scaling, explicit block counts,
Strassen propagation matmuls, and split-block Hadamard row subsets; the
promoted default is equivalent to `hadamard_st3_b16`. Hadamard diagnostics
without an explicit block/sample override also use 16 blocks; none infer a
route, block count, or sample count from shape or the passed budget.
Older experimental modes for compressed K=3, K=1/K=2 diagnostics, low-rank
covariance, axis cubature, and sample blends were removed from `estimator.py`
after losing or becoming irrelevant to the current scorer frontier.

Superseded note for the 2026-07-07 augmented-K3 OOM gate: the original
`k3_aug_diag` failure reflected the Fly runner's default `shared-cpu-8x`
2048 MB Machine memory floor, not the challenge evaluation environment. The
grader allows 64 GB, and the runner now has an opt-in `FLY_VM_MEMORY_MB`
launch-time override while keeping ordinary runs at 2048 MB. A rebuilt Stage A
diagnostic measures the omitted upstream augmented K=3 degree-4 state, namely
the `(3,1)`/`(2,1,1)` power-cumulant slices plus `K211` feedback into the
degree-4 `r=1` core, behind `k3_aug_diag` using only the passed MLP object.
The first corrected 16 GB Fly rerun launched successfully with
`memory_mb=16384` and failed by the 60-second predictor wall-clock limit, not
OOM, before the original end-of-run diagnostic print emitted magnitudes. The
streaming rerun also hit the 60-second predictor wall-clock limit on the first
80 returned Machines, but it emitted layerwise rows before failure. The
augmented projection alone was already about `6.6x` to `7.2x` the local
degree-4 `r=1` core at layer 0, about `8x` to `10x` by layer 1, and commonly
above `20x` by layers 4-6; the total omitted core including K211 feedback was
similarly material, around `6.2x` to `6.8x` at layer 0 and tens of times the
local core by the later emitted rows. The individual `(3,1)` and `(2,1,1)`
power-cumulant slice norms were much larger than the local core norm, while
K211-total feedback was smaller than the augmented projection but still often
order-one to several times the local core. Because the state is clearly
material but the straightforward diagnostic cannot complete within the current
60-second scorer wall-clock path even at 16 GB, Stage B proceeds only as a
mode-gated Fly comparison; a timeout there should be treated as a charged-path
economics kill, not a memory kill. Stage B then compared `r1` against the
mode-gated `k3_aug` port on the same fixed 100 Fly MLPs with full per-MLP JSON
requested. Baseline `r1` returned all 100 rows, scoring `9.093e-1`
mean adjusted/final-layer MSE with `2.307e11` raw FLOPs per MLP and all rows
over combined effective budget. The augmented mode returned zero rows:
100/100 Machines failed in `predict()` at the 60-second wall-clock limit
(`matmul`/`multiply`/`add`), again with `memory_mb=16384` and no OOM signal.
No paired score delta is computable. Verdict: the omitted augmented state is
large, but the straightforward upstream augmented degree-4/K211 port is killed
for the current Fly/scorer path by wall-clock economics; do not promote.
