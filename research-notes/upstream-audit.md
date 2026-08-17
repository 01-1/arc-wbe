# Upstream Cumulant-Propagation Audit

Audit target: ARC public reference implementation cloned at
`ARC-estimation-research/upstream/mlp_cumulant_propagation` from
`https://github.com/alignment-research-center/mlp_cumulant_propagation`.

Local target: repository-root `estimator.py`, especially the shallow
`_factorized_k3_propagation` route and the depth-32 Hadamard route.

Rules note: this audit reads only public upstream code/paper material and local
estimator code/history. It does not run or score the local estimator.

## Executive Verdict

The local K=3 route is a focused, flopscope-oriented port of upstream's
factorized K=3 simple route, with structured factor groups and repeated-slice
caches. It already includes the central power-cumulant conversion path for K=3:
the nonlinearity computes power-cumulant slices, converts `pK` to ordinary
cumulants, subtracts repeated parts from the factored all-distinct third
cumulant, and carries `(3,)` plus `(2,1)` repeated-index slices.

The largest upstream state missing locally is not "power cumulants" in general.
It is the full upstream K=3 augmented/odd degree-4 path: the extra `(3,1)` and
`(2,1,1)` power-cumulant slices plus `K211_contrib` feedback into the degree-4
core. Local `_factored_nonlin_k3_r1_fast` is a hybrid: it uses the upstream
simple K=3 term filter, deliberately skipping those augmented slices, while
still carrying an `r=1` degree-4 harmonic/readout state. This is testable, but
history strongly cautions that pure analytic cumulant lanes are bias-floored at
depth 32 and that compressed joint-k3 transports have failed.

## Correspondence Table

| Upstream component | Local component | Delta |
|---|---|---|
| `kprop_harmonic.get_int_cond(k_max)` tracks diagonal/power slices satisfying `sum(ceil(part_i/2)) <= K`. | `_all_terms_iso_k3`, `_terms_iso_k3`, `_terms_iso_k3_grouped`, `_factored_nonlin_k3_r1_fast`. | Local specializes the K=3 term set and groups factors for flopscope. It is not a generic K/`kind` implementation. |
| `nonlin_kprop(..., use_pK=True)` computes power cumulant slices, `DSTower.from_slices`, then `DS_pK_to_K`. | `_factored_nonlin_k3_r1_fast`: builds `p_slices`, converts with `_ds_pk_to_k`. | Same basic power-cumulant-to-cumulant mechanism for the local K=3 simple route. Not an omitted mechanism. |
| `FactoredTensor.get_repeated()` carries all repeated slices except all-distinct for degree 3: `(3,)` and `(2,1)`. | `_FactoredThird.get_repeated()`, `diag()`, `dslice_21()`, `contracted_diag()`. | Local carries the same degree-3 repeated slices and caches them. It uses grouped/diagonal factors to avoid repeatedly materializing full factors. |
| Upstream K=3 simple: skips `(3,1)` and `(2,1,1)` pK slices for degree 4 and returns `K_out[4] = DS_harmonic_proj(..., r_out=2)`. | Local `_factored_nonlin_k3_r1_fast` also skips those term groups, but returns `4: _ds_harmonic_proj_r1(k_ds[4])`. | Local is not a verbatim simple-mode port: it keeps a degree-4 `r=1` harmonic/readout carrier while retaining the simple-mode term filter. |
| Upstream K=3 augmented: includes `(3,1)` and `(2,1,1)` pK slices, returns `K_out[4] = DS_harmonic_proj(..., r_out=1)`, and adds `pK_111 -> K_211` plus `pK_211 -> K_211` feedback. | Local has the `r=1` degree-4 projection shape, but no equivalent for the augmented skipped slices or `K211_contrib` feedback in `_factorized_k3_propagation`. | This is the most concrete omitted upstream state. It is the local route's main K=3 audit delta. |
| Odd-K extra full trace in paper Section 4.1: for odd K, carry the full trace of the `(K+1)`-order cumulant; for K=3 this is a degree-4 trace-like state. | Local carries degree-4 `r=1` state used for final ReLU diagonal correction, but from the simple-filter slice set and without augmented `K211` feedback. | Local has part of the odd-K trace idea, but not the full upstream augmented state. A gate should compare local hybrid K=3 versus full augmented K=3 on the same Fly route before considering estimator edits. |
| `factor_k3.factored_nonlin_kprop_k3`: all-distinct third cumulant stored as factor columns; each layer adds two main `n`-column factor groups, plus optional simple/augment contributions. | `_FactoredThird.add_factor_groups` adds the same two main groups per nonlinear layer; `_contract_groups` batches repeated dense factors. | Same factor-list growth mechanism. At depth 32, factor columns grow roughly linearly with hidden layers and repeated contractions/readout make total depth cost effectively quadratic, matching the paper's caveat. |
| Upstream factorized augmented K=3 adds extra groups for simple full-metric `112 -> 111` and augmented `211 -> 111` cases. | Local does not add those groups. | Adding them would increase factor count and repeated-slice work. It is likely shallow-only unless paired with a hard gate on flopscope/residual cost. |
| Upstream ReLU nonlinearity uses `relu_wick_coef(mean,var,k,p)` with exact piecewise-polynomial formulas and variance clamp `1e-10`. | Local `_relu_wick_from_stats`/`_relu_mean_from_cumulant_diags` and Gaussian ReLU identities use flopscope stats functions, `_MIN_VARIANCE`, and hand-expanded Wick terms. | Formula class matches. Differences are implementation constants, dtype, and local final-readout Edgeworth terms through orders 3/4/6/7/8. |
| Upstream paper/code FLOP accounting: PyTorch flop counter plus custom elementwise models and symmetric-tensor adjustment factors; paper excludes redundant symmetric work. | Local scorer path is flopscope plus residual-time charging; Hadamard route uses actual charged array operations and Strassen arithmetic. | Do not trust upstream FLOP counts for promotion. Every candidate must first pass a local syntax/scorer-path proof and then `make fly` only if estimator behavior changes. |
| Upstream experiments stop at materially shallower depth than current depth-32 grader and explicitly leave large-depth matching unresolved. | Local default at depth 32 is Hadamard cubature with first-layer covariance recolor, first-successor variance match, and Strassen levels. | Upstream K=3 should inform gates/corrections, not replace the current deep route without new depth economics. |

## Repeated-Index And Power-Cumulant Slices

Upstream tracks power cumulants by integer partitions of output powers. For K=3
simple/factorized, the practically important degree-3 repeated slices are:

- `(3,)`: all three cumulant indices on the same neuron.
- `(2,1)`: two equal indices and one distinct index; upstream stores sorted
  slices and returns permutations as needed.
- `(1,1,1)`: all-distinct third cumulant, stored in factorized form rather
  than materialized.

Local `_FactoredThird` carries all three in the same split: all-distinct as
factor groups, repeated `(3,)` as `diag()`, and repeated `(2,1)` as
`dslice_21()`. Local `_factored_nonlin_k3_r1_fast` also subtracts
`p111.get_repeated()` from `k_ds[3]`, matching upstream's repeated/all-distinct
separation.

The omitted/projected-away upstream pieces are degree-4 K=3 augmented slices:

- `(3,1)` and `(2,1,1)` are deliberately skipped by upstream simple K=3 and by
  local K=3.
- Upstream augmented K=3 includes those slices and projects the degree-4
  cumulant to `r=1`; local already uses an `r=1` projection but without those
  augmented source slices.
- Upstream augmented K=3 also feeds factored `pK_111` and incoming `WK[3]`
  repeated information into `K211_contrib`, modifying `K_out[4].core`.

Thus the local route does not generally omit the paper's power-cumulant
adjustment; it omits the augmented odd-K degree-4 trace/core state.

## Factor Growth And Depth Economics

For K=3, upstream's factorized nonlinearity adds two main `n`-column factor
groups per layer to the all-distinct third cumulant, after carrying forward the
contracted previous groups. That is the implementation counterpart of the
paper's "factor list grows by O(n) per layer" caveat. At width 256 and depth 32,
this is not just a memory concern: every later linear contraction and repeated
slice/readout sees the accumulated group list, so explicit depth cost grows
roughly like a layer sum over prior groups.

Local `_FactoredThird` improves constants by grouping factors and batching
unique dense contractions, but it does not compress or truncate the factor list
in the live K=3 path. Upstream also does not provide a magic compression for
K=3 factor lists in the audited route; it relies on factorization for width
scaling and accepts worse depth scaling. This supports the current local split:
K=3 remains a shallow/diagnostic route; Hadamard remains the depth-32 default.

## ReLU And Finite-Width Differences

The upstream ReLU transform is Wick-coefficient based. It computes
`E[d^k ReLU(Z)^p]` for a Gaussian with the propagated coordinate mean and
variance, including powers `p>1` for power cumulants. Local K=3 uses the same
mathematical family but hand-expands/caches the K=3 terms and uses flopscope's
normal `pdf/cdf` wrappers. Local final-layer readout additionally applies an
Edgeworth-style ReLU mean correction using diagonal `k3`, diagonal `k4`, and
quadratic correction terms through Wick orders 6, 7, and 8.

Finite-width constants differ in three places:

- upstream's `HTensor.metric` can be a full or average-case metric; local K=3
  contracts concrete `W.T @ ... @ W` style state in the WhestBench orientation;
- upstream `AUGMENT` keeps more degree-4 harmonic state at the same asymptotic
  width order; local simple K=3 projects that away;
- local deep Hadamard uses exact zero-mean Gaussian first-layer mean/covariance
  identities and a sample-derived Gaussian marginal variance target after the
  first successor layer, which is a sampling-route correction rather than an
  upstream cumulant-propagation state.

## Ranked Gate Or Estimator-Change Candidates

1. **Gate upstream K=3 augmented state against local K=3 simple on shallow Fly
   mode.** Mechanism: port only the missing augmented degree-4 `r=1`/`K211`
   feedback behind a mode flag and compare against `r1` on shallow or explicit
   diagnostic shapes through Fly, not local scoring. Nearest rejection:
   "analytic cumulant lane bias-floored" and pure K=3 depth-32 were negative.
   Difference: this is not another diagonal-only cumulant ladder or deep
   default proposal; it isolates the exact upstream state omitted from the
   local port and should be killed if shallow `r1` does not improve enough per
   charged FLOP.

2. **Add a read-only trace audit mode for local K=3: report local degree-4
   `r=1` state versus upstream augmented `K211` trace proxies layer by layer.**
   Mechanism: mode-gated diagnostics only, using the passed MLP and no labels,
   to quantify whether the omitted odd-K trace state is numerically large before
   writing estimator behavior. Nearest rejection: final-row Gaussian pull and
   energy thermostats lost. Difference: this is not a correction applied to
   Hadamard samples; it is a structural state-size gate for the upstream omitted
   cumulant component.

3. **Test a final-layer-only augmented K=3 readout shortcut for shallow route.**
   Mechanism: compute the augmented degree-4 `r=1` contribution only as needed
   for `_final_r1_relu_mean_from_tower`, avoiding full propagation if the
   missing state is mainly a final ReLU mean constant. Nearest rejection:
   final-layer sample-cumulant and Gaussian final pulls lost. Difference: this
   uses deterministic propagated upstream cumulant state, not sample-block
   postprocessing or a Gaussian closure pull.

4. **Measure factor-list growth and repeated-slice cache hit economics under
   flopscope for local K=3 at depth 8/16/32.** Mechanism: an instrumentation
   mode records factor columns/groups and charged operations without scoring
   local estimator outputs. Nearest rejection: structured compression and dense
   top-k rank caps lost under residual scoring. Difference: this is a
   measurement gate to avoid re-running bad compression; it should define exact
   depth/factor thresholds before any new compression attempt.

5. **If candidate 1 wins shallow but loses depth economics, test upstream
   augmented state as an analytic-prefix-only diagnostic before Hadamard suffix.**
   Mechanism: use augmented K=3 only for the first one or two layers to generate
   a better prefix moment target, then hand off to the existing Hadamard route.
   Nearest rejection: `hybx2` joint-k3 analytic-prefix sampler and exact
   layer-2 covariance anchoring lost. Difference: this must not reuse the
   failed compressed quadratic transport; it is only justified if candidate 1
   proves the augmented state itself is valuable and a new low-covariance
   carrier is specified.

## Not Worth Testing

- **Wholesale upstream factorized K=3 as the depth-32 default.** Local history
  already collected pure factorized K=3 depth-32 evidence and the paper itself
  warns factorization worsens explicit depth scaling.

- **"Add power cumulants" as a generic task.** Local K=3 already uses
  power-cumulant conversion and repeated third-cumulant slices. The actionable
  missing piece is augmented degree-4 state, not power cumulants broadly.

- **Upstream paper FLOP-count-based promotion.** Their counts adjust symmetric
  tensor work and use PyTorch/custom accounting. The repo score path is
  flopscope plus residual-time charging, so paper FLOPs are only a rough
  design prior.

- **Another broad first-successor or later-layer Gaussian covariance/variance
  correction.** Local history has multiple losses: broader early-layer
  variance correction, full first-successor covariance recolor, correlation
  restoration, second/third variance-only layers, and exact layer-2 covariance
  anchoring.

- **Compressed joint-k3 transport by column truncation or top-k rank caps.**
  M2b/M2c history shows truncation recovers poor diagonal/repeated/distinct
  third-cumulant correlations and that covariance feasibility, not simple rank
  mass, is the wall.

- **Odd-K global energy thermostat for Hadamard samples.** The upstream odd-K
  trace state is more specific than total sample energy. Local global energy
  thermostat and scale shrink/cap variants were weak or negative.

## Paper Cross-Checks

The paper's Section 4.1 additions match the code:

- power cumulants are implemented through `use_pK=True`, Wick coefficients for
  powers `p`, and `DS_pK_to_K`;
- odd K carries a degree `K+1` trace-like state, represented in code through
  simple/augmented degree-4 harmonic choices for K=3;
- factorized K>=3 keeps the top cumulant as factor lists and improves width
  scaling while making explicit depth scaling worse.

Section 6.4's ablation warning is therefore best read locally as: do not remove
or bypass the existing K=3 power-cumulant/repeated-slice machinery, and do not
confuse the local omission of augmented degree-4 state with omission of power
cumulants altogether.
