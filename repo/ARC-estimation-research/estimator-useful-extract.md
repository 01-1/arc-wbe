# Estimator-Useful Extract From ARC Alignment Posts

Source bundle: `alignment-posts-combined.md`, the individual post files in this
folder, and `wide-random-mlps-paper.pdf`:
[*Estimating the expected output of wide random MLPs more efficiently than
sampling*](https://arxiv.org/abs/2605.05179), arXiv:2605.05179v2. This note
extracts ideas that are directly or tangentially useful for the repository
estimator. It is not a full summary of the posts or paper; alignment motivation
is kept only where it changes estimator design.

## Quick Verdict

The useful material is the mechanistic-estimation spine:

- Random ReLU MLPs under Gaussian input are the same mathematical setting as
  the estimator target. See `alignment-posts-combined.md:632` and
  `alignment-posts-combined.md:1313`.
- Cumulant propagation is the direct ancestor of the shallow `r1` route in
  `../estimator.py`. See `alignment-posts-combined.md:676` and
  `../estimator.py:1334`.
- Gaussian moment propagation and exact ReLU mean/covariance identities map to
  the current first-layer covariance recolor and first-successor variance
  match. See `alignment-posts-combined.md:1196` and `../estimator.py:1455`.
- Deduction-projection and mechanistic `L^2` sketching are the most useful
  tangential ideas: choose which analytic state to keep by downstream value per
  FLOP, rather than by low order alone. See `alignment-posts-combined.md:1627`.
- The paper adds several implementation-relevant details missing from the blog:
  power cumulants, an extra full trace for odd maximum cumulant order, the
  factorized K>=3 representation, and a caution that factorization improves
  width scaling while worsening explicit depth scaling.

The not-useful material is mostly the alignment motivation, trained-network
advice, and public-instance calibration. Under this challenge's rules, any
estimator change must use only the passed MLP object and legitimate moment
identities.

## Paper-Specific Additions

### 0.1 Quenched, Not Annealed, Is The Right Mental Model

The paper stresses that the estimator takes the weights of one particular
random MLP and estimates the expectation over random inputs. Only after
comparing estimates to truth do they average results over random weight draws.

Estimator relevance:

- This matches the challenge: `predict()` sees one passed MLP object and must
  estimate its per-layer activation means.
- Avoid importing infinite-width or annealed-over-weights formulas unless they
  condition correctly on the concrete weight matrices.

Actionability:

- Keep any analytic correction weight-conditional, e.g. based on `W.T @ W`,
  propagated covariances, or summaries of the concrete MLP.

### 0.2 Power Cumulants Are Not Optional In Their Working Algorithm

The paper says the naive cumulant propagation sketch is too inaccurate because
it misses large correlations from repeated neurons inside a cumulant. Their
fix is to propagate "power cumulants": cumulants of powers such as
`kappa[X_i^a, X_j^b, ...]`, then recover ordinary cumulants by partition
formulas.

Paper evidence:

- Section 4.1 introduces power cumulants as the first necessary adjustment.
- The power-cumulant ablation in Section 6.4 loses the desired width scaling:
  the paper reports `O(1)` MSE rather than the intended `O(1/n^K)`.

Estimator relevance:

- This explains why tracking only ordinary off-diagonal cumulants or only a
  low-order Gaussian closure is fragile.
- The local K=3 route already has structured diagonal/power-like machinery;
  future K-route comparisons should check whether every repeated-index slice
  implied by the paper is represented or deliberately projected away.

Actionability:

- If auditing the local K=3 route against upstream, prioritize repeated-index
  slices and power-cumulant-to-cumulant conversion before prettier algebra.
- For Hadamard variants, this is a warning that matching only off-diagonal
  structure can lose high-variance diagonal/repeated-neuron effects.

### 0.3 Odd K Needs One More Trace

For odd maximum cumulant order `K`, the paper tracks a single number: the full
trace of the `(K + 1)`-order cumulant tensor. For `K = 1`, that means tracking
the trace of the covariance matrix along with the mean.

Estimator relevance:

- This is a compact, cheap statistic with theoretical backing.
- It is conceptually close to the repo's history of variance-energy and
  marginal variance corrections, but with a clearer cumulant-propagation origin.

Actionability:

- If revisiting shallow/analytic routes, check that odd-K modes carry the
  extra trace-like state.
- For deep Hadamard modes, this suggests cheap global energy statistics may be
  worth considering only when they are tied to a specific projected cumulant
  state; generic energy thermostats have already had weak local evidence.

### 0.4 Augmented Cumulants Improve MSE Constants, Not Asymptotic Width Cost

The paper's augmented algorithm tracks more harmonic components, described as
the tensor except its traceless component in the odd-K sketch. In the technical
supplement, the augmented version keeps every harmonic part that can be
propagated in `O(n^K)` time and contributes at least `Omega(n^-K)` MSE.

Estimator relevance:

- This is the most precise version of "keep statistics by value per FLOP."
- It supports the existing instinct to track selected slices rather than full
  tensors.

Actionability:

- Treat "augmented" as a menu of extra state with a cost/error criterion, not
  as a blanket instruction to add all covariance/cumulant details.
- For this challenge, an augmented state must survive residual-time and
  flopscope realities, not only asymptotic FLOP counts.

### 0.5 Factorization Is The Main Width Win But Hurts Depth Scaling

The paper states that for `K >= 3`, the top cumulant tensor can be represented
in factorized form, reducing the width dependence from `O(n^(K+1))` to
`O(n^K)` while producing the same estimate. For the K=3 explanation, the
third-cumulant factor list grows by `6n` factors per layer, so the explicit
depth cost becomes quadratic in the number of layers.

Estimator relevance:

- This matches the local `r1` factorized K=3 route and explains why it is good
  at shallow depth but loses appeal at depth 32.
- The paper's headline theory assumes fixed constant depth; our current grader
  has depth 32, so the depth caveat is not cosmetic.

Actionability:

- Do not expect upstream factorized cumulant propagation to replace the deep
  Hadamard route without a depth-specific compression or truncation of factors.
- If comparing upstream, measure factor-count growth and layer-by-layer
  residual overhead, not just arithmetic FLOPs.

### 0.6 Experiments Stop Below Our Current Depth

The paper's main experiments vary width from 4 to 256 and hidden layers from 2
to 12, with K from 1 to 4. At width 256, factorized variants beat Monte Carlo
strongly at 2 and 4 hidden layers, but begin underperforming sampling for
8 hidden layers when K reaches 4. The open-problems section explicitly says
large-depth matching is unresolved.

Estimator relevance:

- This supports the current split: analytic K=3 for shallow/diagnostic runs,
  Hadamard-cubature hybrids for depth-32 default.
- It also cautions against overvaluing paper plots at width 256; the depth is
  not the contest depth.

Actionability:

- Any paper-derived route for the current grader needs `make fly` proof.
- The most promising paper-derived work is likely a small local correction to
  the current deep route, not wholesale replacement by K=4 cumulants.

### 0.7 Their FLOP Accounting Assumes Efficient Symmetric Tensor Kernels

The paper counts FLOPs with adjustments that exclude redundant operations from
naive symmetric tensor einsums. It also notes that wall-clock can be poor
because small fractions of FLOPs may be implemented inefficiently.

Estimator relevance:

- The challenge uses flopscope and residual-time charging, so theoretical or
  adjusted paper FLOPs are not automatically transferable.
- Local history already shows residual overhead can dominate adjusted score
  near the score-efficient frontier.

Actionability:

- If porting an upstream idea, first make a tiny mode and measure flopscope
  behavior; do not trust paper FLOP counts directly.
- Symmetric-tensor savings only count if the implementation actually avoids
  charged work under the local scorer.

### 0.8 Use The ARC GitHub Repo For Upstream Comparison

Use the ARC repository:
`https://github.com/alignment-research-center/mlp_cumulant_propagation`.

Actionability:

- For a future code audit, start with `mlp_cumulant_propagation`, then
  reconcile any naming drift in the paper text or internal module names.

## Directly Useful Extracts

### 1. The Estimation Problem Matches The Contest Shape

The posts formulate the relevant task as: given random MLP weights, estimate
the expected network output under Gaussian input, competing with Monte Carlo
sampling in MSE per FLOP. The combined file states the random MLP problem at
`alignment-posts-combined.md:632` through `alignment-posts-combined.md:704`.
The later wide-random-MLP post restates the target at
`alignment-posts-combined.md:1321` through `alignment-posts-combined.md:1363`.

Estimator relevance:

- The challenge estimator predicts per-layer activation means, not just a
  scalar final output, but each row is still an expected ReLU MLP activation
  under standard-normal input.
- The scoring frontier is MSE versus effective FLOPs, matching the posts'
  comparison axis at `alignment-posts-combined.md:1380`.
- Width 256 is explicitly discussed in the wide-MLP empirical section at
  `alignment-posts-combined.md:1384`, though that post's reported depth is
  shallow relative to the current depth-32 grader.

Current implementation overlap:

- `../ESTIMATOR_HISTORY.md:10` documents the current width/depth/budget shape.
- `../ESTIMATOR_HISTORY.md:13` documents the current deep route as Hadamard
  cubature plus moment correction.
- `../estimator.py:2371` dispatches by depth and budget.

Actionability:

- Keep estimator comparisons framed as MSE per effective FLOP, not wall-clock
  alone. The posts explicitly note that wall-clock may underperform despite
  FLOP wins, which matters less for this FLOP-scored setting.
- Any imported idea from the posts still needs a Fly-mode proof because the
  theory is asymptotic and shallow-depth; current depth-32 behavior is its own
  beast.

### 2. Cumulant Propagation Is Already The Shallow Analytic Route

The combined post describes cumulant propagation as tracking means,
variances/covariances, and higher multi-way correlations layer by layer. See
`alignment-posts-combined.md:676` through `alignment-posts-combined.md:702`.

Estimator relevance:

- The estimator's `_factorized_k3_propagation` is exactly this class of method:
  it starts from Gaussian input cumulants, contracts through weights, applies
  the ReLU nonlinearity transform, and returns activation means.
- The challenge's width 256 means full high-order cumulant tensors are too
  expensive; the code uses factorized and grouped structure instead.

Current implementation overlap:

- `../estimator.py:1334` implements the K=3 factorized cumulant propagation
  fallback.
- `../ESTIMATOR_HISTORY.md:29` records the optimized grouped `r=1` K=3 route
  as the shallow fallback and diagnostic baseline.

Actionability:

- The linked upstream `mlp_cumulant_propagation` repo and paper are worth
  comparing against the local K=3 implementation for finite-width tricks,
  ablations, and term-selection details. This is the most direct external
  technical follow-up.
- Do not assume K=3 should be promoted for depth 32. History records it as too
  expensive or weaker for the current deep scorer frontier.

### 3. Gaussian ReLU Moment Identities Are Central

The posts describe covariance propagation as modeling each layer's distribution
and then projecting it back into a tractable family; the relevant note is at
`alignment-posts-combined.md:241` through `alignment-posts-combined.md:246`,
with a footnote description at `alignment-posts-combined.md:1196`.

Estimator relevance:

- The current deep route relies on exact Gaussian ReLU identities for the first
  hidden layer and approximate Gaussian marginal variance for the first
  successor layer.
- This is the strongest bridge between the theoretical posts and the current
  practical estimator: use analytic Gaussian moments where they are reliable,
  then use structured cubature for the remaining finite-depth geometry.

Current implementation overlap:

- `../estimator.py:1455` implements exact zero-mean Gaussian ReLU mean and
  covariance.
- `../estimator.py:1471` implements nonzero-mean Gaussian ReLU marginal
  variance.
- `../estimator.py:2051` recolors the first hidden ensemble to exact mean and
  covariance.
- `../estimator.py:2100` applies the first-successor variance match.

Actionability:

- The useful principle is selective Gaussian closure, not full Gaussian closure.
  The history shows broader exact covariance correction later in the network
  has often lost.
- Future experiments should treat exact moment anchoring as a local correction
  to sampled geometry, not as proof that more analytic recoloring is always
  better.

### 4. Low-Order Deviations From Gaussianity Are The Explicit Research Handle

The wide-random-MLP post says the methods start from Gaussian approximations
and track low-order deviations. See `alignment-posts-combined.md:1469` through
`alignment-posts-combined.md:1479`.

Estimator relevance:

- This matches the estimator's K=3 cumulants, first-layer covariance recolor,
  marginal variance matching, Edgeworth diagnostic hook, and rejected
  skew/kurtosis gates.
- It supports looking for cheap, local low-order corrections, especially where
  Hadamard cubature creates finite-ensemble distortions.

Current implementation overlap:

- `_edgeworth_relu_mean` at `../estimator.py:1490` is a low-order correction
  hook.
- The current posthoc parser still exposes Gaussian pull, Edgeworth blend,
  kurtosis gate, trimming, and second variance strength through diagnostic
  Hadamard modes around `../estimator.py:2208`.

Actionability:

- Low-order correction ideas remain reasonable as diagnostics, but many nearby
  variants have already been falsified in `../ESTIMATOR_HISTORY.md`.
- Before adding another skew/kurtosis correction, check the history for the
  exact failure mode and require a mechanism different from "more moment
  matching."

### 5. ReLU Polynomial/Hermite Approximation Is A Useful Lens

The older random-MLP footnotes point to applying cumulant propagation through
ReLU by using polynomial approximation, then treating the cumulants of the
preactivation as Gaussian for the selected approximation. See
`alignment-posts-combined.md:1279` through `alignment-posts-combined.md:1294`.

Estimator relevance:

- This is effectively a Hermite/low-degree view of the nonlinear transform.
- It gives a conceptual basis for selecting low-order terms such as degree-2,
  degree-3, and degree-4 features rather than blindly tracking raw moments.

Current implementation overlap:

- The grouped K=3 route and degree-4 harmonic tracking in
  `../ESTIMATOR_HISTORY.md:29` fit this lens.
- Several rejected `H2`/final cumulant features are recorded later in the
  history, so this lens is already partially mined.

Actionability:

- If revisiting analytic-suffix or projection methods, use Hermite basis terms
  as candidates and score them by downstream error reduction per FLOP.
- Re-adding broad polynomial features without a projection rule is unlikely to
  beat the current floor.

### 6. Sampling Is A Baseline And A Component, Not The Enemy

The posts contrast mechanistic estimation with Monte Carlo sampling but use
sampling as the comparison target throughout. See
`alignment-posts-combined.md:1365` through `alignment-posts-combined.md:1378`.

Estimator relevance:

- The deep default is not purely analytic. It uses randomized antithetic
  Walsh-Hadamard sign cubature plus analytic moment anchoring.
- This hybrid is consistent with the posts' goal: exploit structure enough to
  beat naive sampling at a fixed FLOP budget.

Current implementation overlap:

- `../estimator.py:1368` constructs randomized Hadamard sign blocks.
- `../estimator.py:2006` runs the first-covariance-recolored Hadamard route.
- `../ESTIMATOR_HISTORY.md:13` records the promoted deep default.

Actionability:

- Continue treating deterministic or randomized cubature as a legitimate
  mechanistic/sampling hybrid.
- A promising idea is one that reduces ensemble variance or bias at the same
  block count, not merely one that is more philosophically "mechanistic."

### 7. Tail/Low-Probability Claims Are Tangential But Suggest Diagnostics

The wide-MLP post says their method improves tail probability estimation over
Monte Carlo in some settings. See `alignment-posts-combined.md:1394` through
`alignment-posts-combined.md:1403`.

Estimator relevance:

- The contest scores means, not rare-event probabilities.
- However, high-leverage final activations and rare large ReLU paths can
  dominate MSE, so tail behavior can still explain estimator failures.

Current implementation overlap:

- History includes high-leverage clipping, trimmed final means, inverse
  variance weighting, and related robust-cubature probes.

Actionability:

- Useful as a diagnostic question: are final-layer errors dominated by a few
  high-radius or high-gate-path rows?
- Not directly useful as a new objective unless a Fly diagnostic can report raw
  activation samples or block-level residuals without violating evaluator
  rules.

## Tangentially Useful Extracts

### 8. Deduction-Projection Is The Right Abstraction For Budgeted State

The random-products post describes deduction-projection estimators as
alternating exact update steps with projections that control state size. See
`alignment-posts-combined.md:1627` through `alignment-posts-combined.md:1643`.

Estimator relevance:

- Cumulant propagation is named as an example of this pattern.
- The local question becomes: after each layer, which state should survive the
  projection under the depth-32 FLOP budget?

Current implementation overlap:

- The K=3 grouped route is a handcrafted projection onto selected structured
  cumulant factors.
- The deep Hadamard route projects the input distribution into a finite
  structured ensemble, then periodically corrects low-order moments.

Actionability:

- Use deduction-projection language to design experiments:
  "deduce one layer exactly/approximately, then project to the highest-value
  state under the remaining budget."
- This is a better search frame than adding ad hoc moment corrections.

### 9. Mechanistic L2 Sketching Suggests Tail-Aware Projection

The random-products post formulates projection as preserving the parts of a
head representation that matter most when paired with a random tail. See
`alignment-posts-combined.md:1660` through `alignment-posts-combined.md:1700`.

Estimator relevance:

- In an MLP, the "tail" after a layer is the random remaining set of weight
  matrices and ReLUs. A projection should keep components of the current
  activation distribution that most affect downstream expected activations.
- This suggests weighting corrections by downstream sensitivity rather than by
  local moment error alone.

Current implementation overlap:

- Some downstream-aware variants have already been tested and lost, including
  next-weight-aware variance strength and downstream-aware covariance gauges.
  See the rejected-ideas portion of `../ESTIMATOR_HISTORY.md` before retrying.

Actionability:

- A cleaner future version would compute a cheap random-tail kernel proxy and
  keep leading modes. At depth 32 this must be extremely cheap, probably a
  diagonal or low-rank proxy rather than full covariance transport.
- Treat this as a research direction, not a simple patch.

### 10. Random Halfspaces Connect To First-Layer ReLU Gates

The random-products post notes that random halfspace intersections correspond
to all selected one-layer ReLU neurons being active. See
`alignment-posts-combined.md:1612` through `alignment-posts-combined.md:1616`.
The older appendix also frames one-layer ReLU zero-output probability this way
around `alignment-posts-combined.md:1247`.

Estimator relevance:

- First-layer ReLU gates are random halfspaces under Gaussian input.
- Exact first-layer Gaussian ReLU covariance is essentially exploiting this
  symmetry.

Current implementation overlap:

- `../estimator.py:1455` uses the arc-cosine ReLU covariance formula.
- `../estimator.py:2051` uses this as the first hidden-layer anchor.

Actionability:

- First-layer gate statistics are a solid place for exact analytic treatment.
- Later layers lose the clean random-halfspace symmetry because their inputs
  are neither independent Gaussian nor zero-mean.

### 11. Compression/Advice Is Conceptual, Not Challenge-Actionable

The posts discuss "advice" or efficient compression for trained networks at
`alignment-posts-combined.md:1072` through `alignment-posts-combined.md:1119`
and `alignment-posts-combined.md:1471` through
`alignment-posts-combined.md:1489`.

Estimator relevance:

- Conceptually, this says learned or structured networks may require knowing
  which higher-order deviations to track.
- The current challenge MLPs are random initialized; no external explanation is
  needed or allowed.

Actionability:

- Do not use external advice, public labels, public seeds, or case-specific
  calibration.
- The only safe interpretation is internal compression of the passed MLP
  weights, such as low-rank summaries, symmetry summaries, or moment summaries
  computed inside `predict()`.

## Current Estimator Correspondence Table

| Post idea | Source | Current local state | Practical note |
|---|---:|---|---|
| Match/outperform sampling by using structure | `alignment-posts-combined.md:71` | Current route is structured Hadamard plus moment correction | Keep MSE/FLOP framing |
| Mechanistic estimator reads behavior from weights | `alignment-posts-combined.md:208` | All live routes use only passed MLP weights and allowed randomness | Rule-compatible |
| Covariance/cumulant propagation | `alignment-posts-combined.md:241` | K=3 shallow route and Gaussian moment hooks | Already central |
| Random MLP Gaussian input target | `alignment-posts-combined.md:632` | Challenge target | Direct match |
| Cumulant propagation layer by layer | `alignment-posts-combined.md:676` | `_factorized_k3_propagation` | Shallow fallback |
| Width-256 ReLU MLP empirical FLOP curves | `alignment-posts-combined.md:1384` | Grader width is 256 but depth is 32 | Need Fly proof |
| Power cumulants and repeated-index slices | paper Section 4.1 and 6.4 | partially represented in K=3 machinery | high priority for upstream audit |
| Odd-K extra full trace | paper Section 4.1 | related to variance/energy corrections | useful only if tied to projected cumulants |
| Factorized K>=3 tensors | paper Section 4.3 and S.4.3 | local `r1` factorization | depth scaling limits default use |
| Paper FLOP accounting excludes symmetric redundancy | paper Section 6.1 and Appendix J | local flopscope may charge differently | benchmark locally |
| Low-order deviations from Gaussian | `alignment-posts-combined.md:1471` | variance match, K=3, Edgeworth diagnostics | Useful but many variants failed |
| Deduction-projection | `alignment-posts-combined.md:1627` | grouped cumulants and finite ensemble projection | Good experiment frame |
| Mechanistic `L^2` sketching | `alignment-posts-combined.md:1672` | only partially explored | Possible future research |
| Trained-network advice/compression | `alignment-posts-combined.md:1072` | not used | Mostly rule-risky unless internal to MLP |

## Ideas Worth Testing Only If They Differ From Rejected Variants

1. Upstream details audit:
   compare `mlp_cumulant_propagation` against local K=3 handling,
   especially power cumulants, repeated-index slices, odd-K trace state,
   factor growth with depth, ablations, and finite-width constants.

2. Tail-aware projection proxy:
   build a very cheap diagonal or small low-rank proxy for downstream
   sensitivity and use it to select which moment corrections survive. This is
   the closest concrete interpretation of mechanistic `L^2` sketching.

3. Hybrid analytic prefix plus sampled suffix:
   use a short analytic prefix to reduce initial moment bias, then sample the
   suffix. History says this has been probed, but the deduction-projection
   framing may help choose a cleaner projection point.

4. Hermite-basis correction:
   revisit low-degree ReLU/Hermite terms only if they can be cheaply projected
   by downstream value. Raw addition of H2/H3/H4-style features has weak prior
   because nearby attempts are documented as losses.

5. First-layer-only exactness:
   keep exploiting exact first-layer gate/arc-cosine identities. The posts
   strongly support this, and the local history supports that broader exact
   recoloring is much less reliable.

## Guardrails From The Posts And Repo Rules

- The estimator must use only the passed MLP object, the budget, and legitimate
  internal randomness/moment identities.
- Do not fit or calibrate against public MLPs, public seeds, leaderboard rows,
  hidden truth, or external Monte Carlo samples.
- Do not turn the "advice" idea into external files, labels, public-case
  fingerprints, or network calls during evaluation.
- For estimator behavior changes, update `../ESTIMATOR_HISTORY.md` in the same
  turn and use the repo's approved estimator checks.

## Bottom Line

The posts validate the estimator's current direction more than they supply a
drop-in patch. The directly useful extraction is:

- keep first-layer exact Gaussian ReLU moment anchoring;
- keep K=3/cumulant propagation as the analytic reference route;
- use Hadamard/cubature hybrids for depth-32 budget efficiency;
- think of future improvements as deduction-projection or `L^2` sketching
  problems, where every retained statistic must earn its FLOPs by downstream
  MSE reduction.
