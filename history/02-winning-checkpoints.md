# Winning Checkpoints

- **K=2 covariance plus sampling floor.** The early analytical baseline tracked
  full covariance with exact marginal ReLU moments and blended in antithetic
  sampling when it fit below the old 10% score floor. At width-256/depth-8 it
  used about `6.69e9` FLOPs and landed around `2.12e-05` final-layer MSE. It
  was useful as a floor, but left too much accuracy on the table.
- **Factorized K=3 `r1`.** Porting the upstream factorized K=3 path, caching
  repeated diagonal slices, and routing through the `r=1` harmonic subset beat
  K=2 despite higher compute. The final-layer shortcut later reduced the
  width-256/depth-8 route to about `1.63e10` FLOPs with unchanged predictions.
- **Structured grouped `r1`.** Keeping third-cumulant terms as structured
  factor groups reduced analytical FLOPs from about `1.63e10` to `1.25e10` on
  width-256/depth-8 while preserving predictions. Local cached-mini scoring
  with residual-time charging kept exact grouped `r1` ahead of the compression
  variants that were safe to run.
- **Depth-32 retargeting.** Exact grouped `r1` fit analytically at
  width-256/depth-32 but left too little residual-time headroom. A compressed
  K=3 depth route was a temporary bridge, but randomized Hadamard cubature was
  far better for deep networks.
- **First-covariance Hadamard.** Plain Hadamard blocks outperformed compressed
  K=3 on depth-32. Recoloring the first hidden ensemble to the exact
  first-layer Gaussian ReLU mean/covariance improved the route further without
  using labels. Fly EWR 80-result sweeps with corrected residual compute found
  the best adjusted-score frontier at 13 blocks:

  | Blocks | Final-layer MSE | Adjusted score | Effective compute |
  |---:|---:|---:|---:|
  | 11 | `4.473e-6` | `4.473e-7` | `2.583e10` |
  | 12 | `3.664e-6` | `3.786e-7` | `2.811e10` |
  | 13 | `3.068e-6` | `3.430e-7` | `3.041e10` |
  | 14 | `3.458e-6` | `4.150e-7` | `3.267e10` |
  | 16 | `3.230e-6` | `4.423e-7` | `3.724e10` |

- **First-successor variance-scaled Hadamard.** Preserving the first-covariance
  ensemble means while rescaling only the first subsequent post-ReLU marginal
  variance toward a Gaussian ReLU moment target improved returned-set Fly
  comparisons without the instability of broader marginal correction. Exact
  variance matching produced the strongest noisy sample at `3.205e-7`
  adjusted / `2.831e-6` final-layer MSE / `3.082e10` effective compute over 79
  returned MLPs, but retries varied. A `1.5x` scale update gave the best clean
  mode sweep at `3.151e-7` adjusted / `2.783e-6` MSE / `3.080e10` effective
  compute, and the promoted default `make fly` proof scored `3.353e-7`
  adjusted / `2.962e-6` MSE / `3.082e10` effective compute with 80 returned and
  no failures. Rechecking 12 and 14 blocks after the strength update lost:
  12 blocks scored `3.390e-7` adjusted / `3.223e-6` MSE / `2.859e10`
  effective compute, and 14 blocks scored `3.615e-7` adjusted / `2.963e-6`
  MSE / `3.318e10` effective compute.
  Explicitly rechecking the lower block frontier after the `1.5x` strength
  update found that the score floor was not enough to offset MSE: 11 blocks
  reached the `0.1` multiplier floor but scored `3.561e-7` adjusted /
  `3.560e-6` MSE / `2.624e10` effective compute, and 12 blocks scored
  `3.589e-7` adjusted / `3.422e-6` MSE / `2.852e10` effective compute, both
  with no failures.
  Owner AICrowd 50-MLP checks later put `4301cef` ("Use first-layer variance
  match by default") at `3.12e-7` adjusted / `2.77e-6` MSE, while `21128f4`
  ("Variance-match early Hadamard layers") scored `3.44e-7` adjusted /
  `3.07e-6` MSE; treat that as external support for keeping the one-layer
  variance route ahead of broader early-layer correction, while still inside
  the documented Fly noise caveats for small deltas. The later `803d3ad`
  strength-tuned default scored `3.18e-7` adjusted / `2.84e-6` MSE on the same
  owner AICrowd 50-MLP path, supporting the `1.5x` first-successor strength as
  the current default despite noisy Fly replicates.
  Shrinking the per-coordinate variance scale halfway toward one global scalar
  produced one promising first-80 run (`3.069e-7` adjusted / `2.710e-6` MSE),
  but a replicate bounced to `3.634e-7`, and the full-100 check was neutral at
  `3.386e-7` adjusted / `2.989e-6` MSE / `3.085e10` effective compute with no
  failures. Do not promote without a stronger paired win. A positive log-space
  power correction, using `sqrt(target/sample) ** 1.5` instead of the linear
  `1.5x` scale update, also lost at `3.559e-7` adjusted / `3.135e-6` MSE /
  `3.085e10` effective compute over 76 returned MLPs, with four worker
  download failures. A global energy thermostat that scaled the whole
  first-successor centered ensemble by one total variance ratio, preserving
  correlations more aggressively than per-coordinate scaling, was also not
  enough: `1.5x` scored `3.319e-7` adjusted / `2.929e-6` MSE / `3.084e10`
  effective compute with no failures. Clipping the current per-coordinate
  first-successor scale factors around one was close but not actionable:
  a `1.5` cap scored `3.232e-7` adjusted / `2.852e-6` MSE / `3.084e10`
  effective compute, while a looser `2.0` cap scored `3.367e-7` adjusted /
  `2.974e-6` MSE / `3.081e10` effective compute, both with no failures.
  Combining the `1.5` cap with 12 blocks did not recover the lower-compute
  frontier: it scored `3.471e-7` adjusted / `3.301e-6` MSE / `2.856e10`
  effective compute over 79 returned MLPs, with one worker returncode failure.
  An ensemble-derived kurtosis gate for the first-successor variance strength,
  damping coordinates whose preactivation ensemble looked less Gaussian, landed
  near the scale-cap result but was not promotable: damping `0.25` scored
  `3.233e-7` adjusted / `2.852e-6` MSE / `3.084e10` effective compute over
  78 returned MLPs, with two worker returncode failures.
  Weighting the first-successor variance strength by the next layer's outgoing
  weight energy, intended to spend marginal correction only on coordinates with
  high downstream sensitivity, also stayed behind the default frontier:
  `3.394e-7` adjusted / `2.994e-6` MSE / `3.084e10` effective compute with no
  worker failures.
  Restoring the original first-successor correlation geometry after the useful
  marginal variance scale, by mapping the corrected ensemble to the original
  post-ReLU correlation matrix with corrected marginal variances, lost on both
  MSE and compute: `3.883e-7` adjusted / `3.148e-6` MSE / `3.353e10`
  effective compute with no worker failures.
  Clipping negative adjusted activations back to nonnegative support after the
  current first-successor variance scale also lost, indicating that the
  unconstrained centered scale's geometry is more useful than enforcing ReLU
  support at that point: `3.624e-7` adjusted / `3.193e-6` MSE / `3.085e10`
  effective compute with no worker failures.
  A monotone nonnegative power transform of the first-successor ReLU
  activations, normalized back to the original sample mean while using the
  target/sample variance ratio as a coordinatewise exponent, reached the compute
  floor but lost too much raw MSE: `3.249e-7` adjusted / `3.238e-6` MSE /
  `2.502e10` effective compute with no worker failures.
  Applying the same variance-ratio correction to centered preactivations before
  the first-successor ReLU, instead of to centered post-ReLU activations, was a
  decisive loss at `1.5x`: `5.821e-7` adjusted / `5.130e-6` MSE /
  `3.089e10` effective compute with no failures.
  Replacing the first-layer recolor matrix inverse with `fnp.linalg.solve`
  was semantically equivalent but did not produce a useful scorer-path win:
  `3.520e-7` adjusted / `3.100e-6` MSE / `3.083e10` effective compute with
  no failures, so the established inverse expression was kept.
  Halving the first-layer antithetic Hadamard matmul, by computing only the
  positive block halves and reconstructing `relu(-pre)` for the antithetic
  halves, preserves the layer-0 ensemble up to row order and floating summation
  while removing redundant first-layer FLOPs. The default `make fly` proof
  after this change scored `3.698e-7` adjusted / `3.036e-6` MSE /
  `3.224e10` effective compute over 80 returned MLPs with no failures. Raw
  `flops_mean` dropped to `2.942e10`, about `1.40e9` below the prior
  `3.082e10` effective-compute checkpoint and consistent with the expected
  layer-0 mechanism, but residual wall-time charge made the reported effective
  compute noisy and higher on this first-80 run. The adjusted-score move from
  the documented `3.353e-7` baseline is about 10% worse, inside the repository
  noise caveat, with MSE still near the prior `2.962e-6`.
  A chi-radial variant that stratifies first-layer Hadamard row radii using
  Wilson-Hilferty chi quantiles lost on the first clean Fly check:
  `4.040e-7` adjusted / `3.540e-6` MSE / `3.136e10` effective compute over
  80 returned MLPs with no failures, so it remains only `hadamard_chi`.
  The explicit 24-block diagnostic scored `3.384e-7` adjusted / `1.639e-6`
  MSE / `5.634e10` effective compute over 80 returned MLPs with no failures.
  Since the MSE moved toward the expected variance-halving value rather than
  staying near `2.9e-6`, the current 13-block route appears substantially
  variance-limited even though 24 blocks is not an adjusted-score candidate.
  Replacing only the large ensemble propagation matmuls with plain recursive
  Strassen arithmetic reduces actually executed array arithmetic without
  modifying or bypassing flopscope accounting; keep it flagged for owner
  rules-spirit review before submission use. One Strassen level at 13 blocks
  cut raw `flops_mean` to `2.612e10` and scored `3.343e-7` adjusted /
  `3.141e-6` MSE / `2.910e10` effective compute, then replicated at
  `3.283e-7` adjusted / `2.974e-6` MSE / `3.011e10` effective compute, both
  over 80 returned MLPs with no failures. The adjusted scores are close to but
  not >15% better than the documented `3.353e-7` default, so `hadamard_st1`
  was not promoted. Two Strassen levels reduced raw `flops_mean` further to
  `2.335e10`, but residual wall-time charge rose sharply and the run scored
  `3.482e-7` adjusted / `3.122e-6` MSE / `2.960e10` effective compute with no
  failures. Because L2 effective compute stayed above the `2.72e10` floor and
  residual charge exceeded 15% of raw FLOPs, no block-reinvestment or L3 run
  was taken from this implementation.
  Batching all Strassen leaves into one `fnp.einsum("brk,bkc->brc", ...)` per
  propagation matmul preserved the raw L2 Strassen FLOP count but did not fix
  the residual-time problem. An initial leaf-ordering bug had correct
  `2.335e10` raw FLOPs but broken predictions (`1.190e-1` adjusted /
  `1.119e0` MSE / `2.854e10` effective compute), so it was discarded. After
  fixing the leaf order, `hadamard_st2` scored `3.423e-7` adjusted /
  `3.205e-6` MSE / `2.949e10` effective compute with `2.335e10` raw FLOPs and
  `6.134e9` residual compute, again over 80 returned MLPs with no failures.
  Since effective compute remained above the `2.72e10` floor and residual
  charge stayed about 26% of raw FLOPs, no block reinvestment or L3 run was
  justified. The batched `hadamard_st1` check also regressed to `3.657e-7`
  adjusted / `3.225e-6` MSE / `3.100e10` effective compute with `2.612e10`
  raw FLOPs and `4.883e9` residual compute. Do not promote batched Strassen.
  Promoting the earlier recursive `st1` remains rules-spirit-review material
  because its raw arithmetic saving is deterministic, but its replicated
  adjusted-score edge over the documented default was only about 2%, not a
  >15% scoring win. The owner accepted the rules-spirit tradeoff and promoted
  recursive `hadamard_st1` as the depth-32 default; the promotion proof scored
  `3.470e-7` adjusted / `3.201e-6` MSE / `2.948e10` effective compute with
  `2.612e10` raw FLOPs over 80 returned MLPs and no failures, again showing
  deterministic raw arithmetic savings with first-80 score noise inside the
  documented caveat band.
  On 2026-07-03 the owner recalibrated Fly residual accounting from
  `0.2601` to `0.0645994832` to match AICrowd, where residual charge is only
  about 2% of score. Effective-compute numbers documented before this
  recalibration overstate residual charge by roughly 4x; in particular, the
  old L2/L3 Strassen rejections for residual-charge reasons should not be read
  as current evidence against deeper Strassen or block reinvestment. Under the
  recalibrated runner, the promoted `st1` default re-baselined at `3.197e-7`
  adjusted / `3.167e-6` MSE / `2.686e10` effective compute with `2.612e10`
  raw FLOPs over 80 returned MLPs and no failures.
  Rechecking the Strassen ladder with batched leaves found that `hadamard_st3`
  at 13 blocks scored `2.933e-7` adjusted / `2.932e-6` MSE / `2.291e10`
  effective compute with `2.114e10` raw FLOPs over 80 returned MLPs and no
  failures, which is below the `2.72e10` score floor and points to rounding up
  to 16 blocks. The first `hadamard_st4` check initially fell back to plain
  matmul because the guard rejected 16-wide leaves; after enabling those
  leaves, `hadamard_st4` at 13 blocks scored `2.746e-7` adjusted /
  `2.746e-6` MSE / `2.156e10` effective compute with `1.957e10` raw FLOPs,
  but only 73 MLPs returned before the 45-second fast-runner cutoff, with no
  failures. Reinvesting blocks gave `hadamard_st3_b16` at `2.833e-7`
  adjusted / `2.808e-6` MSE / `2.745e10` effective compute with `2.599e10`
  raw FLOPs over 80 returned MLPs and no failures. `hadamard_st4_b17` scored
  `2.644e-7` adjusted / `2.634e-6` MSE / `2.729e10` effective compute with
  `2.555e10` raw FLOPs over 22 returned MLPs and no failures, then replicated
  at `1.642e-7` adjusted / `1.632e-6` MSE / `2.733e10` effective compute with
  the same raw FLOPs over 24 returned MLPs and no failures. The replicated
  first-returned score cleared the nominal >15% promotion threshold against
  the recalibrated `st1` baseline, but the low returned counts were a real
  wall-time/selection caveat for L4 under the current Fly collection window.
  A temporary `st4_b17` default proof then returned only 21 MLPs and scored
  `2.976e-7` adjusted / `2.964e-6` MSE / `2.731e10` effective compute with
  `2.555e10` raw FLOPs and no failures, which did not corroborate a >15%
  default win. Do not promote L4 without an 80-result or otherwise
  decision-grade proof that resolves the returned-count bias. After leaving
  `st1` as the default, the final plain `make fly` proof scored `3.030e-7`
  adjusted / `2.992e-6` MSE / `2.717e10` effective compute with `2.612e10`
  raw FLOPs over 80 returned MLPs and no failures.
  Split-block cubature did not deliver the hypothesized variance cut:
  `hadamard_split2` scored `3.068e-7` adjusted / `3.016e-6` MSE / `2.711e10`
  effective compute with `2.612e10` raw FLOPs, about a 5% MSE improvement over
  the recalibrated baseline, while `hadamard_split4` regressed to `3.334e-7`
  adjusted / `3.240e-6` MSE / `2.734e10` effective compute with the same raw
  FLOPs. Both split probes returned 80 MLPs with no failures, so no split
  factor was composed into the promoted route.
  A follow-up L4 timing pass confirmed that the low return count was real
  scorer-path time, not worker failure: on the local one-MLP scorer path,
  `st1` took `13.1s` run duration with `10.4s` tracked backend time, while
  the original `st4_b17` took `30.4s` run duration with `26.1s` tracked
  backend time and remained under the 60-second wall limit. Replacing the
  monolithic 2401-leaf L4 batch with one explicit top Strassen level over
  batched L3 leaves preserved numerical equivalence to plain matmul at
  `2.5e-16` relative error and improved local `st4_b17` to `25.6s` run
  duration / `21.8s` tracked backend time. The normal 45-second Fly window
  still returned only 52 MLPs for optimized `hadamard_st4_b17`, scoring
  `2.350e-7` adjusted / `2.254e-6` MSE / `2.836e10` effective compute with
  `2.555e10` raw FLOPs and no failures, so L4 still needed a wider collection
  window for unbiased measurement.
  With `FLY_MAX_RESULT_SECONDS=90 FLY_FAST_TIMEOUT=140s`, used only to
  complete measurement collection and not as a grader-rule change,
  `hadamard_st4_b16` returned 80 clean MLPs and scored `2.895e-7` adjusted /
  `2.862e-6` MSE / `2.711e10` effective compute with `2.406e10` raw FLOPs.
  The same widened window gave `hadamard_st4_b17` 80 returned MLPs but one
  combined-budget exhaustion, scoring `3.956e-3` adjusted / `3.958e-3` MSE /
  `2.875e10` effective compute with `2.555e10` raw FLOPs. Therefore L4 is not
  the clean promotion candidate despite the attractive early-subset scores.
  The clean `hadamard_st3_b16` fallback replicated under the normal 45-second
  Fly window at `2.811e-7` adjusted / `2.785e-6` MSE / `2.737e10` effective
  compute with `2.599e10` raw FLOPs over 80 returned MLPs and no failures.
  Promote `st3_b16`: it is a deterministic Strassen/block-reinvestment win
  over the recalibrated `st1` baseline with normal-window 80-result proofs,
  while this route family appears to bottom out around `2.2e-7` to `2.4e-7`
  without a new variance-reduction mechanism. The split-block probe was
  neutral to negative, so it is not that mechanism. The first plain default
  proof after promotion had one Fly entrypoint machine failure unrelated to
  estimator behavior and returned 79 scored MLPs at `2.793e-7` adjusted /
  `2.726e-6` MSE / `2.754e10` effective compute with `2.599e10` raw FLOPs.
  A clean retry returned 80 MLPs with no failures and scored `2.921e-7`
  adjusted / `2.888e-6` MSE / `2.756e10` effective compute with `2.599e10`
  raw FLOPs, serving as the promotion proof.
  On 2026-07-03, the owner traced three AICrowd results reporting `2.99e10`
  FLOPs / `2.79e-6` MSE to wrong-file submissions of the old default, not a
  grader flopscope-accounting divergence. The first real grader result for
  HEAD `hadamard_st3_b16` is still pending.
  The deep default now chooses its Hadamard block count from the passed
  `budget` and the MLP shape instead of relying only on the contest-tuned
  `_DEEP_HADAMARD_BLOCKS` constant. The estimate applies the active Strassen
  level's `(7/8)^L` propagation discount, a fixed-overhead factor for recolor
  and reductions, and a `3%` safety margin, then floors `(budget / 10)` over
  that per-block cost with a `[1, 32]` clamp. At the contest `2.72e11` budget
  with L3 it selects 16 blocks, preserving the promoted `st3_b16` route while
  adapting to other budgets using only legitimate `predict()` inputs and MLP
  shape. The proof run scored `2.996e-7` adjusted / `2.980e-6` MSE /
  `2.734e10` effective compute with `2.599e10` raw FLOPs over 80 returned
  MLPs and no failures.
  A variance-mechanism round tested mid-depth re-antithetization and a
  first-layer exact-third-moment control variate. `mirror<K>` propagates half
  the requested rows through layer `K`, then reflects the ensemble around the
  layer-`K` mean before continuing. The cost reduction was real, but zeroing
  deep odd central moments introduced severe bias: `hadamard_st3_b16_mirror8`
  scored `1.496e-6` adjusted / `1.496e-5` MSE / `2.311e10` effective compute
  with `2.171e10` raw FLOPs; `mirror16` scored `6.912e-3` adjusted /
  `6.925e-3` MSE / `2.070e10` effective compute with `1.870e10` raw FLOPs and
  one combined-budget exhaustion; `mirror24` scored `1.095e-6` adjusted /
  `1.095e-5` MSE / `1.748e10` effective compute with `1.569e10` raw FLOPs.
  Do not reinvest mirror savings without a new way to preserve the genuine
  post-ReLU skew.
  The `cv3` final-row control variate, using first-layer blockwise third raw
  moment residuals against the exact zero-mean Gaussian ReLU target, also lost:
  `hadamard_st3_b16_cv3` scored `6.036e-7` adjusted / `5.970e-6` MSE /
  `2.751e10` effective compute with `2.603e10` raw FLOPs. Adaptive
  `hadamard_st4` picked a too-expensive L4 point under the current runner and
  was not clean: only 28 MLPs returned before the fast-window cutoff, with
  `4.593e-2` adjusted / `4.593e-2` MSE / `3.305e10` effective compute,
  `2.854e10` raw FLOPs, and one combined-budget exhaustion. No variance
  mechanism was promoted. The unchanged default proof scored `2.836e-7`
  adjusted / `2.738e-6` MSE / `2.813e10` effective compute with `2.599e10`
  raw FLOPs over 80 returned MLPs and no failures, leaving a large gap to the
  `1.6e-7` target.
  Owner AICrowd later scored the real `hadamard_st3_b16` default at
  `2.411e-7` adjusted / about `2.3e-6` MSE / `2.6e10` raw FLOPs /
  `2.86e10` effective compute, confirming that wall/residual charge, not raw
  FLOPs, is the remaining cost leak. Local one-MLP timing under the current
  runner put default L3 at `16.4s` run duration / `13.4s` flopscope backend /
  `0.51s` residual wall, while adaptive L4 took `24.0s` / `20.4s` / `0.44s`.
  The L4 adaptive block model was tightened with measured per-row costs
  (`3.17e6` for L3, `2.94e6` for L4 at width 256/depth 32) and a
  level-specific residual allowance so L3 remains 16 blocks and L4 rounds down
  to 16 blocks instead of the previous budget-exhausting point.
  With the recalibrated `0.1` Fly residual scale, the default re-baseline
  returned 79 MLPs at `2.802e-7` adjusted / `2.723e-6` MSE / `2.806e10`
  effective compute with `2.599e10` raw FLOPs and no failures. Adaptive
  `hadamard_st4` returned 80 clean MLPs at `2.717e-7` adjusted / `2.572e-6`
  MSE / `2.853e10` effective compute with `2.406e10` raw FLOPs, but residual
  remained high enough to sit above the floor. `hadamard_st4_b15` reached
  floor compute but was not clean: one combined-budget exhaustion made the run
  score `1.913e-2` adjusted / `1.914e-2` MSE despite `2.256e10` raw FLOPs.
  Retuning the first-successor variance strength at 16 blocks was noisy and
  not promotable. `hadamard_st3_b16_s125` had one combined-budget exhaustion
  and scored `5.057e-3`; `s140` was clean at `2.846e-7` adjusted /
  `2.712e-6` MSE / `2.849e10` effective compute; `s165` first looked
  promising at `2.480e-7` adjusted / `2.359e-6` MSE / `2.859e10` effective
  compute, but replicated at only `2.894e-7` adjusted / `2.768e-6` MSE /
  `2.840e10` effective compute. No strength change was promoted.
  The closeout retune resolved `s165` as another neutral bounce, consistent
  with the historical strength-sweep pattern: two more clean Fly replicates
  scored `2.747e-7` adjusted / `2.670e-6` MSE / `2.812e10` effective compute
  and `2.634e-7` adjusted / `2.544e-6` MSE / `2.819e10` effective compute.
  Across the four `s165` runs, the MSEs were `2.359e-6`, `2.768e-6`,
  `2.670e-6`, and `2.544e-6`, with only one run at or below `~2.45e-6` and a
  four-run mean of `2.585e-6`, only about `5%` below the default's
  `~2.72e-6`; therefore `_DEEP_VARIANCE_MATCH_STRENGTH` remains `1.5`.
  Full-json Fly diagnostics for the sporadic `combined_budget_exhausted`
  failures showed a harness-side residual-accounting mismatch rather than an
  estimator raw-FLOP tail. A fresh `hadamard_st3_b16_s125` capture returned
  80 clean MLPs; its slowest residual rows had raw FLOPs `2.599e10`, measured
  residual wall `0.77s`, worker WhestBench wall near `25-30s`, and scaled
  Fly-summary effective compute only `3.37e10`. A fresh `hadamard_st4_b15`
  capture reproduced one failure over 78 returned MLPs: the failed worker had
  raw FLOPs `2.256e10`, measured residual wall `1.476s`, flopscope backend
  `19.02s`, flopscope overhead `1.14s`, and worker WhestBench wall `36.56s`.
  The worker-side post-hoc check still used the local
  `--residual-wall-time-multiplier 2.0` rate, so it reported effective compute
  `3.178e11` (`2.256e10 + 2 * 1e11 * 1.476`) and zeroed the MLP, even though
  the Fly summary later rescales residual by `0.1` for AICrowd-like scoring,
  which would put the same row at only `3.73e10`. This explains why exhaustion
  could appear in runs whose reported mean effective compute was near
  `2.8e10`, and why it also appeared at lower raw FLOPs: it is a slow/loaded
  Fly worker plus local harness failure-threshold artifact, not evidence of a
  shape-dependent pathological slow path in recolor, Cholesky, or Strassen.
  Under the local `2.0` residual multiplier, a default-class `2.599e10` raw
  route trips the `2.72e11` combined budget at about `1.23s` residual wall.
  Under the AICrowd-like `0.1` residual scale used for realistic score
  summaries, the same raw route would need about `24.6s` residual wall to
  exhaust; compared with the current local default residual wall around
  `0.51s` and full one-MLP wall around `16.4s`, that is roughly a `48x`
  residual-wall margin, matching the owner's observation that AICrowd
  submissions have not exhausted.
  A local 5-MLP mini layerwise MSE profile for the default showed error
  peaking in the middle layers and then decaying: block means were
  layers 1-8 `4.94e-6`, 9-16 `5.90e-6`, 17-24 `3.81e-6`, 25-32 `2.60e-6`,
  with the final layer at `2.35e-6` on that mini set. The final unchanged
  default proof returned 80 MLPs with no failures at `2.869e-7` adjusted /
  `2.720e-6` MSE / `2.866e10` effective compute with `2.599e10` raw FLOPs.
  The final cycle frontier is: grader-confirmed `2.411e-7` adjusted at
  `st3_b16`; a realistic low-to-mid `2e-7` frontier if leaner L4 wall time can
  convert its raw-FLOP savings into AICrowd score margin; and `1.6e-7`
  unreachable without a new variance mechanism. The candidate mechanisms
  tested in this cycle, including split blocks, mirror re-antithetization,
  `cv3`, final/reporting control variates, robust aggregation, and marginal or
  covariance correction variants, were falsified or left weaker than the
  current route under the repository noise caveats.
  A follow-up L4 wall-trim round replaced the tiny L4 leaf
  `fnp.einsum("brk,bkc->brc", ...)` with batched matmul only when Strassen
  leaf rows are at most 512, preserving the established L3 einsum path and
  the same counted Strassen FLOPs. Local one-MLP profiling before the change
  put adaptive L4 at `23.1s` wall / `21.4s` backend / `0.63s` residual, with
  the 224 leaf einsums taking `11.7s`, operand adds/stacks about `7.6s`, and
  reassembly about `2.5s`. After the conditional leaf-matmul change, a
  sequential local check measured `hadamard_st4` at `14.8s` wall /
  `13.4s` backend / `0.53s` residual with unchanged `2.406e10` raw FLOPs,
  versus `hadamard_st3_b16` at `23.6s` wall / `22.6s` backend / `0.79s`
  residual in the same run. A later post-change cProfile pass on a less-loaded
  local run measured L4 at `11.5s` wall / `9.9s` backend / `0.63s` residual;
  the leaf matmuls were then only `1.7s`, leaving operand combination,
  stacking, concatenation/reassembly, and flopscope wrapper overhead as the
  remaining L4 wall bottlenecks.
  Normal-window Fly proof showed the wall-trim was not enough to promote L4:
  the default L3 baseline returned 80 clean MLPs at `2.959e-7` adjusted /
  `2.824e-6` MSE / `2.852e10` effective compute with `2.599e10` raw FLOPs,
  while adaptive `hadamard_st4` returned 80 clean MLPs at `2.911e-7`
  adjusted / `2.757e-6` MSE / `2.881e10` effective compute with `2.406e10`
  raw FLOPs. `hadamard_st4_b15` reached lower reported mean effective compute
  (`2.751e10`, `2.256e10` raw) but again had one
  `combined_budget_exhausted` row and scored `1.913e-2`, so it remains
  unusable under the current harness. Leave `st3_b16` as the default; the
  L4 leaf-kernel trim is kept as a diagnostic improvement and future
  foundation, not as a promotion.

## Pre-refactor checkpoints (recovered)

Recovered from `docs/how-to/estimator-history.md` at `88ea3f0^`. These
predate the 2026-06-28 refactor but were already on the phase-1 shape,
so they belong with the checkpoints rather than the warmup round.

### Route as of 2026-06-28 (pre-refactor snapshot)

The root estimator is now depth-aware. For the current contest shape
width-256/depth-32, unforced `predict()` uses randomized antithetic
Walsh-Hadamard sign cubature with 13 blocks, then recolors the first hidden
activation ensemble so its mean and covariance match the exact zero-mean
Gaussian ReLU moments for the first layer. Shallower MLPs still use the
optimized factorized K=3 `r=1` path.

Historical router experiments remain documented below because they are useful
for interpreting benchmark results and avoiding repeated dead ends.

### Deep Hadamard First-Covariance Route

For the current width-256/depth-32 grader shape, deep cumulant propagation is
less score-efficient than deterministic cubature through the network. The
default deep route uses randomized antithetic Walsh-Hadamard sign blocks as
input samples. After the first linear/ReLU layer, it applies an affine recolor
so the ensemble's first hidden activations match the exact zero-mean Gaussian
ReLU mean vector and covariance for `W0.T @ W0`; the recolored ensemble is then
propagated through the remaining ReLU layers.

Fly EWR 80-result sweeps with the corrected residual-compute scale found the
best adjusted-score frontier at 13 Hadamard blocks:

- 11 blocks: final-layer MSE `4.473e-6`, adjusted score `4.473e-7`,
  effective compute `2.583e10`.
- 12 blocks: final-layer MSE `3.664e-6`, adjusted score `3.786e-7`,
  effective compute `2.811e10`.
- 13 blocks: final-layer MSE `3.068e-6`, adjusted score `3.430e-7`,
  effective compute `3.041e10`.
- 14 blocks: final-layer MSE `3.458e-6`, adjusted score `4.150e-7`,
  effective compute `3.267e10`.
- 16 blocks: final-layer MSE `3.230e-6`, adjusted score `4.423e-7`,
  effective compute `3.724e10`.

Nearby first-layer moment experiments did not beat full first-covariance
recoloring. Diagonal-only mean/variance matching was cheaper but worse
(`4.387e-6` final-layer MSE, `4.387e-7` adjusted). Adding a marginal skew
correction improved that diagnostic but still lost (`3.927e-6`, `3.927e-7`).
Clipping the full recolored first-layer ensemble back to nonnegative support
was much worse (`7.244e-6`, `7.561e-7`), so preserving the exact first-layer
covariance is more valuable than preserving activation support in this route.

### 32x256 Budget Retargeting

The rules were retargeted from width-256/depth-8 with a `6.8e10` FLOP budget
to width-256/depth-32 with a `2.72e11` FLOP budget. The exact grouped `r1`
route still fits under the analytical flopscope cap on a synthetic
width-256/depth-32 smoke, but it leaves too little room for residual wall-time
charging: one local `BudgetContext` run used about `2.31e11` analytical FLOPs
and about `0.67s` residual wall time, pushing effective compute above the new
budget under the standard residual charge.

The default route is now depth-aware. For MLPs with at least 16 layers,
unforced `predict()` applies top-k rank compression to the factorized third
cumulant after every hidden nonlinearity, with cap `8 * width` (`2048` at the
new width 256). `WHEST_K3_MODE=r1` remains available as the exact grouped route
for explicit comparisons, and the existing `r1_cap<N>`/`r1_compressed`
diagnostics now extend their rank schedules across all hidden layers by
repeating the last cap instead of silently stopping after the old seven hidden
updates.

This retargeting used only freshly generated local MLPs and Monte Carlo
samples, not cached public labels. On one width-256/depth-32 synthetic check
with a 50k-sample Monte Carlo reference, the `r1_cap2048` route used about
`5.4e10` analytical FLOPs and had the best final-layer MSE among the capped
K=3 routes tested (`~2.3e-5` final-layer MSE), while keeping a wide budget
cushion for residual time. The root demo and Makefile default budget were also
updated to `2.72e11`.

The compressed K=3 route was then superseded for depth-32 by randomized
Hadamard cubature. A single Hadamard sign block contains all 256 rows of a
Sylvester Walsh-Hadamard matrix and their antithetic negations, randomly
flipped by an independent Rademacher diagonal. Each block therefore has exact
zero input mean and identity input covariance, while using ordinary sample
propagation through the nonlinear network. On cached mini smokes with the
standard `RESIDUAL_WALL_TIME_MULTIPLIER=2.0`, fixed 16-block Hadamard was far
better than the compressed K=3 retargeting but paid too much multiplier on
larger runs. Twelve blocks stayed near the 10% multiplier floor.

A first-layer covariance recoloring improved the default three-MLP target
without using labels. The first pre-activation vector is exactly zero-mean
Gaussian with covariance `W0.T @ W0`, so the post-ReLU mean and covariance are
available from the closed-form zero-mean bivariate ReLU kernel. The estimator
now linearly whitens/recolors the 12-block Hadamard first hidden ensemble to
match those exact moments, then continues ordinary sample propagation. This
uses only the passed MLP weights.

Cached-mini comparisons:

- Previous compressed K=3 depth route, one MLP: adjusted score about
  `9.23e-6`, raw final-layer MSE about `2.42e-5`.
- Plain 12-block Hadamard, five MLPs: adjusted score about `2.86e-7`, raw
  final-layer MSE about `2.86e-6`, with the multiplier at the 0.1 floor in one
  JSON detail run.
- First-cov 12-block Hadamard, three MLPs: adjusted score about `1.95e-7`,
  raw final-layer MSE about `1.84e-6`, mean multiplier about `0.105`.
- First-cov 12-block Hadamard, five MLPs: adjusted score about `2.7e-7` to
  `2.9e-7` across repeated local subprocess timings, raw final-layer MSE about
  `2.63e-6`.
- First-cov 16-block Hadamard, five MLPs: raw final-layer MSE improved to
  about `2.02e-6`, but the multiplier rose to about `0.140`, giving a worse
  adjusted score around `2.83e-7`.

Rejected follow-ups in the same pass:

- Full per-layer Gaussian marginal moment correction destroyed joint geometry;
  one-MLP adjusted scores ranged from about `4.9e-6` to `6.6e-5`.
- `H D H` rotated sign blocks, permuted/sign variants, spherical radial
  scaling, fourth-moment axis mixes, Halton/Gaussian bridge samples, and
  ordinary Rademacher samples were all worse than fixed Hadamard in smokes.
- Half-strength first-cov blending, block-trimmed means, blockwise empirical
  shrinkage, whitened Gaussian particles, and final-layer-only Gaussian
  marginal correction did not beat plain 12-block Hadamard or the full
  first-cov route on the five-MLP cached split.
