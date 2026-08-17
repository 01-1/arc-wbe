# 2026-08-10 Block-scaling closure and leaderboard calibration

No estimator change. The default remains `hadamard_st3_b16`. `b64` was run as
a mode-gated diagnostic only.

- **Four-point block ladder: the route has no measurable bias floor.** A
  full-100 JSON Fly run of `hadamard_st3_b64` (`WALL_TIME=240`,
  `--max-result-seconds 300`, `--residual-wall-time-multiplier 0.1`,
  `--min-results 100`, no `--summary-only`) returned `97/100` with zero
  estimator failures; the three missing shards were Fly capacity errors
  (`000062:408`, `000057:408`, `000091:1`). Log:
  `paired_fly_logs/b64_full_json.log`. Paired on the `97` common MLPs against
  `b8_full_json.log`, `default_a_full_json.log`, and `b32_full_json.log`:

  | blocks | samples | mean final-layer MSE | s.e. |
  |---|---|---|---|
  | 8 | 4,096 | `5.460e-6` | `6.42e-7` |
  | 16 | 8,192 | `2.667e-6` | `2.89e-7` |
  | 32 | 16,384 | `1.501e-6` | `1.48e-7` |
  | 64 | 32,768 | `7.086e-7` | `7.49e-8` |

  The observed `b64` value came in *below* even the no-floor prediction
  (`7.720e-7`), and `3.2`/`3.7` sigma below the floor-bearing alternatives
  fitted pre-`b64` (`F + c/B + d/B^2` predicted `9.479e-7`; free-exponent
  predicted `9.831e-7`). Refits on four points, now with one residual degree
  of freedom: `p=1` gives `F = 6.72e-8` (bootstrap 90% `[-1.20e-7, 2.44e-7]`,
  `P(F<0) = 0.27`); free exponent gives `p = 1.039` (90% `[0.840, 1.240]`);
  quadratic gives `F = 1.32e-7` (90% `[-6.40e-8, 3.37e-7]`). `SSres` is
  `1.70e-14`/`1.58e-14`/`1.47e-14`, so extra parameters buy nothing. Verdict:
  the block-independent component is **bounded below `~1.5e-7` and consistent
  with zero**, and the exponent is identified at `1.039 +/- 0.20`. This
  confirms the 2026-07-06 CORRECTION and closes the block-scaling question.

- **Adjusted score is invariant to block count.** `MSE x effective_compute` is
  conserved across the ladder: `7.51e4 FLOP` at `b16` (`2.667e-6` x
  `2.815e10`) and `7.45e4 FLOP` at `b64` (`7.086e-7` x `1.051e11`). Adjusted
  score is correspondingly flat: `2.75e-7` at `b16` versus `2.74e-7` at `b64`
  at `3.7x` the compute. Consequence: **no block-count experiment can move the
  score**, and every past block sweep was structurally incapable of doing so.
  Only the constant `V` matters.

- **RETRACTED: two of the three standing "bias floor" numbers were never
  measurements of this route.** The `6.0066460386e-7` figure is
  `mixture_squared_bias_proxy` in
  `terminal_mixture_readout_gate_20260710_results.json` — the rejected
  two-Gaussian EM readout candidate, not the baseline. The `9.819731e-7`
  figure is the `haar_cv` row in `haar_sphere_foldcv_gate_20260710_report.md`
  — the rejected Haar-sphere/Stein fold-CV estimator. Recomputing the default
  route's own replicate decomposition from the raw per-replicate estimates in
  `terminal_mixture_readout_gate_20260710_fly.jsonl` against the truth bank
  gives squared-bias proxy `3.546539e-7` (90% `[1.85e-7, 5.32e-7]`), which
  reproduces `baseline_three_rep_mean_mse` `1.128943e-6` and
  `baseline_pair_variance` `2.322867e-6` exactly. There was never a 3x
  disagreement between three measurements of one quantity.

  FURTHER CORRECTION (2026-08-11 gate re-audit,
  [`GATE_REAUDIT.md`](../GATE_REAUDIT.md)):
  the `9.819731e-7` figure is not a bias estimate at all. The Haar fold-CV,
  spherical-Stein, and Sobol-sphere-v2 aggregators share a decomposition bug:
  they label `M3` (the MSE of the three-replicate mean) as `bias_squared`, and
  the population variance of three estimates as the full per-replicate
  variance. For three replicates the correct quantities are
  `sigma^2 = (3/2)*(M1 - M3)` and `b^2 = M1 - sigma^2 = (3/2)*M3 - (1/2)*M1`.
  Corrected, `haar_cv` has `b^2 = 1.27388e-7`, spherical Stein has
  `b^2 = -1.98e-8` (consistent with zero, so its `stein_bias: FAIL` was an
  artifact of the aggregator), and Sobol-sphere v2 has `b^2 = 5.05465e-8`.

  **The bank cannot measure this route's bias.** The three gates now give the
  *same* default route at b16 a corrected `b^2` of `3.54656e-7` (Haar),
  `5.37199e-8` (Sobol-sphere v2), and `3.82e-8` (Stein) — a `9x` spread, with
  two values below the bank's own `~1.1e-7` truth floor. Treat `3.546539e-7`
  as a noise draw, not a measurement. The only decision-grade bias bound for
  this route is the four-point Fly ladder above (`<1.5e-7`, consistent with
  zero), which uses `N=1e9` truth and no bank data.

- **2026-08-11 gate re-audit under the corrected scoring model: negative.**
  Every archived gate was re-examined against the three results above (no bias
  floor, compute-invariant score, only `V` and `alpha` matter). Report:
  [`GATE_REAUDIT.md`](../GATE_REAUDIT.md).
  No archived candidate demonstrates a promotion-grade lower `V` or a measured
  `alpha > 1`. Every successful QMC/LHS gate ran at a single sample count, so
  no exponent can be fitted and no QMC rejection can be reversed on an
  exponent argument; the three v1 Sobol attempts returned zero scientific rows
  (SciPy import failures). Four candidates show truth-independent replicate
  variance below baseline: odd-state Rao-Blackwell (`0.5643x`, killed by
  `b^2 = 2.56e-4`, ~2000x its allowance), IID-sphere recolor (`0.9678x`,
  optimistic `V* = 7.258e4`), Sobol-sphere v2 (`0.9967x`, `V* = 7.476e4`), and
  terminal mixture (`0.9987x`, `V* = 7.490e4`). The last three are within
  noise of the current `7.5e4` and have unmetered candidate overhead, so their
  `V*` is optimistic in the wrong direction. Useful framing from the audit:
  at full budget a variance saving of `x%` tolerates only a specific `b^2`
  before it stops paying — `<8.88e-9` for IID-sphere, `<9.00e-10` for
  Sobol-sphere, `<3.65e-10` for terminal mixture. Verdict: the archive does not
  justify reviving any rejected family. At most it justifies one truth-clean,
  metered, multi-count measurement of the IID-sphere/Sobol-sphere recolor to
  close the last variance-only ambiguity. Otherwise effort goes to unsearched
  families.

- **Truth-noise accounting.** `docs/concepts/ground-truth.md` is authoritative:
  leaderboard and public-release datasets bake ground truth at `N = 1e9` with
  `avg_variance ~= 0.18`, giving a truth-side MSE floor of `~2e-10`. The Fly
  dataset ladder is therefore truth-clean at the `1e-7` scale under discussion.
  A local diagnostic on research seeds 11/22/33 measured
  `Var_x[relu(z_final)]` at `0.0556`/`0.1415`/`0.0880`, bracketing the
  documented `0.18` and independently corroborating it. The locally generated
  `analysis/truth_bank/` is a different matter: at `~1.64e6` samples its truth
  floor is `0.18/1.64e6 = 1.1e-7`, i.e. `~31%` of the `3.55e-7` proxy above.
  **Do not use the local truth bank for MSE or bias estimation**; it exists for
  research that needs raw activation samples Fly cannot return.

- **Reduction-factor framing, and the corrected size of the gap.** Since
  `adjusted = MSE * C / budget`, variance reduction over a sampling baseline
  depends only on the adjusted score. The public leaderboard's
  `vs Sampling` column is exactly `7.0e-7 / adjusted` (verified constant to
  `sd 5.1e-9` across all 100 rows in `top100.csv`); a pure-naive-MC baseline
  would give `2.776e-6 / adjusted`, a fixed `3.97x` apart. On the leaderboard's
  own scale the current route sits at **`2.56x`**, against rank 4 at `13x`,
  the top-100 median at `4.50x`, and rank 100 at `2.80x`. **The gap to the
  best honest entry is `5.1x`, and to the top-100 median `1.76x`.** This
  supersedes the `~1.7x` uniform-edge estimate in the 2026-07-06 corrected
  competitor read, which was computed against the pack's bottom.

- **The field is collectively variance-limited.** Across `top100.csv` the
  compute multiplier has median `0.470` (p25 `0.279`, p75 `0.640`, max
  `0.881`); only `6 of 100` sit at the `0.1` floor. Under `MSE = F + V/C` the
  score is `(V + F*C)/budget`, so any entrant with a real bias floor should
  pin to `0.1`. The broad unclustered scatter is the signature of indifference,
  i.e. `alpha ~= 1` field-wide — the same exponent measured directly above.
  Superlinear variance reduction was considered as an explanation for
  above-floor operating points and is **rejected** at population scale.

- **The top 3 are detached from the distribution.** Consecutive adjusted-score
  ratios are `1->2: 15.00x`, `2->3: 12.27x`, `3->4: 2.93x`, then `1.00x-1.15x`
  for ranks 4 through 20. The entire spread of ranks 4-100 is `4.6x`
  (`13x` down to `2.8x`), so the rank-1-to-rank-2 gap alone exceeds the total
  spread of the other 97 entrants. Rank 1 is `1.00e-10` adjusted / `1.00e-9`
  MSE at multiplier `0.100`, which is `5x` above the `2e-10` truth floor and
  therefore not arithmetically impossible — but it requires `27,800x`
  reduction versus rank 4's `51x`, and matching a `4.19e15`-FLOP label
  computation to within `5x` using `2.72e10` FLOPs.

- **Leaderboard feedback is an information channel (detail withheld).** Public
  per-instance scoring feedback is, in principle, invertible with enough
  submissions. The derivation and the per-entrant submission counts were
  recorded here on 2026-08-10 and removed on 2026-08-17: publishing a working
  method for recovering evaluation labels is not defensible during an active
  competition, and it was never load-bearing for any conclusion in this file.
  Reported privately if it matters.

- **Per-participant leaderboard analysis (removed 2026-08-17).** An entry here
  profiled identifiable competitors — per-MLP and per-layer error profiles,
  inferred mechanisms, own-error separations, identity correlations between
  entries, and searches of their public repositories. Removed before
  publication. Pseudonyms would not have helped: the exact scores quoted
  re-identify each entrant against the public leaderboard. No other entry
  depends on it.
- **Public-50 identity: UNRESOLVED.** Whether the 50 public leaderboard MLPs
  overlap the published `aicrowd/arc-whestbench-public-2026` splits could not
  be determined. The published splits are `mini` (100) and `full` (1000);
  neither is 50, and `docs/concepts/ground-truth.md` says leaderboard datasets
  are baked by "the same process as the public release", implying a separate
  instance. A per-MLP difficulty-fingerprint test was attempted and is
  **underpowered**: among entrants known to share the same 50 MLPs, log-MSE
  correlations between entrants are mostly null (three of the four pairs
  examined fall between `-0.07` and `+0.05`; one reaches `+0.61`), because
  per-MLP error is dominated by each method's own variance realization. Our
  `b16` rows reach `+0.411` against one entrant and are near-null against the
  rest, versus a shuffle null of `|r| < 0.288`. Settling this needs the HF
  dataset plus a construction the fingerprint test cannot supply.

- **Standing target.** The route is variance-limited with no bias floor, at
  `2.56x` versus sampling. Closing `1.76x` reaches the top-100 median;
  closing `5.1x` reaches rank 4. Since block count is score-invariant and the
  anchored-CV, angular/BQ, cumulant, and mixture lanes are closed, the
  remaining candidates are a better terminal readout or a mechanism outside
  the searched families. Artifacts: `paired_fly_logs/b64_full_json.log`,
  `top100.csv`.
