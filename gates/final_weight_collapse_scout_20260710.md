# Final-weight collapse oracle-ceiling scout

This is a read-only, optimistic oracle ceiling. The layer-30 bank truth is an illegal feature for estimator promotion. A negative result closes the ceiling question; a positive result only authorizes a later legal Fly gate.

## Frozen closeout rule

`{"close_if_every_noisy_smoother": "baseline_M1/blend_M1 < 1.15 OR blend_M1 > 2.2e-6", "or_if_every_oracle_ceiling": "truth-response held-out M1 > 1.6e-6", "otherwise": "INCONCLUSIVE", "survives_if_any": "noisy blend_M1 <= 1.8e-6 AND gain >= 1.35 AND truth-response held-out M1 <= 1.2e-6"}`

## Verdict: **CLOSE**

frozen closeout condition satisfied

Integrity: `{"all_checks_pass": true, "baseline_mse_matches": 300, "checksum_valid": 100, "finite_rows": 100, "rows_100": true, "zero_duplicates": true, "zero_failures": true, "zero_pending": true}`
Baseline: `{"M1": 2.6775205219117653e-06, "per_mlp_M1": {"max": 8.641517755022472e-06, "mean": 2.6775205219117653e-06, "median": 2.3232024826789396e-06, "min": 7.802582163759293e-07, "q10": 1.2455732472374145e-06, "q90": 4.779991186844018e-06}, "per_mlp_three_rep_mean_MSE": {"max": 5.483233629013352e-06, "mean": 1.1289427473988804e-06, "median": 7.068018870320309e-07, "min": 2.0422345342058414e-07, "q10": 3.191695197090287e-07, "q90": 2.4330033886308196e-06}, "three_rep_mean_MSE": 1.1289427473988804e-06}`

## Smoother results

### polynomial_ridge

- Direct prediction M1: `0.0752355289095`.
- Truth-oracle global lambda: `0.000218422481643`.
- Noisy-response truth-oracle global blend M1: `2.67392971579e-06`; three-rep-mean MSE: `1.12583583764e-06`.
- Baseline/blend global gain: `1.00134289473`.
- Truth-response held-out approximation ceiling M1 (illegal/oracle-only): `0.0752443156273`.
- Per-MLP baseline/blend ratio: `{"max": 1.021450119635715, "mean": 1.0011608243810486, "median": 1.001805035023469, "min": 0.9743535059840797, "q10": 0.9880054222261779, "q90": 1.0136369837651418}`.
- Per-MLP baseline/oracle-ceiling ratio: `{"max": 0.0002265665901358511, "mean": 5.216142587841662e-05, "median": 4.551562834430126e-05, "min": 6.520753989364736e-06, "q10": 2.011995912494086e-05, "q90": 8.884264622295743e-05}`.
### rbf_kernel_ridge

- Direct prediction M1: `0.00690206805441`.
- Truth-oracle global lambda: `0.000731972439346`.
- Noisy-response truth-oracle global blend M1: `2.67381852173e-06`; three-rep-mean MSE: `1.12681362761e-06`.
- Baseline/blend global gain: `1.00138453682`.
- Truth-response held-out approximation ceiling M1 (illegal/oracle-only): `0.0069039134488`.
- Per-MLP baseline/blend ratio: `{"max": 1.0271114366261476, "mean": 1.0013326664262416, "median": 1.001361292275404, "min": 0.9701714231097083, "q10": 0.9932706153980644, "q90": 1.0104026551996832}`.
- Per-MLP baseline/oracle-ceiling ratio: `{"max": 0.0037241075636931546, "mean": 0.0007468729700414245, "median": 0.0006316245585558201, "min": 5.0743607354028356e-05, "q10": 0.0001427562152055094, "q90": 0.0014863467179142855}`.
### knn16

- Direct prediction M1: `0.0585542204194`.
- Truth-oracle global lambda: `0.000681749638577`.
- Noisy-response truth-oracle global blend M1: `2.65026962877e-06`; three-rep-mean MSE: `1.10326942452e-06`.
- Baseline/blend global gain: `1.01028230971`.
- Truth-response held-out approximation ceiling M1 (illegal/oracle-only): `0.0586045574341`.
- Per-MLP baseline/blend ratio: `{"max": 1.165818618872078, "mean": 1.0102181895324973, "median": 1.006635960012292, "min": 0.8696302128341944, "q10": 0.9321810023201661, "q90": 1.0912712583321198}`.
- Per-MLP baseline/oracle-ceiling ratio: `{"max": 0.0001898532047746844, "mean": 5.9644929159369745e-05, "median": 5.3154036200497366e-05, "min": 1.2052113481957417e-05, "q10": 2.458262027351655e-05, "q90": 0.00010004290588370479}`.

The three smoothers were fixed in advance: cubic polynomial ridge on standardized `[a,u,logq]`, Gaussian RBF kernel ridge on standardized `[u,logq]`, and uniform 16-nearest-neighbor regression. Output folds are fixed by `j mod 4`; no truth-based model selection was used.

No estimator was generated or scored locally, and no Fly run was launched.
