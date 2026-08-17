# Hadamard-oriented LHS gate

Rows: `100/100`; failures: `0`; checksums: `PASS`.

**FAIL**

| method | mean MSE | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current | 2.546587e-06 | 1.921486e-06 | 1.031934e-06 | 4.797766e-06 | 4.425189e-07 |
| lhs_independent | 3.836092e-06 | 2.216646e-06 | 1.067355e-06 | 8.014077e-06 | 6.353758e-07 |
| lhs_hadamard | 3.979278e-06 | 2.267082e-06 | 1.129684e-06 | 6.999369e-06 | 5.440081e-07 |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_candidate | 1.0637 | 0.7954 | 0.2854 | 2.1363 | 0.0902 |
| independent_over_candidate | 1.2734 | 1.0038 | 0.3911 | 2.3724 | 0.0903 |

## LHS diagnostics

- `lhs_independent`: `{"antipode_max_abs": {"max": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "q10": 0.0, "q90": 0.0}, "coordinate_mean_max_abs": {"max": 8.86757334228605e-08, "mean": 5.653004564010189e-08, "median": 5.492393029271625e-08, "min": 3.699824446812272e-08, "q10": 4.2590545490384105e-08, "q90": 7.373564585577696e-08}, "diagonal_second_moment_max_abs_error": {"max": 0.004083395004272461, "mean": 0.0024430274963378905, "median": 0.002433478832244873, "min": 0.0013360977172851562, "q10": 0.0017494916915893554, "q90": 0.003254044055938721}, "diagonal_second_moment_mean": {"max": 1.0000755786895752, "mean": 0.9999983912706375, "median": 0.999994158744812, "min": 0.999933660030365, "q10": 0.9999665558338166, "q90": 1.0000293970108032}, "offdiagonal_max_abs": {"max": 0.08197157829999924, "mean": 0.0671785494685173, "median": 0.06662888452410698, "min": 0.05840884894132614, "q10": 0.0623295072466135, "q90": 0.07256566286087036}, "offdiagonal_rms": {"max": 0.015738260000944138, "mean": 0.015596004948019982, "median": 0.015595475677400827, "min": 0.015476556494832039, "q10": 0.015505529008805752, "q90": 0.015670601278543472}, "strata_exact": true, "strata_unique_per_coordinate": {"max": 8192.0, "mean": 8192.0, "median": 8192.0, "min": 8192.0, "q10": 8192.0, "q90": 8192.0}}`
- `lhs_hadamard`: `{"antipode_max_abs": {"max": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "q10": 0.0, "q90": 0.0}, "coordinate_mean_max_abs": {"max": 5.309702828526497e-07, "mean": 1.0089616353070597e-07, "median": 6.126720109023154e-08, "min": 2.0314473658800125e-08, "q10": 3.0529918149113655e-08, "q90": 2.2361200535669925e-07}, "diagonal_second_moment_max_abs_error": {"max": 0.004083395004272461, "mean": 0.0024430274963378905, "median": 0.002433478832244873, "min": 0.0013360977172851562, "q10": 0.0017494916915893554, "q90": 0.003254044055938721}, "diagonal_second_moment_mean": {"max": 1.0000755786895752, "mean": 0.9999983912706375, "median": 0.999994158744812, "min": 0.999933660030365, "q10": 0.9999665558338166, "q90": 1.0000293970108032}, "offdiagonal_max_abs": {"max": 0.058580439537763596, "mean": 0.051505890376865864, "median": 0.051380665972828865, "min": 0.04589500650763512, "q10": 0.04794260747730732, "q90": 0.05504680089652539}, "offdiagonal_rms": {"max": 0.012145155109465122, "mean": 0.012033017976209521, "median": 0.012038868851959705, "min": 0.011936617083847523, "q10": 0.0119679712690413, "q90": 0.012085835821926594}, "strata_exact": true, "strata_unique_per_coordinate": {"max": 8192.0, "mean": 8192.0, "median": 8192.0, "min": 8192.0, "q10": 8192.0, "q90": 8192.0}}`

## Gate decisions

- `complete_100`: **PASS**.
- `checksums`: **PASS**.
- `no_failures`: **PASS**.
- `candidate_mse`: **FAIL**.
- `current_over_candidate_global`: **FAIL**.
- `current_over_candidate_median`: **FAIL**.
- `current_over_candidate_q10`: **FAIL**.
- `current_over_candidate_min`: **FAIL**.
- `independent_strata`: **PASS**.
- `hadamard_strata`: **PASS**.
