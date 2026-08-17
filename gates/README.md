# Gate artifacts

Each gate is a preregistered, single-shot, pass/fail test — see the
[repository README](../README.md#what-a-gate-is) for what that means and what
each file suffix holds. Verdicts are extracted from the reports where present;
the authoritative record for any gate is its raw `*_fly.jsonl` rows, not its
prose.

50 filename groups, 246 files. These group by filename stem, so some variants of
one experiment appear separately; the [gate re-audit](../GATE_REAUDIT.md) counts
41 distinct artifact-backed gates here, plus 30 more that left only a history
entry.

| experiment | files | verdict |
|---|---:|---|
| [`active_subspace_reflection`](active_subspace_reflection_gate_20260710.md) | 7 |  |
| `active_subspace_reflection_fast` | 3 |  |
| [`alias_pretest`](alias_pretest_20260705.md) | 3 |  |
| [`angular_importance`](angular_importance_gate_20260709.md) | 7 |  |
| [`antithetic_lhs`](antithetic_lhs_v1_prereg_20260710.md) | 7 | FAIL |
| [`block_predictability`](block_predictability_gate_20260707.md) | 6 |  |
| [`bq_nscaling`](bq_nscaling_20260706.md) | 3 |  |
| [`collapse`](collapse_gate_20260706.md) | 3 |  |
| [`contraction`](contraction_gate_20260706.md) | 8 | INCONCLUSIVE |
| [`cross_output_eb`](cross_output_eb_prereg_20260710.md) | 8 | FAIL |
| [`cross_output_rowcf`](cross_output_rowcf_prereg_20260710.md) | 8 | PASS |
| [`filament_stage1`](filament_stage1_20260706.md) | 3 |  |
| [`final_weight_collapse_scout`](final_weight_collapse_scout_20260710.md) | 3 |  |
| `fingerprint_experiment` | 1 |  |
| `fingerprint_experiments` | 1 |  |
| [`fingerprint_theory`](fingerprint_theory_20260705.md) | 1 |  |
| [`folded_zca_qmc`](folded_zca_qmc_v1_prereg_20260710.md) | 7 | FAIL |
| [`gaussian_sum_pretest`](gaussian_sum_pretest_20260705.md) | 3 |  |
| [`haar_sphere_foldcv`](haar_sphere_foldcv_gate_20260710.md) | 7 | FAIL |
| [`hadamard_lhs`](hadamard_lhs_v1_prereg_20260710.md) | 7 | FAIL |
| [`k4_ladder`](k4_ladder_20260706.md) | 3 |  |
| [`layer1_rank_gauss`](layer1_rank_gauss_v1_prereg_20260710.md) | 7 | FAIL |
| `measure_r1_bias` | 1 |  |
| [`nngp_design_pretest`](nngp_design_pretest_20260705.md) | 3 |  |
| [`odd_lr_transport`](odd_lr_transport_v1_20260710_prereg.md) | 7 | FAIL |
| [`odd_rb_k8`](odd_rb_k8_v1_prereg_20260710.md) | 7 | FAIL |
| [`packed`](packed_gate_cluster_pruning_gate_20260709.md) | 11 | FAIL |
| `pair` | 1 |  |
| [`paired_block_crn`](paired_block_crn_gate_20260707.md) | 1 | FAIL |
| [`prefix_rank_stratified`](prefix_rank_stratified_prereg_20260710.md) | 8 | FAIL |
| [`prefix_rank_stratified_superseded_27block_draft`](prefix_rank_stratified_superseded_27block_draft_20260710.md) | 1 |  |
| [`prefix_vector_greg`](prefix_vector_greg_v3_prereg_20260710.md) | 7 | FAIL |
| [`r1_bias_measurement`](r1_bias_measurement_20260705.md) | 2 |  |
| [`readout_smoothing`](readout_smoothing_gate_20260706.md) | 5 |  |
| `readout_smoothing_dummy_estimator` | 1 |  |
| [`region_granularity`](region_granularity_gate_20260707.md) | 5 |  |
| `results` | 1 |  |
| [`screen`](screen_report.md) | 2 |  |
| [`sobol_gaussian`](sobol_gaussian_v1_prereg_20260710.md) | 7 | FAIL |
| [`sobol_jacobian_lt`](sobol_jacobian_lt_prereg_20260710.md) | 14 | FAIL |
| [`sobol_runtime_feasibility`](sobol_runtime_feasibility_report_20260710.md) | 1 |  |
| `sobol_runtime_feasibility_check` | 1 |  |
| `sobol_runtime_feasibility_generator` | 1 |  |
| [`sobol_sphere`](sobol_sphere_gate_20260710.md) | 14 | FAIL |
| [`sobol_triangular_lt`](sobol_triangular_lt_gate_20260710.md) | 12 | FAIL |
| [`spherical_stein`](spherical_stein_gate_20260710.md) | 7 | FAIL |
| [`spline_conditional_readout`](spline_conditional_readout_gate_20260707.md) | 6 | FAIL |
| [`suffix_cv_pretest`](suffix_cv_pretest_20260705.md) | 3 |  |
| [`tail_projection_proxy`](tail_projection_proxy_gate_20260707.md) | 5 |  |
| [`terminal_mixture_readout`](terminal_mixture_readout_gate_20260710.md) | 6 |  |
