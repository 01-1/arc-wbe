# Prefix vector-GREG v3 Stage A aggregate

Rows=100 valid_checksums=100 failures=[]

## K=4
MSE candidate/current/random-GREG/raw-rank = 4.797990519327577e-06 / 2.5465748871623376e-06 / 7.300783671538012e-06 / 4.9250727457635e-06
ratio global/median/q10/min = 0.5307586325783803 / 0.5160123404082375 / 0.23643774290847594 / 0.06756538227978749
full-main R2 median/q10 = 0.34106543385416843 / 0.17141474567194812; projected raw FLOPs mean/max=24014181342.586437 / 24014181342.58644; dense ratio=0.8515625
full-pool ceiling MSE=1.409988943185083e-06; regression breakdown={'main_predictor_matmul': 2013265920.0, 'pilot_final_svd_top_pc': 67108864.0, 'pilot_xtx': 134217728.0, 'pilot_xty': 134217728.0, 'rhs_factorization_and_triangular_solves': 44739242.66666668, 'standardize_pca_sort_reduce': 33522484.919774264, 'total': 2427071967.5864406}
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_vector_greg=PASS, beats_raw_rank=PASS, compute_anchor=PASS, full_main_r2_median=FAIL, full_main_r2_q10=FAIL, three_rep_bias_proxy=PASS
K verdict: FAIL

## K=6
MSE candidate/current/random-GREG/raw-rank = 3.709559395785539e-05 / 2.5465748871623376e-06 / 1.6641522514423573e-05 / 5.662405626503204e-06
ratio global/median/q10/min = 0.0686489853769567 / 0.44239815811371175 / 0.18452415221043847 / 0.00042844841150330485
full-main R2 median/q10 = 0.4105835000253692 / -0.3389827685847456; projected raw FLOPs mean/max=23519064155.086437 / 23519064155.08644; dense ratio=0.83203125
full-pool ceiling MSE=1.409988943185083e-06; regression breakdown={'main_predictor_matmul': 2013265920.0, 'pilot_final_svd_top_pc': 67108864.0, 'pilot_xtx': 134217728.0, 'pilot_xty': 134217728.0, 'rhs_factorization_and_triangular_solves': 44739242.66666668, 'standardize_pca_sort_reduce': 33522484.919774264, 'total': 2427071967.5864406}
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_vector_greg=FAIL, beats_raw_rank=FAIL, compute_anchor=PASS, full_main_r2_median=FAIL, full_main_r2_q10=FAIL, three_rep_bias_proxy=PASS
K verdict: FAIL

## K=8
MSE candidate/current/random-GREG/raw-rank = 6.69932204978463e-06 / 2.5465748871623376e-06 / 1.153636455836515e-05 / 6.224118079379658e-06
ratio global/median/q10/min = 0.380124267536027 / 0.4105395070025452 / 0.1807693287140112 / 0.04776547049050939
full-main R2 median/q10 = 0.4644246414945532 / -1.3631182941298097; projected raw FLOPs mean/max=24608321967.586437 / 24608321967.58644; dense ratio=0.875
full-pool ceiling MSE=1.409988943185083e-06; regression breakdown={'main_predictor_matmul': 2013265920.0, 'pilot_final_svd_top_pc': 67108864.0, 'pilot_xtx': 134217728.0, 'pilot_xty': 134217728.0, 'rhs_factorization_and_triangular_solves': 44739242.66666668, 'standardize_pca_sort_reduce': 33522484.919774264, 'total': 2427071967.5864406}
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_vector_greg=PASS, beats_raw_rank=FAIL, compute_anchor=PASS, full_main_r2_median=FAIL, full_main_r2_q10=FAIL, three_rep_bias_proxy=PASS
K verdict: FAIL

Passing K: []
Overall verdict: FAIL
