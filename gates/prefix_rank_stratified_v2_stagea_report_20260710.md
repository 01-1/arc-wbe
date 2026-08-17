# Prefix-rank stratified Stage A aggregate

Rows=100 valid_checksums=100 failures=[]

## K=2
candidate/current/control-GREG/raw-rank MSE = 4.75429806405265e-06 / 2.5465832617025135e-06 / 4.787134860347148e-06 / 4.755142009649674e-06
global/median/q10/min ratio = 0.535638116793157 / 0.573922821460908 / 0.2431109012158302 / 0.14339565749424113
corr median/q10 = 0.5754332851398452 / 0.5022838210986469; projected FLOPs/ratio = 22478320312.5 / 0.88671875; full-pool ceiling MSE=1.4099880781246557e-06
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_greg=PASS, beats_raw_rank=PASS, compute_anchor=PASS, predictor_corr=FAIL
K verdict: FAIL

## K=4
candidate/current/control-GREG/raw-rank MSE = 5.419247874852971e-06 / 2.5465832617025135e-06 / 5.577575638191405e-06 / 5.394768777647136e-06
global/median/q10/min ratio = 0.469914519599568 / 0.5323113202309675 / 0.19686179068541926 / 0.09689450899082197
corr median/q10 = 0.7357783007243077 / 0.6576374365644208; projected FLOPs/ratio = 21587109375.0 / 0.8515625; full-pool ceiling MSE=1.4099880781246557e-06
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_greg=PASS, beats_raw_rank=FAIL, compute_anchor=PASS, predictor_corr=FAIL
K verdict: FAIL

## K=8
candidate/current/control-GREG/raw-rank MSE = 7.0093065628892675e-06 / 2.5465832617025135e-06 / 8.575083820169155e-06 / 6.178733130482763e-06
global/median/q10/min ratio = 0.36331457881802226 / 0.37866460855928236 / 0.18529709974652897 / 0.051629510324595944
corr median/q10 = 0.8738107923560245 / 0.6957402792238162; projected FLOPs/ratio = 22181250000.0 / 0.875; full-pool ceiling MSE=1.4099880781246557e-06
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_greg=PASS, beats_raw_rank=FAIL, compute_anchor=PASS, predictor_corr=FAIL
K verdict: FAIL

## K=12
candidate/current/control-GREG/raw-rank MSE = 0.00552232979697223 / 2.5465832617025135e-06 / 0.0006591551923868985 / 3.515090712878844e-05
global/median/q10/min ratio = 0.0004611429152780314 / 0.06417722106621176 / 0.03219138958761947 / 7.55867670515566e-06
corr median/q10 = 0.9328414719900047 / 0.7268796850181243; projected FLOPs/ratio = 21983203125.0 / 0.8671875; full-pool ceiling MSE=1.4099880781246557e-06
Gates: 100_checksums=PASS, candidate_mse=FAIL, global_ratio=FAIL, median_ratio=FAIL, q10_ratio=FAIL, min_ratio=FAIL, beats_random_greg=FAIL, beats_raw_rank=FAIL, compute_anchor=PASS, predictor_corr=PASS
K verdict: FAIL

Passing K: []
Overall verdict: FAIL

## Launch metadata audit

The launched computation used source constants `POOL_BLOCKS=32`,
`N_PAIRS=8192`, and selected-main counts `{K2:2816, K4:2304, K8:1536,
K12:256}`. The emitted per-shard config string from the pre-correction
payload said `"27 independent full positive bases"`; this was metadata-only.
The source has been corrected to report 32. No scientific arrays or selected
counts from the launch were changed.
