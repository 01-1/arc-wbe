# Cross-output row-CF Stage A aggregate

- Rows returned: 100
- Valid checksums: 100
- Rep rows: 100
- Current MSE mean: 2.7185386361844027e-06
- Candidate MSE mean: 3.248758631072819e-06
- Global current/candidate ratio: 0.836793047714559
- Per-MLP ratio mean/median/q10/min: {'mean': 0.8020616407042777, 'median': 0.8143415291215118, 'q10': 0.638139562309973, 'q90': 0.9606406372900989, 'min': 0.5136031618213214, 'max': 1.0138175739429014}
- Lambda mean/median/q10/min/max: {'mean': 0.8537102018201519, 'median': 0.9841253539907842, 'q10': 0.5648585829161865, 'q90': 1.0, 'min': 0.23348475731686685, 'max': 1.0}
- Lambda A stats: {'mean': 0.8526184571046187, 'median': 0.9979495721801952, 'q10': 0.5637011207244389, 'q90': 1.0, 'min': 0.23847441240145037, 'max': 1.0}
- Lambda B stats: {'mean': 0.8548019465356852, 'median': 1.0, 'q10': 0.5684381574396412, 'q90': 1.0, 'min': 0.22849510223228334, 'max': 1.0}
- Predictor discrepancy stats: {'mean': 1.1584376352762717e-05, 'median': 8.747640858416266e-06, 'q10': 3.654421771594822e-06, 'q90': 2.2764370137138933e-05, 'min': 2.0575632168191617e-06, 'max': 7.283143405916121e-05}
- Three-rep squared-bias proxy mean: None

## Gates

- `100_checksums`: **PASS**
- `candidate_mse`: **FAIL**
- `global_ratio`: **FAIL**
- `median_ratio`: **FAIL**
- `q10_ratio`: **FAIL**
- `minimum_ratio`: **FAIL**
- `mean_lambda_range`: **PASS**

**Verdict: FAIL**
