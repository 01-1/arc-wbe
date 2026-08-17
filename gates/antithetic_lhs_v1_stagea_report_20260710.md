# Antithetic Gaussian-LHS v1 Stage A

Rows=100 valid_checksums=100 failures=[]
Current/LHS/IID MSE=2.5465865670876637e-06 / 4.003497837267373e-06 / 3.7688097480000153e-06
Current/LHS global ratio=0.6360904065895191; per-MLP median/q10/min=0.8001586907829001 / 0.2463703204532904 / 0.10568210225283173
Current/IID global ratio=0.6757004830076802; per-MLP median/q10/min=0.8325192717137746 / 0.27934933112655475 / 0.09761544391108903
IID/LHS global ratio=0.9413792391536431; per-MLP median/q10/min=0.8787488496555391 / 0.36688663233597507 / 0.17297958096791882
LHS diagnostics={'exact_strata_all': True, 'antipode_max_abs': {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}, 'max_full_coordinate_mean_abs': {'mean': 1.734723475976807e-20, 'median': 0.0, 'min': 0.0, 'max': 1.734723475976807e-18}, 'radius_mean': {'mean': 15.98438315771478, 'median': 15.984391742596952, 'min': 15.983365355522665, 'max': 15.985394248735204}, 'radius_median': {'mean': 15.977635562432411, 'median': 15.977329026116333, 'min': 15.955041508293217, 'max': 15.999416610189268}, 'radius_min': {'mean': 13.45272371324312, 'median': 13.47120498054187, 'min': 12.92741700581091, 'max': 13.880113571590833}, 'radius_max': {'mean': 18.592953660180584, 'median': 18.555300465782928, 'min': 18.201884165175166, 'max': 19.493063314193382}, 'diag_second_moment_min': {'mean': 0.999305335930311, 'median': 0.9993017254916108, 'min': 0.9991945178236878, 'max': 0.9994089216654695}, 'diag_second_moment_median': {'mean': 0.9998789568342877, 'median': 0.99988053385262, 'min': 0.9997867647908333, 'max': 0.9999566183885419}, 'diag_second_moment_max': {'mean': 1.0024462635215745, 'median': 1.0023381070833701, 'min': 1.001572307890527, 'max': 1.0045310419298836}, 'cov_frobenius_relative_error': {'mean': 0.24954835145179907, 'median': 0.24954199301880672, 'min': 0.2473412815683245, 'max': 0.25306481147969484}, 'cov_max_abs_offdiag': {'mean': 0.0668428568636255, 'median': 0.06661188184369418, 'min': 0.05964999085542257, 'max': 0.08110791401033035}}
IID diagnostics={'antipode_max_abs': {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}, 'max_full_coordinate_mean_abs': {'mean': 1.734723475976807e-20, 'median': 0.0, 'min': 0.0, 'max': 1.734723475976807e-18}, 'radius_mean': {'mean': 15.984773626337292, 'median': 15.98594560853204, 'min': 15.956840560015074, 'max': 16.011264116604746}, 'radius_median': {'mean': 15.980004596050051, 'median': 15.980819937835605, 'min': 15.946885186759346, 'max': 16.0112891317357}, 'radius_min': {'mean': 13.473756347478082, 'median': 13.4717001658845, 'min': 12.892634734445746, 'max': 13.892354875234986}, 'radius_max': {'mean': 18.60394316129561, 'median': 18.544961578459652, 'min': 18.216147097013078, 'max': 19.31715761664561}, 'diag_second_moment_min': {'mean': 0.9366642977341882, 'median': 0.9387578685676621, 'min': 0.9136687126218836, 'max': 0.9536774835094992}, 'diag_second_moment_median': {'mean': 0.9998545500987177, 'median': 0.9996846341115879, 'min': 0.9959280610728889, 'max': 1.0036015138610215}, 'diag_second_moment_max': {'mean': 1.0642805565594262, 'median': 1.0630269424280305, 'min': 1.0477659435856317, 'max': 1.0989420435919177}, 'cov_frobenius_relative_error': {'mean': 0.2505904653430351, 'median': 0.2506023050739613, 'min': 0.24793166321444385, 'max': 0.25269342557605773}, 'cov_max_abs_offdiag': {'mean': 0.0667013687351614, 'median': 0.06617614535632166, 'min': 0.05933958430491471, 'max': 0.08754963053674651}}

Gates:
- 100_checksums: PASS
- lhs_mse: FAIL
- global_ratio: FAIL
- median_ratio: FAIL
- q10_ratio: FAIL
- min_ratio: FAIL

Verdict: FAIL
