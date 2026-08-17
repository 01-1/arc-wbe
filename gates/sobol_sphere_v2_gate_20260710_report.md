# Sobol-sphere recolor v2 gate

## Fly execution record

Frozen command:

```text
make fly-payload FLY_MLPS=100 FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/sobol_sphere_v2_manifest_20260710.json FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/sobol_runtime_feasibility_generator_20260710.py paired_fly_logs/fingerprint_theory/sobol_sphere_v2_payload_20260710.py analysis/truth_bank/truth_bank.npz" FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/sobol_sphere_v2_gate_20260710_fly.jsonl FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```

The sandbox-only upload attempt failed before launch; the one permitted
packaging retry succeeded. Fly returned `100/100` shards, with `0` failures,
`0` pending, and `100` JSONL rows. No runtime retry or parameter change was
made.

The v1 gate produced zero estimates because the stable image lacked SciPy. This v2 run uses the reviewed dependency-free generator, validated against SciPy 1.16.2 at the Sobol-uniform level with maximum error 0.0.

Successful shards: `100/100`; checksums: `PASS`.

**FAIL**

| method | mean MSE | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current | 2.659818e-06 | 2.3171360190544783e-06 | 1.4109927993316066e-06 | 4.687421061100205e-06 | 5.830496079638551e-07 |
| iid_sphere_recolor | 2.669833e-06 | 2.297435787482113e-06 | 1.1988946738739395e-06 | 4.768504104181206e-06 | 6.969720117054014e-07 |
| sobol_sphere_recolor | 2.648143e-06 | 2.5164161867978515e-06 | 1.0873531766282468e-06 | 4.1533894907920365e-06 | 7.09709312780917e-07 |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_sobol | 1.1447768617892544 | 1.0349050602319232 | 0.5139302374340373 | 1.8018200668959086 | 0.27886203393678055 |
| iid_over_sobol | 1.120755775810981 | 1.0392119784664358 | 0.5621892188365334 | 1.8215185161954384 | 0.22846702353185233 |

## 3-rep bias/variance

- `current`: bias² mean `9.224192e-07`, variance mean `1.737399e-06`, total mean `2.659818e-06`.
- `iid_sphere_recolor`: bias² mean `9.883927e-07`, variance mean `1.681440e-06`, total mean `2.669833e-06`.
- `sobol_sphere_recolor`: bias² mean `9.164118e-07`, variance mean `1.731731e-06`, total mean `2.648143e-06`.

## Gate decisions

- `complete_100`: **PASS**.
- `checksums`: **PASS**.
- `sobol_mean_mse`: **FAIL**.
- `current_over_sobol_global`: **FAIL**.
- `current_over_sobol_median`: **FAIL**.
- `current_over_sobol_q10`: **FAIL**.
- `current_over_sobol_min`: **FAIL**.
- `sobol_bias_proxy`: **PASS**.
