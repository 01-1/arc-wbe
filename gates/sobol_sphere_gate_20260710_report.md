# Sobol-sphere recolor gate

## Fly execution record

Frozen command:

```text
make fly-payload FLY_MLPS=100 FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/sobol_sphere_manifest_20260710.json FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/sobol_sphere_payload_20260710.py analysis/truth_bank/truth_bank.npz" FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/sobol_sphere_gate_20260710_fly.jsonl FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```

The first attempt was blocked before launch by the local sandbox's Tigris
upload restriction. The one allowed packaging/upload retry succeeded in
launching 100 Machines. Every observed payload failed at import with
`ModuleNotFoundError: No module named 'scipy'`; no MLP, truth, estimate, or
checksum computation ran. The runner returned 93 failures and stopped with
7 pending at `max_result_seconds=420`; the JSONL contains 0 successful rows.
Therefore the mechanical gate verdict below is `FAIL`, but the research
comparison is **not evaluable** and must not be interpreted as a sampler
quality result. No runtime retry was made.

Successful shards: `0/100`; checksums: `FAIL`.

**FAIL**

| method | mean MSE | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| no successful rows | n/a | n/a | n/a | n/a | n/a |
| current | n/a | n/a | n/a | n/a | n/a |
| iid_sphere_recolor | n/a | n/a | n/a | n/a | n/a |
| sobol_sphere_recolor | n/a | n/a | n/a | n/a | n/a |

| ratio | mean | median | q10 | q90 | min |
|---|---:|---:|---:|---:|---:|
| current_over_sobol | n/a | n/a | n/a | n/a | n/a |
| iid_over_sobol | n/a | n/a | n/a | n/a | n/a |

## 3-rep bias/variance

No 3-rep estimates were returned; the Fly run failed before payload import.

## Gate decisions

- `complete_100`: **FAIL**.
- `checksums`: **FAIL**.
- `sobol_mean_mse`: **FAIL**.
- `current_over_sobol_global`: **FAIL**.
- `current_over_sobol_median`: **FAIL**.
- `current_over_sobol_q10`: **FAIL**.
- `current_over_sobol_min`: **FAIL**.
- `sobol_bias_proxy`: **FAIL**.
