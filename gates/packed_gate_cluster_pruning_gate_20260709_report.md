# Packed gate-clustered two-sided pruning gate

Source: `paired_fly_logs/fingerprint_theory/packed_gate_cluster_pruning_gate_20260709.jsonl`. Successful unique shards: `100/100`; checksums: `PASS`.

The payload reproduced the real current 16-block route machine-side and used full dense preactivations only to measure true output-dead coordinates and certificate violations. Truth-bank means were not read. All candidate sorting and certificates were label-free functions of the MLP and its own route activations.

## Verdict

**FAIL: no plan passed the frozen product/raw/certificate gate; close two-sided pruning**

Frozen gates: decision-layer mean `live_input x uncertified_output <= 0.31` (`<=0.29` strong), per-MLP q90 projected 28-block raw `<=2.4e10`, zero certificate violations, and a padding-32/64 plan with mean/max buckets `<=8/16`, median groups per bucket `>=4`, peak packed memory `<=512 MiB`, and activation gather/scatter ratio `<=2.5x`.

Even an oracle output-dead certificate cannot reach the product gate. The best grouping (`gatekey32`, G=32) obeys `E[L(1-D)] >= E[L]-E[D] = 0.4129` (per-MLP q10/q90 `0.3729`/`0.4502`), already above `0.31` before screening or padding costs.

## Best projected plans

| rank | strategy | G | cert | pad | product mean | cert recall med | raw28 q90 | violations | buckets mean/max | groups/bucket med | act traffic max | arithmetic | packing | gate |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | gatekey32 | 64 | box | 16 | 0.7328 | 0.000 | 5.150e+10 | 0 | 2.2/3 | 64.0 | 2.00 | fail | fail | fail |
| 2 | support_lex | 64 | box | 16 | 0.7328 | 0.000 | 5.150e+10 | 0 | 2.2/3 | 64.0 | 2.00 | fail | fail | fail |
| 3 | support_lex | 32 | box | 16 | 0.7056 | 0.000 | 5.174e+10 | 0 | 2.4/4 | 125.8 | 2.00 | fail | fail | fail |
| 4 | gatekey32 | 32 | box | 16 | 0.7056 | 0.000 | 5.174e+10 | 0 | 2.4/4 | 125.8 | 2.00 | fail | fail | fail |
| 5 | gatekey32 | 128 | box | 16 | 0.7567 | 0.000 | 5.184e+10 | 0 | 2.0/4 | 32.0 | 2.00 | fail | fail | fail |
| 6 | support_lex | 128 | box | 16 | 0.7567 | 0.000 | 5.184e+10 | 0 | 2.0/4 | 32.0 | 2.00 | fail | fail | fail |
| 7 | contiguous | 64 | box | 16 | 0.7622 | 0.000 | 5.318e+10 | 0 | 1.7/3 | 64.0 | 2.00 | fail | fail | fail |
| 8 | contiguous | 128 | box | 16 | 0.7818 | 0.000 | 5.325e+10 | 0 | 1.6/2 | 32.0 | 2.00 | fail | fail | fail |
| 9 | activation_norm | 64 | box | 16 | 0.7604 | 0.000 | 5.329e+10 | 0 | 1.9/3 | 64.0 | 2.00 | fail | fail | fail |
| 10 | support_lex | 64 | box | 32 | 0.7328 | 0.000 | 5.330e+10 | 0 | 1.6/2 | 64.0 | 2.00 | fail | PASS | fail |
| 11 | gatekey32 | 64 | box | 32 | 0.7328 | 0.000 | 5.330e+10 | 0 | 1.6/2 | 64.0 | 2.00 | fail | PASS | fail |
| 12 | gatekey32 | 32 | box | 32 | 0.7056 | 0.000 | 5.340e+10 | 0 | 1.7/3 | 128.0 | 2.00 | fail | PASS | fail |
| 13 | support_lex | 32 | box | 32 | 0.7056 | 0.000 | 5.340e+10 | 0 | 1.7/3 | 128.0 | 2.00 | fail | PASS | fail |
| 14 | activation_norm | 128 | box | 16 | 0.7800 | 0.000 | 5.342e+10 | 0 | 1.7/3 | 32.0 | 2.00 | fail | fail | fail |
| 15 | gatekey32 | 128 | box | 32 | 0.7567 | 0.000 | 5.356e+10 | 0 | 1.5/3 | 64.0 | 2.00 | fail | PASS | fail |
| 16 | support_lex | 128 | box | 32 | 0.7567 | 0.000 | 5.356e+10 | 0 | 1.5/3 | 64.0 | 2.00 | fail | PASS | fail |
| 17 | contiguous | 32 | box | 16 | 0.7386 | 0.000 | 5.379e+10 | 0 | 1.9/3 | 128.0 | 2.00 | fail | fail | fail |
| 18 | activation_norm | 32 | box | 16 | 0.7367 | 0.001 | 5.393e+10 | 0 | 2.0/4 | 128.0 | 2.00 | fail | fail | fail |
| 19 | contiguous | 64 | box | 32 | 0.7622 | 0.000 | 5.481e+10 | 0 | 1.4/2 | 128.0 | 2.00 | fail | PASS | fail |
| 20 | contiguous | 128 | box | 32 | 0.7818 | 0.000 | 5.488e+10 | 0 | 1.3/2 | 64.0 | 2.00 | fail | PASS | fail |
| 21 | activation_norm | 128 | box | 32 | 0.7800 | 0.000 | 5.496e+10 | 0 | 1.3/2 | 64.0 | 2.00 | fail | PASS | fail |
| 22 | pc1 | 64 | box | 16 | 0.7594 | 0.000 | 5.498e+10 | 0 | 2.0/4 | 64.0 | 2.00 | fail | fail | fail |
| 23 | activation_norm | 64 | box | 32 | 0.7604 | 0.000 | 5.498e+10 | 0 | 1.4/2 | 128.0 | 2.00 | fail | PASS | fail |
| 24 | pc1 | 128 | box | 16 | 0.7788 | 0.000 | 5.509e+10 | 0 | 1.9/4 | 32.0 | 2.00 | fail | fail | fail |

## Unsorted contiguous controls (padding 32)

| G | cert | product mean [q10,q90] | raw28 q90 | recall median | buckets mean/max |
|---:|---|---:|---:|---:|---:|
| 64 | box | 0.7622 [0.7416,0.7825] | 5.481e+10 | 0.000 | 1.4/2 |
| 128 | box | 0.7818 [0.7592,0.8040] | 5.488e+10 | 0.000 | 1.3/2 |
| 32 | box | 0.7386 [0.7185,0.7576] | 5.557e+10 | 0.000 | 1.4/2 |
| 128 | combined | 0.7817 [0.7590,0.8040] | 6.118e+10 | 0.000 | 1.3/2 |
| 64 | combined | 0.7620 [0.7415,0.7825] | 6.155e+10 | 0.000 | 1.4/2 |
| 32 | combined | 0.7384 [0.7184,0.7576] | 6.318e+10 | 0.000 | 1.4/2 |

## Closeout

No estimator-mode design is authorized by this gate. The negative result closes packed two-sided pruning as formulated; do not edit `estimator.py` from this lane.

Full per-plan statistics and per-layer mean/q10/q90 curves are in the adjacent results JSON.
