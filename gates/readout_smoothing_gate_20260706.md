# Readout-Smoothing Gate (truth-bank edition)

Fly-bank-style machine-side gate. Each Machine rebuilt its bank MLP from the bank seed, verified the bank checksum, ran the readout comparison locally on that Machine, and returned only compact summary statistics. The local step only aggregates JSON and applies the truth-noise-floor subtraction from `metadata.json` sample counts.

- Returned MLPs: 92
- Missing bank indices: 7, 13, 22, 33, 54, 57, 71, 84
- Candidate n: 1024, 4096, 8192
- Replicates per n/mode: 8
- Large-n P1/P3 pass: 262144
- Checksum rebuild: PASS (92/92)

## Final-layer ratio table

| mode | n | smoothed/direct median | q10 | q90 | completed MLPs |
|---|---:|---:|---:|---:|---:|
| iid | 1024 | 1.02 | 1.00653 | 1.04649 | 92 |
| iid | 4096 | 1.10862 | 1.03573 | 1.20813 | 92 |
| iid | 8192 | 1.19943 | 1.07495 | 1.36965 | 92 |
| antithetic | 1024 | 1.02474 | 1.00839 | 1.05136 | 92 |
| antithetic | 4096 | 1.09919 | 1.03757 | 1.20941 | 92 |
| antithetic | 8192 | 1.2041 | 1.09643 | 1.42564 | 92 |

## Premise verdicts

- P1 FAIL: layer 31 abs-skew median/q90 0.431745/0.623726; excess-kurtosis median [q10,q90] 0.376738 [0.136261,0.753371].
- P2 FAIL: antithetic layer 31 ratio at n=8192 median 1.2041, q90 1.42564; gate requires median <= 0.667 and q90 < 0.9.
- P3 FAIL: layer 31 smooth bias^2 raw median 1.11136e-06, floor median 2.64946e-08, floor-subtracted median 1.09031e-06; saved variance median -1.04341e-06.

## Decision: DEAD

Bias values at or below the estimated truth-noise floor are interpreted as upper bounds. No estimator.py path was imported or scored by this gate entrypoint.
