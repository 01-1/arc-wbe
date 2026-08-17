# Tail-Aware Projection Proxy Gate (truth-bank edition)

Fly-bank measurement gate. Each Machine rebuilt its bank MLP from seed, verified the weight checksum, sampled antithetic particles, computed a suffix Hutchinson diagonal `L^2` kernel from concrete remaining weights/ReLU masks, and measured final-layer MSE changes from coordinate mean corrections. No estimator behavior was changed or scored.

- Returned MLPs: 100
- Missing bank indices: none
- Layer rows: 800
- Checksum rebuild: PASS (100/100)

## Premise verdicts

- P1 FAIL: tail/local top-32 reduction ratio median 1.00938 [0.770359,1.23642], win fraction 0.555; gate requires median >= 1.10 and wins >= 0.60.
- P2 FAIL: Spearman tail median 0.182097 [-0.0123535,0.384111], tail-local median delta 0.0112866; gate requires tail >= 0.15 and delta >= 0.05.
- P3 FAIL: tail/successor top-32 reduction ratio median 1 [0.78127,1.23512], win fraction 0.515; gate requires median >= 1.05 and wins >= 0.55.

## By-layer tail/local ratio

| layer | median | q10 | q90 | win frac |
|---:|---:|---:|---:|---:|
| 4 | 0.997139 | -3.62148e+24 | 1.87616 | 0.550 |
| 8 | 1.01862 | -2.51467e+24 | 1.27935 | 0.630 |
| 12 | 1.04739 | 0.692401 | 1.37238 | 0.620 |
| 16 | 0.996472 | 0.827589 | 1.32205 | 0.480 |
| 20 | 0.994348 | 0.768949 | 1.16472 | 0.490 |
| 24 | 0.994624 | 0.912501 | 1.14559 | 0.430 |
| 28 | 1.01567 | 0.95854 | 1.07634 | 0.620 |
| 30 | 1.00917 | 0.975924 | 1.03778 | 0.620 |

## Decision: DEAD

All MSE levels and reductions are aggregated after subtracting the truth-bank floor. Small absolute changes near the floor are interpreted through paired ratios/win fractions rather than standalone levels.
