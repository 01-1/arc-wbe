# ReLU Region Granularity Gate (truth-bank edition)

Pre-registered Fly-bank structural measurement. Each Machine rebuilt its truth-bank MLP from seed, verified the weight checksum, sampled Gaussian-bulk chords `x(t)=x0+t*u` over `[-1,1]`, counted sampled ReLU gate flips, and decomposed per-layer chord output variation into same-pattern affine interval variance versus between-interval variance. No estimator behavior was changed or scored.

- Returned MLPs: 100
- Missing bank indices: none
- Checksum rebuild: PASS (100/100)

## Premise verdicts

- P1 PASS: total breakpoint density median 160.958 [132.933,194.763] per sigma; implied region extent median 0.00621279 sigma. Gate requires <= 512 per sigma.
- P2 FAIL: layer-31 within-region variance share median 0.000364774 [0.000321346,0.000408837], max layer median 0.000364774. Gate requires layer 31 >= 0.05 or any layer >= 0.10.
- P3 PASS: effective live hyperplanes median 2239.5 [1957.5,2509.7] vs nominal 8192; layer-31 live median 49. Gate requires total <= 2048 or layer 31 <= 128.

## Layer snapshot

| layer | bp/sigma med | live med | frozen med | within-share med | flip-interval med | coflip-interval med |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.04167 | 98 | 0.617188 | 0.000247946 | 0.145833 | 0.0104167 |
| 1 | 5.16667 | 99 | 0.613281 | 0.000251672 | 0.14974 | 0.0104167 |
| 3 | 5.125 | 92 | 0.640625 | 0.000259089 | 0.147786 | 0.0110677 |
| 7 | 4.95833 | 81 | 0.683594 | 0.000274997 | 0.143229 | 0.0117188 |
| 15 | 5.04167 | 68.5 | 0.732422 | 0.000303846 | 0.147135 | 0.0117188 |
| 23 | 5 | 58 | 0.773438 | 0.00033254 | 0.145833 | 0.0130208 |
| 30 | 4.875 | 48.5 | 0.810547 | 0.000360127 | 0.141927 | 0.0104167 |
| 31 | 4.875 | 49 | 0.808594 | 0.000364774 | 0.139323 | 0.0117188 |

Layer 31 within-region variance share using total chord variance denominator: 0.00035522 [0.00031284,0.000399341].
Deep-layer flip co-occurrence fraction: 0.507107 [0.379574,0.614836].

## Decision: DEAD

The within-region share uses only adjacent grid intervals whose sampled full layer gate pattern is unchanged, so it is conservative with respect to missed sub-grid crossings. The breakpoint density is a sampled line-chord density, not an exact arrangement count.
