# Offline anchored-CV screen

R per MLP: 100; truth samples per MLP: 400,000

| MLP seed | honest adj R^2 | generous CV R^2 | bias MSE | truth-noise MSE | seed-var/R | net bias MSE | decomp ratio | sanity MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | -0.0024 | -0.1962 | 4.2872e-08 | 1.4022e-07 | 2.7774e-08 | -1.2513e-07 | 0.901 | 2.7925e-06 |
| 22 | 0.0012 | -0.2524 | 1.0902e-07 | 3.5192e-07 | 4.8539e-08 | -2.9145e-07 | 0.772 | 4.9144e-06 |
| 33 | 0.0091 | -0.1539 | 2.5631e-07 | 2.1810e-07 | 3.6870e-08 | 1.3490e-09 | 0.879 | 3.9065e-06 |

## Pooled

- Honest scalar adjusted R^2: 0.0049
- Generous CV R^2: -0.0494
- Mean bias MSE: 1.3607e-07
- Mean net bias MSE: -1.3841e-07
- Mean variance-decomposition ratio: 0.851
- Mean sanity MSE: 3.8711e-06

## Verdict

Anchored features do not explain >= 40% of final-error variance in this offline screen.
