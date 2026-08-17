# Cheap-Suffix CV Pre-Test, 2026-07-05

Scope: offline only. No estimator edits, no tracked-file edits, no Fly, no
network, no pytest. Script and JSON outputs are in this directory:
`suffix_cv_pretest.py` and `suffix_cv_pretest_results.json`.

## Setup

- Route replica: copied from `paired_fly_logs/offline_screen/screen.py`
  conventions: width 256, depth 32, He fp32, 16 Hadamard blocks, first-layer
  recolor, first-successor variance scaling.
- Local self-generated MLP seeds: `11`, `22`.
- Estimator seeds per MLP: `60`, seeds `1000..1059`.
- Branch snapshots: post-activation ensemble after layers `K in {16, 20, 24}`.
- Measurement: for each MLP separately, across estimator seeds and per output
  coordinate, correlate the full final mean with each cheap final mean. The
  decision statistic below is then variance-weighted across MLPs. This avoids
  the misleading raw concatenated statistic where MLP-to-MLP differences can
  dominate.

Bias of the cheap suffix is intentionally not used in the gate. The MLMC
telescope would handle the cheap estimator's unknown expectation; the screen
only asks whether the cheap endpoint tracks the full endpoint pathwise from
the same mid-depth ensemble state.

## Controls

| Control | K=16 rho^2 | K=20 rho^2 | K=24 rho^2 | Notes |
|---|---:|---:|---:|---|
| cheap == full | `1.0000` | `1.0000` | `1.0000` | variance-weighted pooled control passed |
| full + independent per-coordinate noise | `0.5031` | `0.5105` | `0.5216` | target was rho^2 ~= `0.50`; finite-sample drift acceptable |

The full-control unweighted mean-coordinate rho^2 is below 1 because some
coordinates have effectively zero across-seed variance; the variance-weighted
decision statistic is exactly 1.

## Decision Table

`h` is the marginal cheap-suffix/full-suffix FLOP ratio, using production
matmul/reduction work and excluding one-time offline SVD/PC fitting. `factor`
is the memo gate formula `(1 - rho^2) * (1 + h)`.

| Candidate | h | K=16 rho^2 | K=16 factor | K=20 rho^2 | K=20 factor | K=24 rho^2 | K=24 factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank r=32 | `0.2500` | `0.0278` | `1.2153` | `0.0757` | `1.1554` | `0.0887` | `1.1392` |
| rank r=64 | `0.5000` | `0.0912` | `1.3632` | `0.1269` | `1.3097` | `0.1186` | `1.3222` |
| rank r=128 | `1.0000` | `0.1242` | `1.7517` | `0.1705` | `1.6590` | `0.2569` | `1.4861` |
| diagonal Gaussian mean-map | `0.000244` | `0.5901` | `0.4100` | `0.6907` | `0.3094` | `0.8018` | `0.1982` |
| row subsample 1/4 | `0.2500` | `0.1373` | `1.0783` | `0.1194` | `1.1007` | `0.1153` | `1.1059` |
| projected width 128 | `0.2656/0.2708/0.2812` | `0.0153` | `1.2462` | `0.0160` | `1.2505` | `0.0186` | `1.2574` |

## Interpretation

Candidate 4 **SURVIVES** this pre-registered gate, but only through the
diagonal-Gaussian mean-map suffix. It clears both requirements at every tested
branch layer:

- K=16: rho^2 `0.5901`, factor `0.4100`.
- K=20: rho^2 `0.6907`, factor `0.3094`.
- K=24: rho^2 `0.8018`, factor `0.1982`.

The strongest measured configuration is K=24 diagonal Gaussian, but K=20 may
be the more useful production target because it covers more remaining suffix
depth while still giving rho^2 `0.6907` at essentially negligible marginal
cost. K=16 also clears the gate and gives still broader suffix coverage.

The other cheap suffixes are dead for this mechanism. Rank truncation preserves
some scalar mean-error movement but does not track per-coordinate seed
fluctuations well enough, and its realistic cost ratio is too high. The 128D
projected suffix is almost uncorrelated. The row-subsampled suffix behaves as
a calibration point rather than a practical CV candidate and also fails the
gate.

## Caveats

- The raw concatenated-across-MLPs table in the JSON is not the decision
  statistic; it is retained for audit but is confounded by between-MLP
  variation. The report uses within-MLP, across-estimator-seed correlations.
- The diagonal Gaussian suffix sees only snapshot mean and diagonal variance.
  Its high rho^2 means the current estimator's final mean fluctuations are
  strongly encoded in those low-order snapshot statistics, not that the
  diagonal closure is an accurate standalone estimator.
- Branch-layer rho^2 rises as K approaches the output, as expected from
  reduced chaotic decorrelation. In an MLMC/CV implementation the cheap level
  must track the full final output across seeds, including pre-K variation
  through the shared snapshot. This screen measures exactly that end-to-end
  shared-snapshot correlation, but it does not yet optimize the allocation of
  many cheap rows versus fewer paired full-cheap corrections.
- The cost ratio for the diagonal suffix counts two vector-matrix-style
  propagations per remaining layer, mean through `W` and variance through
  `W^2`, divided by the `8192`-row full suffix. Elementwise normal CDF/PDF and
  residual wall time still need scorer-path measurement.

## Gate Verdict

Candidate 4 is **ALIVE**. The winning family is the diagonal-Gaussian
mean-map cheap suffix, with K=16/20/24 all passing the pre-registered
`rho^2 >= 0.45` and factor `<= 0.645` gate. The recommended next action is a
focused estimator prototype of a diagonal-mean/variance cheap-level telescope,
starting with K=20 and K=24 allocation arithmetic, followed by scorer-path
compute measurement before any Fly comparison.
