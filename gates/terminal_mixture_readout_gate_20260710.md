# Terminal two-Gaussian mixture ReLU readout gate, 2026-07-10

## Preregistration

This is a research-only, machine-side truth-bank measurement. It does not
modify `estimator.py` or the submission estimator. Each of the 100 fresh
truth-bank MLPs is rebuilt from its bank seed and its rebuilt weight checksum
must match the bank checksum.

Each MLP gets three independent replicates of the legal current route:
16 antithetic Hadamard blocks, exact first-layer Gaussian ReLU mean/covariance
recoloring, a `1.5x` first-successor variance match, fp32 propagation, and
three Strassen levels. The payload retains the final preactivation rows `z`
before the final ReLU.

The positive half-bases are concatenated as 16 independent `4096 x 256`
Hadamard rows first; the negative ReLU partners are formed afterward. No
interleaved positive/negative array is sliced, so each block keeps its own
independent flip vector. The first-successor variance correction uses fp64
only for variance statistics and applies the final centered update in fp32,
matching the current route.

The baseline is the direct row mean `mean(max(z, 0))` per output. The fixed
candidate is an output-wise, univariate two-Gaussian heteroscedastic mixture,
fit independently for each output using only that replicate's `z` rows:

- initialize equal component weights at `0.5` and component means/variances
  from the lower and upper sorted halves around the sample median;
- run exactly 12 deterministic EM updates;
- use variance floor `1e-4 * total marginal variance + 1e-30`;
- apply weight floor `0.02`, then renormalize after every update;
- compute responsibilities with stable log-responsibilities and log-sum-exp;
- estimate the candidate as the analytic mixture `E[ReLU(Z)]`.

The existing one-Gaussian analytic plug-in from each replicate's sample mean
and variance is reported as a control only; it cannot select or tune the
mixture. Truth-bank final means are read only after both fixed estimators are
computed, solely for scoring.

For each MLP, aggregate the three replicate MSEs and estimate replicate-pair
variance, the squared error of the three-replicate mean, and a conservative
squared-bias proxy obtained by subtracting one-third of the replicate variance
from that mean squared error. The gate passes only if the mixture has mean
baseline/candidate MSE ratio `>=1.35`, median ratio `>=1.20`, q10 ratio
`>=0.90`, no per-MLP ratio `<0.70`, and mean squared-bias proxy `<=1.0e-6`.

The exact frozen launch is:

```text
make fly-payload \
  FLY_MLPS=100 \
  FLY_PAYLOAD_MANIFEST=paired_fly_logs/fingerprint_theory/terminal_mixture_readout_manifest_20260710.json \
  FLY_PAYLOAD_FILES="estimator.py local_engine.py paired_fly_logs/fingerprint_theory/terminal_mixture_readout_payload_20260710.py analysis/truth_bank/truth_bank.npz" \
  FLY_PAYLOAD_JSONL=paired_fly_logs/fingerprint_theory/terminal_mixture_readout_gate_20260710_fly.jsonl \
  FLY_PAYLOAD_MAX_RESULT_SECONDS=420
```

No parameter sweep, truth-labeled fitting, local estimator scoring, or
estimator/history change is permitted for this gate.

## Result

The first attempted launch was discarded before measurement: its inherited
interleaved block helper and basename-only manifest path caused returncode-2
script lookup failures. The corrected packaging and route were then run once
with the frozen command above. The corrected payload returned `100/100`
shards with zero failures and verified every rebuilt weight checksum.

Aggregate artifacts:

- `terminal_mixture_readout_gate_20260710_fly.jsonl`
- `terminal_mixture_readout_gate_20260710_results.json`

Mixture versus direct-ReLU baseline:

- mean baseline MSE `2.6775205219e-6`; mean candidate MSE
  `2.8027283307e-6`; pooled mean ratio `0.9553264555x`;
- median per-MLP ratio `0.9530426299x`; q10/q90 `0.9101160915x` /
  `0.9801239599x`; minimum ratio `0.8753156605x`; zero ratios below `0.70`;
- mean replicate-pair variance `2.3197934031e-6`;
- three-replicate mean MSE `1.2561993954e-6`; squared-bias proxy
  `6.0066460386e-7`.

The one-Gaussian control also lost: mean ratio `0.7121048393x`, median
`0.7063586581x`, and q10 `0.5681661893x`. Verdict: **FAIL**. The mixture
misses the required `1.35x` mean and `1.20x` median gains despite passing its
bias-proxy and tail conditions. No estimator change or further run is
warranted.
