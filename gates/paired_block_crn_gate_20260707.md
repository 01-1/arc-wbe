# Paired-Block CRN Gate, 2026-07-07

Purpose: audit the remaining common-randomness paired-block hypothesis for the
depth-32 Hadamard route without changing estimator behavior. The live question
was whether two legal correlated block estimates can preserve post-ReLU
nonlinear structure while canceling leading final-error noise when averaged.

Rules boundary: a submitted estimator may use only the passed MLP object and
MLP-independent randomness. Truth-bank labels are research-only measurement
labels and were not used to fit, tune, branch, or change estimator behavior.

## Pre-Registered Family

The exact coupling family for this scout was fixed before looking at labels:

- Use legal current-route Hadamard blocks with shared first-layer weights,
  first-layer covariance recolor, first-successor variance match, and full
  nonlinear propagation through all post-ReLU layers.
- Treat adjacent sibling blocks from the existing 16-block payload as the
  smallest available common-randomness screen. These siblings share the same
  MLP, same route, same recolor/variance-match transform, and paired
  antithetic construction, so they test whether block-level final errors have
  enough positive covariance structure to support a CRN average.
- The estimator candidate, if the screen passed, would be a simple
  mode-gated paired-block sampler that spends the same charged block count on
  fixed sibling pairs and averages pair means. No truth labels, public/private
  case identifiers, or grader state would enter estimator behavior.

Nearest killed neighbor: the 2026-07-07 block predictability gate killed
feature-weighted block allocation. This audit is distinct because it does not
ask whether features can weight blocks; it asks whether legally correlated
block estimates already exhibit final-error cancellation when paired and
averaged, preserving nonlinear post-ReLU trajectories.

PASS thresholds were fixed to the owner bar: `>=20%` mean final-MSE reduction,
median reduction `>=10%`, q10 no worse than `-5%`, plus a clear label-free
implementation path. A model-based proxy would have needed predicted `>=1.3x`
variance reduction with a simple label-free rule.

## Measurement

Smallest measurement used: the existing full 100-shard Fly payload from
`block_predictability_gate_20260707.jsonl`, which had already propagated
8 independent legal 16-block current-route ensembles per truth-bank MLP and
recorded per-block final-error summaries. No new Fly run was launched because
this payload directly contains the relevant block-level common-randomness
screen; rerunning the same current-route blocks would only spend capacity to
reproduce the same killed neighborhood.

Command:

```sh
python paired_fly_logs/fingerprint_theory/block_predictability_aggregate.py \
  paired_fly_logs/fingerprint_theory/block_predictability_gate_20260707.jsonl \
  --output /tmp/block_predictability_reaggregate.json
```

The re-aggregation completed over 100 MLPs / 12,800 block rows.

Results:

- Weighting/allocation proxy: mean/median/q10/q90 variance ratios
  `1.001` / `1.000` / `0.994` / `1.009`, below the `1.3x` proxy bar.
- Paired-control proxy: mean/median/q10 reductions
  `2.9%` / `1.5%` / `0.04%`, below the `20%` / `10%` / `-5%` PASS bar.
- The log squared-error prediction correlation was `0.542`, confirming that
  block difficulty signal exists, but it did not convert into useful
  variance reduction under equal charged block count.

## Verdict

FAIL. The current-route block-level CRN structure has far too little measured
pair cancellation to justify implementing a mode-gated estimator candidate.
Default estimator behavior is unchanged. No `python -m py_compile`, no
`make fly-mode`, and no final `make fly` were warranted because no
`estimator.py` behavior changed.

Final `make fly` crossed `<1.6e-7`: no, not run; no candidate passed the
pre-registered gate.
