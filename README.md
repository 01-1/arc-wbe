# ARC White-Box Estimation Challenge 2026 — research artifacts

Closed research directions, measurements, and the raw rows behind both, from a
Phase 1 entry.

This is not an estimator release. It is the part of the work usually thrown
away: the configurations that lost, the gates that failed, and the numbers that
closed them. Nobody optimising for rank pays to characterise configurations they
already know lose, so measured dead ends are scarce — which is the reason this
repository exists.

**Write-up:** *(link pending)* — what the results mean and how they were
reached. This repository is the evidence behind it; the two are meant to be read
together.

## Where to start

| If you want | Go to |
|---|---|
| The research arc in one page | [`history/SUMMARY.md`](history/SUMMARY.md) |
| Ideas already ruled out, with the number that ruled each out | [`history/03-rejected-and-guarded-ideas.md`](history/03-rejected-and-guarded-ideas.md) — 69 lanes, indexed by line |
| A bias/variance bug you may share | [`GATE_REAUDIT.md`](GATE_REAUDIT.md), "Mis-attributions and quantity-label errors" |
| To reproduce the sampling-error exponent | [`analysis/block_ladder/`](analysis/block_ladder/) — `python fit_ladder.py` |
| Raw evidence for a single gate | [`gates/`](gates/) |
| The commit history | branch [`full-history/main`](../../tree/full-history/main) |

## Layout

| Path | Contents |
|---|---|
| [`history/`](history/) | Estimator history: current route, winning checkpoints, 69 closed lanes, QMC/LHS closures, the block-scaling closure, benchmarking notes, and the recovered pre-refactor round |
| [`GATE_REAUDIT.md`](GATE_REAUDIT.md) | Re-audit of all 71 gate rejections under a corrected scoring model, with nine label-level errors found and corrected |
| [`gates/`](gates/) | ~240 files covering 41 artifact-backed gates |
| [`analysis/block_ladder/`](analysis/block_ladder/) | Four-point sample-count ladder: per-MLP data, extraction, fit |
| [`analysis/truth_bank/`](analysis/truth_bank/) | 100-MLP research truth bank (256×32, ~1.64e6 antithetic samples each) |
| [`research-notes/`](research-notes/) | Lane closeouts, scouting notes, upstream audits |
| [`cloud/fly_runner.py`](cloud/fly_runner.py) | The distributed benchmark runner behind every full-100 measurement |
| [`references/`](references/) | Links to source material, not redistributed |

## What a "gate" is

A **gate** is a preregistered, single-shot, pass/fail promotion test for a
candidate estimator change. The hypothesis, the exact launch command, and the
numeric thresholds are written down *before* the run; the run happens once; the
verdict is mechanical.

Each gate leaves a fixed artifact set in [`gates/`](gates/):

| Suffix | Role |
|---|---|
| `*_prereg_*.md` | Hypothesis, frozen launch command, numeric pass thresholds |
| `*_manifest_*.json` | Machine-side task spec |
| `*_payload_*.py` | Code run on each machine, one MLP per shard |
| `*_fly.jsonl` | Raw per-shard rows |
| `*_aggregate_*.py` | Aggregation and decision logic |
| `*_results.json` | Statistics plus per-criterion PASS/FAIL |
| `*_report.md` | The verdict |

For example, the terminal two-Gaussian mixture readout gate passed only if *all*
of: mean baseline/candidate MSE ratio ≥ `1.35`, median ≥ `1.20`, q10 ≥ `0.90`,
no per-MLP ratio below `0.70`, and mean squared-bias proxy ≤ `1.0e-6`. It
returned `0.955x` on the mean ratio and died, despite passing its bias
condition.

The ceremony exists because per-MLP MSE on this benchmark has roughly 105%
coefficient of variation. Choose the metric after seeing the data and something
always looks like a winner. Frozen thresholds are what make 69 rejections
evidence rather than 69 discouraged guesses.

Preregistration is not sufficient, though, and this archive proves it: the
spherical-Stein gate's `stein_bias: FAIL` was frozen, computed, and wrong — its
aggregator mislabelled `M3` as `bias²`, and the corrected value is `−1.98e-8`,
consistent with zero. The discipline constrains the analyst, not the code. That
is why raw rows ship next to every verdict, and why any prose figure here —
including mine — should be treated as unverified until re-derived.

### How the counts relate

- **41 artifact-backed gates** in [`gates/`](gates/), roughly six files each.
- **30 history-only gates** — scorer comparisons and offline screens that left a
  history entry but no separate artifact set. 71 audited in total.
- **69 closed lanes** in the history. These are narrative closures and do *not*
  map one-to-one onto gates: some cover several gates, some gates produced no
  lane entry, and several lanes were closed on theory alone with no gate at all.

## Caveats

- Measurements are on the **public release** dataset
  (`hf://aicrowd/arc-whestbench-public-2026@v1-phase1`, `mini` split, 100 MLPs),
  which is **not** the 50-MLP leaderboard scoring set. Same generator, different
  instances.
- [`analysis/truth_bank/`](analysis/truth_bank/) bakes truth at only ~1.64e6
  samples, a `~1.1e-7` truth-side MSE floor — the same order as the quantities
  of interest. **Do not use it for MSE or bias estimation.** It exists for
  research needing raw activation structure. Gate figures marked bank-derived
  are not decision-grade.
- The sampling-error exponent is measured for one estimator family. Whether it
  holds for the methods at the top of the leaderboard is untested.
- Raw Fly run logs are not published: they carry presigned object-store URLs and
  machine identifiers.
  [`analysis/block_ladder/extract_ladder.py`](analysis/block_ladder/extract_ladder.py)
  documents exactly what was taken from them.
- Per-participant leaderboard analysis was removed before publication. Exact
  scores and ranks re-identify entrants against the public board, so pseudonyms
  would not have been enough.

## License

MIT. Portions of the tooling derive from the AIcrowd starter kit, also MIT.
Third-party papers and posts are linked from [`references/`](references/) rather
than redistributed.
