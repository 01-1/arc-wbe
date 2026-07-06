# Fly Truth Bank

This directory stores a private research truth bank for offline estimator gates.
It is not a grader fixture and was generated from fresh deterministic seeds, not
from public/private evaluation rows or label archives.

## Contents

- `truth_bank.npz`: compressed arrays:
  - `seeds`: shape `(100,)`, unsigned 63-bit MLP seeds.
  - `truths`: shape `(100, 32, 256)`, fp64 per-layer activation means.
  - `weights_sha256`: one SHA256 checksum per MLP over contiguous fp32 weight
    bytes in `local_engine.build_mlp` layer order.
- `metadata.json`: sample counts, wall times, FLOPs, checksums, and generation
  settings.
- `raw_results.jsonl`: raw Fly `WHEST_RESULT_JSON` rows collected from the
  Machines.
- `loader.py`: import helper:

```python
from analysis.truth_bank import load_bank

seeds, truths, metadata = load_bank()
```

## Generation

Seeds are derived as `sha256("arc-whest-fly-truth-bank-20260706-v1:<index>")`,
taking the first 63 bits as an integer. Exact seeds `11`, `22`, and `33` are
rejected and rehashed if encountered. The generation process does not read
grader fixture seeds, grader labels, public leaderboard cases, or private test
state.

Each Fly Machine runs one MLP built with `local_engine.build_mlp(width=256,
depth=32, seed=seed)`, then performs antithetic Monte Carlo with fp32
activations and fp64 accumulation for a timer-targeted run.

Generated on 2026-07-06 with `FLY_IMAGE_LABEL=whest-truth-bank-20260706-v3`.
The run collected 100/100 MLPs. Per-MLP sample counts ranged from 712,704 to
2,850,816, with mean 1,640,304.64 samples. The mean per-MLP compute was
6.87993631277056e12 forward-pass FLOPs. Checksum verification rebuilt index 0
locally from seed `6604632520249517929` and matched the Fly Machine checksum.

Commands:

```bash
make fly-truth-dry FLY_APP=whest-timing-20260627 FLY_MLPS=100
make fly-truth FLY_APP=whest-timing-20260627 FLY_MLPS=100
make truth-bank FLY_MLPS=100
```

## Fly Bank Gate

Run the current estimator against this bank on Fly and collect both predictions
and metrics:

```bash
make fly-bank-dry FLY_APP=whest-timing-20260627 FLY_MLPS=100
make fly-bank FLY_APP=whest-timing-20260627 FLY_MLPS=100 \
  FLY_IMAGE_LABEL=whest-bank-gate-$(date +%Y%m%d%H%M%S)
```

The target uploads only:

- `$(FLY_ESTIMATOR)` (default `estimator.py`)
- `scripts/fly_bank_gate_entrypoint.py`
- `analysis/truth_bank/truth_bank.npz`

Results are written to `analysis/truth_bank/bank_predictions.jsonl`. Each line is
one Fly shard result. Its `records` array includes `bank_index`, `seed`,
`prediction` (shape `32 x 256`), `all_layers_mse`, `final_layer_mse`,
`flops_used`, and `wall_time_s`.
