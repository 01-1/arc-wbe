# Use Evaluation Datasets

> [← Documentation](../README.md)

## 🎯 When to use this page

Every `whest run` generates fresh random MLPs and samples many forward passes to establish ground truth. This is correct but slow — especially when you are iterating on an estimator and re-running the same evaluation dozens of times during development.

Pre-created evaluation datasets let you do that expensive work once and reuse it across your entire development cycle:

- **Faster iteration** — `whest run --dataset` skips MLP generation and ground truth sampling entirely.
- **Fair comparisons** — every estimator you test is scored against the exact same MLPs with the same ground truth.
- **Reproducibility** — the dataset artifact records the seed and all creation parameters, so anyone can recreate it exactly.

For explicit seeds, `create-dataset` now uses a hierarchical `SeedSequence` split: each MLP gets one child stream for weight sampling and one child stream for Monte Carlo sampling.

## 💾 Public starter dataset

The public baseline dataset now lives on Hugging Face:

<https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026>

Use that dataset (or a locally generated HF-compatible dataset) as the input to `--dataset` so you and your collaborators run against the same fixed ground-truth set:

```bash
whest run --estimator estimator.py --dataset aicrowd/arc-whestbench-public-2026
```

## 🚀 Do this now

### 1. Create your dataset (once)

```bash
whest create-dataset -o my_dataset
```

This generates MLPs and samples ground truth means. Output is saved as a Hugging Face-compatible dataset artifact for easy sharing.

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-mlps` | 10 | Number of random MLPs to generate |
| `--n-samples` | 10000 | Samples per MLP for ground truth estimation |
| `--seed` | auto | RNG seed (auto-generated if omitted, always recorded) |
| `-o, --output` | `eval_dataset` | Output dataset path/id |
| `--width` | (contest default) | Neuron count per MLP |
| `--depth` | (contest default) | Layers per MLP |
| `--flop-budget` | (contest default) | FLOP cap for the estimator |

If you want to avoid extra host probing during local development, set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1` before `whest create-dataset` or `whest run`. This skips only the OS-native fallback probes used to fill missing hardware fields in report and dataset metadata. Cheap fields and `psutil`-backed fields are still recorded, and fallback-backed fields may remain `null`.

### 2. Run against it (every time)

```bash
whest run --estimator estimator.py --dataset my_dataset
```

The `--n-mlps` flag is ignored when `--dataset` is provided — the values come from the dataset artifact.

You can keep reusing the same dataset file across your entire development cycle. Edit your estimator, re-run the command, compare scores — the ground truth stays the same so differences reflect only your estimator changes.

## ✅ Expected outcome

- `create-dataset` produces a HF-compatible dataset at the specified location.
- `run --dataset` shows "Loading dataset" instead of "Generating MLPs" and skips ground truth sampling.
- `run --dataset` still shows a `Sampling Budget Breakdown (Ground Truth)` section in human output, restored from the dataset metadata for exactly the MLPs used in that run.
- Score reports are consistent across runs with the same dataset.

## Dataset portability

Unlike the old time-based scoring model, flopscope uses analytical FLOP counting rather than wall-clock timing. This means datasets are **fully portable across machines** — the stored ground truth and FLOP budgets are hardware-independent. You can create a dataset on a laptop and run it on a cloud instance with identical results.

## Dataset traceability

When using `--dataset`, the results JSON includes a `dataset` reference under `run_config` so you can always trace exactly which dataset produced a given score:

```json
{
  "run_config": {
    "dataset": {
      "path": "my_dataset",
      "sha256": "a1b2c3...",
      "seed": 42,
      "n_mlps": 10
    }
  }
}
```

Example seeded run command:

```bash
whest run --estimator estimator.py --seed 20260417
```

`run --seed` stores the chosen seed in `run_config.seed`, and all `create-dataset` metadata now records:

```json
{"seed_protocol": {"name": "whestbench_seedsequence_hierarchy", "version": "1.0", "seeded": true}}
```

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Score Report Fields](../reference/score-report-fields.md)
