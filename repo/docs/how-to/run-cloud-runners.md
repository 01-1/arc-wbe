# Run Cloud Runners

> [← How-to](README.md)

Use these runners when local timing is too slow or too noisy. Both paths use the
public Hugging Face baked dataset only on the local workstation, via
`scripts/randomize_whest_dataset.py`. The cloud workers receive ordinary local
dataset directories and run with `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`,
so they do not download from Hugging Face or recompute Monte Carlo ground truth.

## Modal: Small Batches, Many Cores

The Modal runner is for quick checks over a small number of MLPs on a large CPU
container:

```bash
make modal-small MODAL_MLPS=3 MODAL_CORES=32 WALL_TIME=60
```

It stages a random baked-label subset locally, uploads that directory into a
Modal Volume, and runs:

```bash
whest run --dataset /datasets/<staged-dataset> --runner subprocess
```

Useful knobs:

| Make variable | Default | Meaning |
|---|---:|---|
| `MODAL_MLPS` | `3` | Number of baked MLP rows to send to Modal |
| `MODAL_CORES` | `32` | Modal CPU request and `--max-threads` |
| `MODAL_MEMORY` | `65536` | Modal memory in MiB |
| `MODAL_REGION` | `us-east-1` | Modal region |
| `MINI_RANDOM_SEED` | `20260624` | Row-sampling seed |

## Fly.io: Many MLPs, 8 Shared Cores

The Fly runner is for wide fan-out on EWR Machines. It splits the local baked
dataset into one-MLP archives, uploads those small archives to an S3-compatible
object store, and launches one short-lived Machine per MLP. Machines run with
`shared-cpu-8x` by default, download exactly one dataset archive plus the current
`estimator.py`, and stream results back through Fly logs.

Install the AWS CLI and set AWS-compatible object-store credentials. The
repository Makefile defaults to the saved `whest-tigris` profile and
the public Tigris bucket used for Fly testing:

```bash
AWS_PROFILE=whest-tigris aws s3 ls s3://whest-fly-runner-20260628/
```

Workers use plain HTTPS public object URLs by default:

```bash
https://whest-fly-runner-20260628.fly.storage.tigris.dev
```

Start with a dry run:

```bash
make fly-large-dry FLY_APP=<your-fly-app> FLY_MLPS=100
```

Then launch from already-uploaded archives:

```bash
make fly-large FLY_APP=<your-fly-app> FLY_MLPS=100
```

To upload archives and warm the image once:

```bash
make fly-large-build FLY_APP=<your-fly-app> FLY_MLPS=100
```

Then run timed Machines without rebuilding or re-uploading dataset archives. The
current `estimator.py` is uploaded as a tiny object by default:

```bash
make fly-large-fast FLY_APP=<your-fly-app> FLY_MLPS=100
```

What happens:

1. `scripts/randomize_whest_dataset.py` creates a local random subset from the
   baked public dataset.
2. `scripts/split_whest_dataset.py` splits that subset into `N` one-MLP dataset
   directories.
3. `cloud/fly_runner.py` writes `N` `mlp-000000.tar.gz` archives. With
   `--upload`, it uploads them to object storage using `aws s3 cp`.
4. It uploads the current `estimator.py` under a content-hashed object key
   unless `--no-upload-estimator` or `--estimator-url` is used.
5. It builds and pushes a stable Fly image containing dependencies and runner
   scripts, but no dataset or estimator bytes.
6. It launches `N` short-lived Machines. Each Machine downloads the matching
   dataset archive and estimator file, runs WhestBench, prints
   `WHEST_RESULT_DONE ...`, then exits with `--rm`.

Useful knobs:

| Make variable | Default | Meaning |
|---|---:|---|
| `FLY_APP` | required | Existing Fly app for registry and Machines |
| `FLY_MLPS` | `100` | Number of one-MLP archives and Machines |
| `FLY_REGION` | `ewr` | Fly region |
| `FLY_VM_SIZE` | `shared-cpu-8x` | Machine size |
| `FLY_LAUNCH_CONCURRENCY` | `100` | Local parallel `fly machine run` calls |
| `FLY_AWS_PROFILE` | `whest-tigris` | Local AWS profile for uploads |
| `FLY_OBJECT_BUCKET` | `whest-fly-runner-20260628` | Writable Tigris bucket |
| `FLY_OBJECT_PREFIX` | `whest/fly-mlps` | Object key prefix under the bucket |
| `FLY_OBJECT_FLAGS` | public Tigris URL | URL mode for dataset downloads |
| `FLY_ESTIMATOR` | `estimator.py` | Estimator file to upload and run |
| `FLY_ESTIMATOR_OBJECT_FLAGS` | public Tigris URL | Estimator bucket/base URL |
| `FLY_ESTIMATOR_FLAGS` | empty | Estimator URL/upload overrides |
| `FLY_IMAGE_LABEL` | `whest-runner` | Stable runner image label |

For repeated tests with the same sampled dataset, use `make fly-large-fast` after
`make fly-large-build`. Dataset upload is off by default; estimator upload is on
by default because estimator iteration is the main loop. The fast target also
uses `--skip-build`, so the measured path starts at Machine launch and waits
only until each Machine prints the done sentinel; teardown time is ignored.
