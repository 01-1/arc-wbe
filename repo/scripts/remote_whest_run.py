#!/usr/bin/env python3
"""Run WhestBench inside a cloud worker against a local baked dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cloud_whest_common import whest_env

REPO_ROOT = Path(__file__).resolve().parents[1]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Local baked dataset directory.")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--estimator", default="estimator.py")
    parser.add_argument("--runner", default="subprocess", choices=("local", "subprocess", "server", "inprocess"))
    parser.add_argument("--n-mlps", type=_positive_int, help="Logical MLP count before sharding.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=_positive_int, default=1)
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--wall-time-limit", type=float, default=60.0)
    parser.add_argument("--residual-wall-time-multiplier", type=float, default=2.0)
    parser.add_argument("--max-threads", type=_positive_int)
    parser.add_argument("--mode", help="Optional WHEST_K3_MODE value.")
    parser.add_argument("--format", choices=("rich", "plain", "json"), default="plain")
    parser.add_argument("--detail", choices=("raw", "full"), default="raw")
    parser.add_argument("--no-fail-fast", action="store_true")
    return parser.parse_args(argv)


def _selection_metadata(
    source_metadata: dict[str, Any],
    *,
    split: str,
    indices: list[int],
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    out = dict(source_metadata)
    out["n_mlps"] = len(indices)
    out["split"] = split
    out.pop("prepared_splits", None)
    out["row_selection"] = {
        "method": "contiguous-shard",
        "source_dataset": "local",
        "source_split": split,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "start": indices[0] if indices else 0,
        "end_exclusive": (indices[-1] + 1) if indices else 0,
        "indices": indices,
    }
    return out


def _maybe_slice_dataset(args: argparse.Namespace) -> tuple[Path, int | None]:
    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        raise SystemExit(f"--dataset must be a local directory: {dataset}")
    if args.shard_count == 1:
        return dataset, args.n_mlps
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")

    from whestbench.dataset import load_dataset, metadata
    from whestbench.dataset_io import write_dataset_dir

    ds = load_dataset(str(dataset), split=args.split)
    logical_total = min(args.n_mlps or len(ds), len(ds))
    start = logical_total * args.shard_index // args.shard_count
    end = logical_total * (args.shard_index + 1) // args.shard_count
    indices = list(range(start, end))
    if not indices:
        raise SystemExit(
            f"empty shard {args.shard_index}/{args.shard_count}; "
            f"increase --n-mlps or lower --shard-count"
        )

    selected = ds.select(indices)
    out_dir = Path(tempfile.mkdtemp(prefix=f"whest-shard-{args.shard_index:04d}-")) / "dataset"
    write_dataset_dir(
        selected,
        output_dir=out_dir,
        split=args.split,
        metadata=_selection_metadata(
            metadata(ds),
            split=args.split,
            indices=indices,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        ),
    )
    return out_dir, len(indices)


def _build_command(args: argparse.Namespace, dataset: Path, n_mlps: int | None) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/whest_with_residual_multiplier.py",
        "--residual-wall-time-multiplier",
        str(args.residual_wall_time_multiplier),
        "--",
        "run",
        "--estimator",
        args.estimator,
        "--dataset",
        str(dataset),
        "--split",
        args.split,
        "--runner",
        args.runner,
        "--flop-budget",
        str(args.flop_budget),
        "--wall-time-limit",
        str(args.wall_time_limit),
        "--format",
        args.format,
        "--detail",
        args.detail,
    ]
    if not args.no_fail_fast:
        cmd.append("--fail-fast")
    if n_mlps is not None:
        cmd.extend(["--n-mlps", str(n_mlps)])
    if args.max_threads is not None:
        cmd.extend(["--max-threads", str(args.max_threads)])
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    dataset, n_mlps = _maybe_slice_dataset(args)

    env = whest_env(args.max_threads)
    if args.mode:
        env["WHEST_K3_MODE"] = args.mode

    cmd = _build_command(args, dataset, n_mlps)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
