#!/usr/bin/env python3
"""Download one one-MLP dataset archive, then run WhestBench."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RESIDUAL_FLOPS_PER_SECOND = 100_000_000_000.0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--estimator-url", required=True)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--wall-time-limit", type=float, default=60.0)
    parser.add_argument("--residual-wall-time-multiplier", type=float, default=2.0)
    parser.add_argument("--max-threads", type=int, default=8)
    parser.add_argument("--runner", choices=("local", "subprocess", "server", "inprocess"), default="local")
    parser.add_argument("--mode")
    parser.add_argument("--format", choices=("plain", "json", "rich"), default="plain")
    parser.add_argument("--detail", choices=("raw", "full"), default="raw")
    parser.add_argument("--done-sentinel")
    parser.add_argument("--linger-after-result", type=float, default=0.0)
    return parser.parse_args(argv)


def _download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "whest-fly-runner/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with output.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _extract(archive: Path, output_dir: Path) -> Path:
    with tarfile.open(archive, mode="r:gz") as tar:
        tar.extractall(output_dir, filter="data")
    candidates = [path for path in output_dir.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise SystemExit(f"expected one dataset directory in {archive}, got {len(candidates)}")
    dataset = candidates[0]
    if not (dataset / "metadata.json").is_file():
        raise SystemExit(f"downloaded archive did not contain a WhestBench dataset: {dataset}")
    return dataset


def _json_object_from_output(output: str) -> dict[str, object] | None:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _copy_one_mlp_fields(result: dict[str, object]) -> dict[str, object]:
    per_mlp = result.get("per_mlp")
    if not isinstance(per_mlp, list) or len(per_mlp) != 1:
        return {}
    mlp = per_mlp[0]
    if not isinstance(mlp, dict):
        return {}

    compact: dict[str, object] = {}
    for source, target in (
        ("adjusted_final_layer_score", "mlp_adjusted_final_layer_score"),
        ("flops_used", "mlp_flops_used"),
        ("effective_compute", "mlp_effective_compute"),
        ("residual_wall_time_s", "mlp_residual_wall_time_s"),
    ):
        value = mlp.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            compact[target] = value

    residual_wall_time = compact.get("mlp_residual_wall_time_s")
    if isinstance(residual_wall_time, (int, float)):
        compact["mlp_residual_compute"] = max(
            0.0, float(residual_wall_time) * RAW_RESIDUAL_FLOPS_PER_SECOND
        )
    return compact


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    returncode = 1
    timings: dict[str, float] = {}
    started_at = time.monotonic()
    try:
        work = Path(tempfile.mkdtemp(prefix="whest-fly-"))
        archive = work / "dataset.tar.gz"
        estimator = work / "estimator.py"
        dataset_root = work / "dataset"
        dataset_root.mkdir()
        step_started_at = time.monotonic()
        _download(args.dataset_url, archive)
        timings["worker_download_dataset_s"] = time.monotonic() - step_started_at
        step_started_at = time.monotonic()
        _download(args.estimator_url, estimator)
        timings["worker_download_estimator_s"] = time.monotonic() - step_started_at
        step_started_at = time.monotonic()
        dataset = _extract(archive, dataset_root)
        timings["worker_extract_dataset_s"] = time.monotonic() - step_started_at

        cmd = [
            sys.executable,
            "scripts/remote_whest_run.py",
            "--estimator",
            str(estimator),
            "--dataset",
            str(dataset),
            "--split",
            args.split,
            "--runner",
            args.runner,
            "--n-mlps",
            "1",
            "--flop-budget",
            str(args.flop_budget),
            "--wall-time-limit",
            str(args.wall_time_limit),
            "--residual-wall-time-multiplier",
            str(args.residual_wall_time_multiplier),
            "--max-threads",
            str(args.max_threads),
            "--format",
            args.format,
            "--detail",
            args.detail,
        ]
        if args.mode:
            cmd.extend(["--mode", args.mode])
        step_started_at = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        timings["worker_whestbench_s"] = time.monotonic() - step_started_at
        timings["worker_total_s"] = time.monotonic() - started_at
        if args.format == "json":
            payload = _json_object_from_output(proc.stdout)
            result = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(result, dict):
                one_mlp_fields = _copy_one_mlp_fields(result)
                compact_result = {key: value for key, value in result.items() if key != "per_mlp"}
                compact_result.update(one_mlp_fields)
                compact_result.update(timings)
                print("WHEST_RESULT_JSON " + json.dumps(compact_result, sort_keys=True), flush=True)
            else:
                print(proc.stdout, end="")
        else:
            print(proc.stdout, end="")
        returncode = proc.returncode
    finally:
        if args.done_sentinel:
            print(f"{args.done_sentinel} returncode={returncode}", flush=True)
        if args.linger_after_result > 0:
            time.sleep(args.linger_after_result)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
