#!/usr/bin/env python3
"""Run small WhestBench batches on Modal with many CPU cores."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
REMOTE_WORKDIR = "/workspace"
REMOTE_DATA_ROOT = "/datasets"
DEFAULT_VOLUME = "whestbench-cloud-datasets"


def _ignore(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts
        & {
            ".git",
            ".cache",
            ".pytest_cache",
            ".ruff_cache",
            ".uv-cache",
            ".venv",
            "__pycache__",
            "references",
            "assets",
        }
    )


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("build-essential", "ca-certificates", "git")
    .pip_install("uv", "protobuf")
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_WORKDIR, copy=True, ignore=_ignore)
    .workdir(REMOTE_WORKDIR)
    .run_commands("uv sync --frozen --no-dev")
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WHEST_SKIP_HARDWARE_FALLBACK_PROBES": "1",
            "FLOPSCOPE_GPU": "0",
            "UV_CACHE_DIR": "/tmp/uv-cache",
        }
    )
)

volume_name = os.environ.get("WHEST_MODAL_VOLUME", DEFAULT_VOLUME)
dataset_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
app = modal.App("whestbench-modal-runner", image=image, volumes={REMOTE_DATA_ROOT: dataset_volume})


@app.function(cpu=32, memory=65_536, timeout=3600, volumes={REMOTE_DATA_ROOT: dataset_volume})
def run_whest(
    *,
    dataset_dir: str,
    split: str,
    n_mlps: int,
    runner: str,
    flop_budget: int,
    wall_time_limit: float,
    residual_wall_time_multiplier: float,
    max_threads: int,
    mode: str | None,
    output_format: str,
    detail: str,
) -> str:
    dataset_volume.reload()
    cmd = [
        f"{REMOTE_WORKDIR}/.venv/bin/python",
        "scripts/remote_whest_run.py",
        "--dataset",
        dataset_dir,
        "--split",
        split,
        "--n-mlps",
        str(n_mlps),
        "--runner",
        runner,
        "--flop-budget",
        str(flop_budget),
        "--wall-time-limit",
        str(wall_time_limit),
        "--residual-wall-time-multiplier",
        str(residual_wall_time_multiplier),
        "--max-threads",
        str(max_threads),
        "--format",
        output_format,
        "--detail",
        detail,
    ]
    if mode:
        cmd.extend(["--mode", mode])
    proc = subprocess.run(
        cmd,
        cwd=REMOTE_WORKDIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


@app.local_entrypoint()
def main(
    n_mlps: int = 3,
    seed: int = 20260624,
    source_dataset: str = "hf://aicrowd/arc-whestbench-public-2026@v1-phase1",
    split: str = "mini",
    dataset: str | None = None,
    force_dataset: bool = False,
    runner: str = "subprocess",
    flop_budget: int = 272_000_000_000,
    wall_time_limit: float = 60.0,
    residual_wall_time_multiplier: float = 2.0,
    cores: int = 32,
    memory: int = 65_536,
    timeout: int = 3600,
    region: str = "us-east-1",
    mode: str | None = None,
    output_format: str = "plain",
    detail: str = "raw",
    uv_cache_dir: str = "/i/e/.uv-cache",
    remote_name: str | None = None,
) -> None:
    from scripts.cloud_whest_common import (
        dataset_fingerprint,
        dataset_name,
        ensure_randomized_dataset,
    )

    if dataset is None:
        dataset_path = ensure_randomized_dataset(
            n_mlps=n_mlps,
            seed=seed,
            source_dataset=source_dataset,
            split=split,
            force=force_dataset,
            uv_cache_dir=uv_cache_dir,
        )
    else:
        dataset_path = Path(dataset).resolve()
        if not dataset_path.is_dir():
            raise SystemExit(f"--dataset must be a directory: {dataset_path}")

    remote_name = remote_name or f"{dataset_name(dataset_path)}-{dataset_fingerprint(dataset_path)}"
    remote_path = f"{REMOTE_DATA_ROOT}/{remote_name}"
    print(f"Uploading {dataset_path} to Modal volume {volume_name}:{remote_path}")
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(dataset_path), f"/{remote_name}")

    output = run_whest.with_options(
        cpu=cores,
        memory=memory,
        timeout=timeout,
        region=region,
    ).remote(
        dataset_dir=remote_path,
        split=split,
        n_mlps=n_mlps,
        runner=runner,
        flop_budget=flop_budget,
        wall_time_limit=wall_time_limit,
        residual_wall_time_multiplier=residual_wall_time_multiplier,
        max_threads=cores,
        mode=mode,
        output_format=output_format,
        detail=detail,
    )
    print(output, end="")
