#!/usr/bin/env python3
"""Shared helpers for WhestBench cloud runners."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATASET = "hf://aicrowd/arc-whestbench-public-2026@v1-phase1"
DEFAULT_SPLIT = "mini"


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def dataset_name(path: Path) -> str:
    return path.resolve().name


def dataset_fingerprint(path: Path) -> str:
    """Cheap fingerprint for matching a staged dataset to a cloud cache."""

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix()
        stat = child.stat()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if child.name == "metadata.json":
            digest.update(child.read_bytes())
    return digest.hexdigest()[:16]


def ensure_randomized_dataset(
    *,
    n_mlps: int,
    seed: int,
    source_dataset: str = DEFAULT_SOURCE_DATASET,
    split: str = DEFAULT_SPLIT,
    output: Path | None = None,
    force: bool = False,
    uv_cache_dir: str = "/i/e/.uv-cache",
) -> Path:
    """Stage an HF-derived baked dataset subset locally without recomputing labels."""

    if n_mlps <= 0:
        raise SystemExit("--n-mlps must be positive")

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/randomize_whest_dataset.py",
        "--dataset",
        source_dataset,
        "--split",
        split,
        "--n-mlps",
        str(n_mlps),
        "--seed",
        str(seed),
    ]
    if output is not None:
        cmd.extend(["--output", str(output)])
    if force:
        cmd.append("--force")

    env = dict(os.environ)
    env["UV_CACHE_DIR"] = uv_cache_dir
    env.setdefault("HF_HOME", "/i/e/.cache/huggingface")
    env.setdefault("HF_DATASETS_CACHE", "/i/e/.cache/huggingface/datasets")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout)
    staged = Path(proc.stdout.strip().splitlines()[-1])
    if not staged.is_absolute():
        staged = REPO_ROOT / staged
    if not staged.is_dir():
        raise SystemExit(f"randomized dataset was not created: {staged}")
    return staged


def whest_env(max_threads: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/i/e/.cache/huggingface")
    env.setdefault("HF_DATASETS_CACHE", "/i/e/.cache/huggingface/datasets")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("WHEST_SKIP_HARDWARE_FALLBACK_PROBES", "1")
    env.setdefault("FLOPSCOPE_GPU", "0")
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    if max_threads is not None and max_threads > 0:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = str(max_threads)
    return env
