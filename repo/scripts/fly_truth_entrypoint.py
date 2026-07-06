#!/usr/bin/env python3
"""Generate one Fly truth-bank row for a freshly seeded He MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "local_engine.py").is_file():
    REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_VERSION = "fly-truth-entrypoint-v1"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--target-seconds", type=float, default=60.0)
    parser.add_argument("--chunk-pairs", type=int, default=1024)
    parser.add_argument("--min-pairs", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.width <= 0 or args.depth <= 0:
        raise SystemExit("--width and --depth must be positive")
    if args.target_seconds <= 0:
        raise SystemExit("--target-seconds must be positive")
    if args.chunk_pairs <= 0 or args.min_pairs <= 0:
        raise SystemExit("--chunk-pairs and --min-pairs must be positive")
    return args


def _build_weights(width: int, depth: int, seed: int) -> list[np.ndarray]:
    from local_engine import build_mlp

    mlp = build_mlp(width=width, depth=depth, seed=seed)
    return [np.ascontiguousarray(np.asarray(weight), dtype=np.float32) for weight in mlp.weights]


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def _run_truth(
    weights: list[np.ndarray],
    *,
    seed: int,
    target_seconds: float,
    chunk_pairs: int,
    min_pairs: int,
) -> tuple[np.ndarray, int, float]:
    rng = np.random.default_rng(seed ^ 0x5EED_5EED_5EED_5EED)
    depth = len(weights)
    width = int(weights[0].shape[0])
    sums = np.zeros((depth, width), dtype=np.float64)
    sample_count = 0
    started = time.monotonic()

    while sample_count < min_pairs * 2 or time.monotonic() - started < target_seconds:
        z = rng.standard_normal((chunk_pairs, width)).astype(np.float32)
        x = np.concatenate((z, -z), axis=0)
        for layer_index, weight in enumerate(weights):
            x = x @ weight
            np.maximum(x, 0.0, out=x)
            sums[layer_index] += x.sum(axis=0, dtype=np.float64)
        sample_count += x.shape[0]

    wall_time = time.monotonic() - started
    return sums / float(sample_count), sample_count, wall_time


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    weights = _build_weights(args.width, args.depth, args.seed)
    weights_sha256 = _weights_sha256(weights)
    truth, sample_count, wall_time = _run_truth(
        weights,
        seed=args.seed,
        target_seconds=args.target_seconds,
        chunk_pairs=args.chunk_pairs,
        min_pairs=args.min_pairs,
    )
    flops = int(sample_count) * 2 * args.depth * args.width * args.width
    payload = {
        "task": "truth",
        "script_version": SCRIPT_VERSION,
        "mlp_index": args.mlp_index,
        "seed": args.seed,
        "width": args.width,
        "depth": args.depth,
        "dtype": {
            "weights": "float32",
            "activations": "float32",
            "accumulator": "float64",
            "truth": "float64",
        },
        "antithetic": True,
        "chunk_pairs": args.chunk_pairs,
        "sample_count": sample_count,
        "wall_time_s": wall_time,
        "flops": flops,
        "flops_per_sample": 2 * args.depth * args.width * args.width,
        "weights_sha256": weights_sha256,
        "truth": truth.tolist(),
        "truth_shape": list(truth.shape),
        "final_layer_mean_min": float(np.min(truth[-1])),
        "final_layer_mean_max": float(np.max(truth[-1])),
        "final_layer_mean_mean": float(np.mean(truth[-1])),
        "finite": bool(np.isfinite(truth).all()),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
