#!/usr/bin/env python3
"""Assemble Fly truth JSONL rows into the tracked research truth bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "truth_bank")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--verify-index", type=int, default=0)
    return parser.parse_args(argv)


def _weight_checksum(width: int, depth: int, seed: int) -> str:
    from local_engine import build_mlp

    mlp = build_mlp(width=width, depth=depth, seed=seed)
    digest = hashlib.sha256()
    for weight in mlp.weights:
        digest.update(np.ascontiguousarray(np.asarray(weight), dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_number}: expected JSON object")
        if row.get("task") == "truth":
            rows.append(row)
    return rows


def _validate_rows(rows: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("mlp_index")
        if not isinstance(index, int):
            raise SystemExit("truth row missing integer mlp_index")
        if index in by_index:
            raise SystemExit(f"duplicate truth row for mlp_index={index}")
        by_index[index] = row
    missing = [index for index in range(expected_count) if index not in by_index]
    if missing:
        raise SystemExit(f"missing truth rows: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return [by_index[index] for index in range(expected_count)]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    rows = _validate_rows(_read_rows(args.input_jsonl), args.expected_count)
    seeds = np.array([int(row["seed"]) for row in rows], dtype=np.uint64)
    truths = np.array([row["truth"] for row in rows], dtype=np.float64)
    if truths.shape != (args.expected_count, 32, 256):
        raise SystemExit(f"expected truth shape {(args.expected_count, 32, 256)}, got {truths.shape}")
    if not np.isfinite(truths).all():
        raise SystemExit("truth bank contains non-finite values")

    verify_row = rows[args.verify_index]
    local_checksum = _weight_checksum(
        int(verify_row["width"]),
        int(verify_row["depth"]),
        int(verify_row["seed"]),
    )
    checksum_ok = local_checksum == verify_row.get("weights_sha256")
    if not checksum_ok:
        raise SystemExit(
            f"checksum mismatch for index {args.verify_index}: local={local_checksum} "
            f"remote={verify_row.get('weights_sha256')}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "truth_bank.npz",
        seeds=seeds,
        truths=truths,
        weights_sha256=np.array([row["weights_sha256"] for row in rows]),
    )
    sample_counts = [int(row["sample_count"]) for row in rows]
    flops = [int(row["flops"]) for row in rows]
    wall_times = [float(row["wall_time_s"]) for row in rows]
    metadata = {
        "bank_version": "fly-truth-bank-v1",
        "source_jsonl": str(args.input_jsonl),
        "n_mlps": args.expected_count,
        "width": 256,
        "depth": 32,
        "truth_shape": list(truths.shape),
        "seed_derivation": {
            "method": "sha256(label:index) first 63 bits",
            "label": "arc-whest-fly-truth-bank-20260706-v1",
            "excluded_exact_seeds": [11, 22, 33],
            "grader_fixture_seeds": "not read or used; this bank uses fresh deterministic research seeds",
        },
        "dtype": rows[0].get("dtype"),
        "antithetic": True,
        "script_version": rows[0].get("script_version"),
        "sample_count": {
            "min": min(sample_counts),
            "max": max(sample_counts),
            "mean": sum(sample_counts) / len(sample_counts),
            "per_mlp": sample_counts,
        },
        "flops": {
            "min": min(flops),
            "max": max(flops),
            "mean": sum(flops) / len(flops),
            "per_mlp": flops,
        },
        "wall_time_s": {
            "min": min(wall_times),
            "max": max(wall_times),
            "mean": sum(wall_times) / len(wall_times),
            "per_mlp": wall_times,
        },
        "final_layer_mean": {
            "min": float(np.min(truths[:, -1, :])),
            "max": float(np.max(truths[:, -1, :])),
            "mean": float(np.mean(truths[:, -1, :])),
        },
        "checksum_verification": {
            "index": args.verify_index,
            "seed": int(verify_row["seed"]),
            "local_weights_sha256": local_checksum,
            "remote_weights_sha256": verify_row["weights_sha256"],
            "ok": checksum_ok,
        },
        "rows": [
            {
                key: row[key]
                for key in (
                    "mlp_index",
                    "seed",
                    "sample_count",
                    "wall_time_s",
                    "flops",
                    "weights_sha256",
                    "final_layer_mean_min",
                    "final_layer_mean_max",
                    "final_layer_mean_mean",
                )
            }
            for row in rows
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_mlps": args.expected_count,
                "sample_count_min": min(sample_counts),
                "sample_count_max": max(sample_counts),
                "sample_count_mean": sum(sample_counts) / len(sample_counts),
                "flops_mean": sum(flops) / len(flops),
                "checksum_verification": metadata["checksum_verification"],
                "final_layer_mean": metadata["final_layer_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
