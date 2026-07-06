#!/usr/bin/env python3
"""Score a truth-bank shard and return predictions plus metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "local_engine.py").is_file():
    REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_VERSION = "fly-bank-gate-entrypoint-v1"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument("--mode")
    args = parser.parse_args(argv)
    if args.shard_count <= 0:
        raise SystemExit("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")
    return args


def _load_estimator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("bank_gate_estimator", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not import estimator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    estimator_cls = getattr(module, "Estimator", None)
    if estimator_cls is None:
        raise SystemExit(f"{path} does not define Estimator")
    return estimator_cls()


def _setup_estimator(estimator: Any, *, width: int, depth: int, flop_budget: int, seed: int) -> None:
    setup = getattr(estimator, "setup", None)
    if setup is None:
        return
    from whestbench import SetupContext

    scratch = tempfile.mkdtemp(prefix="bank-gate-scratch-")
    ctx = SetupContext(
        width=width,
        depth=depth,
        flop_budget=flop_budget,
        api_version="1",
        scratch_dir=scratch,
        submission_dir=str(Path.cwd()),
        seed=seed,
    )
    setup(ctx)


def _predict_one(estimator: Any, *, seed: int, truth: np.ndarray, flop_budget: int) -> dict[str, Any]:
    import flopscope as flops
    from local_engine import build_mlp

    mlp = build_mlp(width=int(truth.shape[1]), depth=int(truth.shape[0]), seed=int(seed))
    started_at = time.monotonic()
    with flops.BudgetContext(flop_budget=flop_budget, quiet=True) as ctx:
        prediction = estimator.predict(mlp, flop_budget)
    wall_time_s = time.monotonic() - started_at
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} != truth shape {truth.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains non-finite values")
    diff = pred - truth
    return {
        "prediction": pred.tolist(),
        "prediction_shape": list(pred.shape),
        "all_layers_mse": float(np.mean(diff * diff)),
        "final_layer_mse": float(np.mean(diff[-1] * diff[-1])),
        "flops_used": int(ctx.flops_used),
        "wall_time_s": wall_time_s,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.mode:
        os.environ["WHEST_K3_MODE"] = args.mode
    bank = np.load(args.bank)
    seeds = bank["seeds"]
    truths = bank["truths"].astype(np.float64)
    n_rows = int(seeds.shape[0])
    indices = [
        index
        for index in range(n_rows)
        if index * args.shard_count // n_rows == args.shard_index
    ]
    estimator = _load_estimator(args.estimator)
    _setup_estimator(
        estimator,
        width=int(truths.shape[2]),
        depth=int(truths.shape[1]),
        flop_budget=args.flop_budget,
        seed=args.setup_seed,
    )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for bank_index in indices:
        seed = int(seeds[bank_index])
        try:
            record = _predict_one(
                estimator,
                seed=seed,
                truth=truths[bank_index],
                flop_budget=args.flop_budget,
            )
            record.update({"bank_index": bank_index, "seed": seed})
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - return structured failure for research runs.
            failures.append(
                {
                    "bank_index": bank_index,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    payload = {
        "task": "bank",
        "script_version": SCRIPT_VERSION,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "bank_rows": n_rows,
        "records": records,
        "failures": failures,
        "n_records": len(records),
        "n_failures": len(failures),
        "summary": {
            "all_layers_mse_mean": (
                sum(record["all_layers_mse"] for record in records) / len(records)
                if records
                else None
            ),
            "final_layer_mse_mean": (
                sum(record["final_layer_mse"] for record in records) / len(records)
                if records
                else None
            ),
            "flops_used_mean": (
                sum(record["flops_used"] for record in records) / len(records)
                if records
                else None
            ),
        },
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
