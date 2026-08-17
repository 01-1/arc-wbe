#!/usr/bin/env python3
"""Machine-side tail-aware projection proxy gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_engine import build_mlp  # noqa: E402

SCRIPT_VERSION = "tail-projection-proxy-gate-fly-v1"
WIDTH = 256
DEPTH = 32
SAMPLES = 1024
LAYERS = (4, 8, 12, 16, 20, 24, 28, 30)
TOP_K = 32
HUTCH_PROBES = 8
ONE_COORDS = 64
EPS = 1.0e-30


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument("--mode")
    return parser.parse_args(argv)


def weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for w in weights:
        digest.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return digest.hexdigest()


def np_weights(seed: int) -> list[np.ndarray]:
    mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=int(seed))
    return [np.asarray(w, dtype=np.float32) for w in mlp.weights]


def sample_seed(bank_seed: int, bank_index: int) -> int:
    label = f"tail-projection-proxy-v1:{bank_seed}:{bank_index}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def forward_all(weights: list[np.ndarray], seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    half = SAMPLES // 2
    x0 = rng.standard_normal((half, WIDTH), dtype=np.float32)
    x = np.concatenate([x0, -x0], axis=0)
    acts: list[np.ndarray] = []
    y = x
    for w in weights:
        y = np.maximum(y @ w, 0.0).astype(np.float32, copy=False)
        acts.append(y.copy())
    pred = np.asarray([a.astype(np.float64).mean(axis=0) for a in acts], dtype=np.float64)
    return acts, pred


def continue_from(y: np.ndarray, weights: list[np.ndarray], start_weight: int) -> np.ndarray:
    z = y.astype(np.float32, copy=True)
    for j in range(start_weight, DEPTH):
        z = np.maximum(z @ weights[j], 0.0).astype(np.float32, copy=False)
    return z.astype(np.float64).mean(axis=0)


def tail_diag_kernel(acts: list[np.ndarray], weights: list[np.ndarray], layer: int, rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros(WIDTH, dtype=np.float64)
    for _ in range(HUTCH_PROBES):
        b = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(SAMPLES, WIDTH))
        for j in range(DEPTH - 1, layer, -1):
            b = b * (acts[j] > 0.0)
            b = b @ weights[j].T
        b = b * (acts[layer] > 0.0)
        acc += np.mean(b.astype(np.float64) * b.astype(np.float64), axis=0)
    return acc / float(HUTCH_PROBES)


def successor_energy(weights: list[np.ndarray], layer: int) -> np.ndarray:
    if layer + 1 >= DEPTH:
        return np.ones(WIDTH, dtype=np.float64)
    return np.sum(weights[layer + 1].astype(np.float64) ** 2, axis=1)


def final_mse(vec: np.ndarray, truth_final: np.ndarray) -> float:
    return float(np.mean((vec - truth_final) ** 2))


def apply_coord_shift(
    acts: list[np.ndarray],
    weights: list[np.ndarray],
    layer: int,
    coords: np.ndarray,
    delta: np.ndarray,
    truth_final: np.ndarray,
) -> float:
    y = acts[layer].copy()
    y[:, coords] = np.maximum(y[:, coords] - delta[coords].astype(np.float32), 0.0)
    return final_mse(continue_from(y, weights, layer + 1), truth_final)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata(np.asarray(x, dtype=np.float64))
    yr = rankdata(np.asarray(y, dtype=np.float64))
    xr -= xr.mean()
    yr -= yr.mean()
    denom = math.sqrt(float(np.sum(xr * xr) * np.sum(yr * yr)))
    return float(np.sum(xr * yr) / denom) if denom > 0.0 else float("nan")


def process_one(bank_index: int, original_index: int, seeds: np.ndarray, truths: np.ndarray, checksums: np.ndarray) -> dict[str, Any]:
    started = time.monotonic()
    seed = int(seeds[bank_index])
    weights = np_weights(seed)
    checksum = weights_sha256(weights)
    truth = truths[bank_index].astype(np.float64)
    rng = np.random.default_rng(sample_seed(seed, int(original_index)))
    acts, pred = forward_all(weights, sample_seed(seed, int(original_index)))
    base_final = final_mse(pred[-1], truth[-1])
    rows = []
    for layer in LAYERS:
        e = pred[layer] - truth[layer]
        local_score = e * e
        tail_diag = tail_diag_kernel(acts, weights, layer, rng)
        succ = successor_energy(weights, layer)
        tail_score = local_score * tail_diag
        succ_score = local_score * succ
        local_top = np.argsort(local_score)[-TOP_K:]
        tail_top = np.argsort(tail_score)[-TOP_K:]
        succ_top = np.argsort(succ_score)[-TOP_K:]
        local_mse = apply_coord_shift(acts, weights, layer, local_top, e, truth[-1])
        tail_mse = apply_coord_shift(acts, weights, layer, tail_top, e, truth[-1])
        succ_mse = apply_coord_shift(acts, weights, layer, succ_top, e, truth[-1])

        pool = np.unique(np.concatenate([local_top, tail_top, succ_top, np.argsort(tail_score)[-ONE_COORDS:]]))
        one_improve = []
        one_local = []
        one_tail = []
        one_succ = []
        for coord in pool:
            mse = apply_coord_shift(acts, weights, layer, np.asarray([coord]), e, truth[-1])
            one_improve.append(base_final - mse)
            one_local.append(local_score[coord])
            one_tail.append(tail_score[coord])
            one_succ.append(succ_score[coord])
        one_improve_a = np.asarray(one_improve, dtype=np.float64)
        rows.append(
            {
                "layer": int(layer),
                "base_final_mse": base_final,
                "local_top_mse": local_mse,
                "tail_top_mse": tail_mse,
                "successor_top_mse": succ_mse,
                "local_reduction": base_final - local_mse,
                "tail_reduction": base_final - tail_mse,
                "successor_reduction": base_final - succ_mse,
                "tail_over_local_reduction": (base_final - tail_mse) / max(base_final - local_mse, EPS),
                "tail_over_successor_reduction": (base_final - tail_mse) / max(base_final - succ_mse, EPS),
                "tail_wins_local": bool(tail_mse < local_mse),
                "tail_wins_successor": bool(tail_mse < succ_mse),
                "spearman_local": corr(np.asarray(one_local), one_improve_a),
                "spearman_tail": corr(np.asarray(one_tail), one_improve_a),
                "spearman_successor": corr(np.asarray(one_succ), one_improve_a),
                "pool_size": int(pool.size),
                "tail_diag_cv": float(np.std(tail_diag) / max(float(np.mean(tail_diag)), EPS)),
            }
        )
    return {
        "bank_index": int(original_index),
        "shard_bank_index": int(bank_index),
        "seed": seed,
        "weights_sha256": checksum,
        "bank_weights_sha256": str(checksums[bank_index]),
        "checksum_ok": checksum == str(checksums[bank_index]),
        "base_final_mse": base_final,
        "rows": rows,
        "wall_time_s": float(time.monotonic() - started),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    bank = np.load(args.bank)
    seeds = bank["seeds"]
    truths = bank["truths"]
    checksums = bank["weights_sha256"]
    original_indices = bank["original_indices"] if "original_indices" in bank.files else np.arange(seeds.shape[0])
    n_rows = int(seeds.shape[0])
    indices = [idx for idx in range(n_rows) if idx * args.shard_count // n_rows == args.shard_index]
    records = []
    failures = []
    for bank_index in indices:
        try:
            records.append(process_one(bank_index, int(original_indices[bank_index]), seeds, truths, checksums))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "bank_index": int(bank_index),
                    "seed": int(seeds[bank_index]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    print(
        json.dumps(
            {
                "task": "tail_projection_proxy_gate",
                "script_version": SCRIPT_VERSION,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "bank_rows": n_rows,
                "records": records,
                "failures": failures,
                "n_records": len(records),
                "n_failures": len(failures),
                "config": {
                    "samples": SAMPLES,
                    "layers": list(LAYERS),
                    "top_k": TOP_K,
                    "hutch_probes": HUTCH_PROBES,
                    "one_coords": ONE_COORDS,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
