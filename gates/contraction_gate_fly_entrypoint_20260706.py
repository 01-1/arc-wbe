#!/usr/bin/env python3
"""Machine-side the reference entrant contraction gate entrypoint.

Runs one truth-bank shard on Fly.  It rebuilds the bank MLP locally on the
Machine, uses the shard's bank truth row for toy scoring, and returns compact
per-MLP summary statistics only.
"""

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

SCRIPT_VERSION = "the reference entrant-contraction-gate-fly-v1"
WIDTH = 256
DEPTH = 32
INJECTION_LAYERS = (2, 8, 16, 24)
PERTURBATION_TYPES = ("iid_gaussian", "bias_all", "top2_bias", "orthogonal_bias")
Q1_SAMPLES = 1024
Q2_SAMPLES = 512
EPS_NOISE = 0.08
EPS_BIAS = 0.035


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", type=Path, required=True, help="Ignored; present for fly-bank wrapper compatibility.")
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


def slope(xs: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(y) & (y > 0)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return float("nan")
    ly = np.log(y)
    xm = float(x.mean())
    ym = float(ly.mean())
    return float(np.sum((x - xm) * (ly - ym)) / np.sum((x - xm) ** 2))


def forward_to_layer(x: np.ndarray, weights: list[np.ndarray], layer: int) -> np.ndarray:
    y = x.astype(np.float32, copy=False)
    for j in range(layer + 1):
        y = np.maximum(y @ weights[j], 0.0).astype(np.float32, copy=False)
    return y


def continue_collect(y: np.ndarray, weights: list[np.ndarray], start_weight: int) -> np.ndarray:
    rows = []
    z = y.astype(np.float32, copy=False)
    for j in range(start_weight, DEPTH):
        z = np.maximum(z @ weights[j], 0.0).astype(np.float32, copy=False)
        rows.append(z.astype(np.float64).mean(axis=0))
    return np.asarray(rows, dtype=np.float64)


def top_and_orth_dirs(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yd = y.astype(np.float64, copy=False)
    yc = yd - yd.mean(axis=0, keepdims=True)
    cov = (yc.T @ yc) / max(1, y.shape[0] - 1)
    evals, evecs = np.linalg.eigh((cov + cov.T) * 0.5)
    order = np.argsort(evals)[::-1]
    top = evecs[:, order[:2]]
    top_dir = top[:, 0] + 0.5 * top[:, 1]
    top_dir = top_dir / max(np.linalg.norm(top_dir), 1e-30) * math.sqrt(WIDTH)
    v = rng.standard_normal(WIDTH)
    for col in range(min(8, evecs.shape[1])):
        u = evecs[:, order[col]]
        v = v - np.dot(v, u) * u
    orth_dir = v / max(np.linalg.norm(v), 1e-30) * math.sqrt(WIDTH)
    return top_dir.astype(np.float32), orth_dir.astype(np.float32)


def q1_rows(seed: int, weights: list[np.ndarray], bank_index: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(9_100_000 + bank_index)
    base = rng.standard_normal((Q1_SAMPLES // 2, WIDTH), dtype=np.float32)
    x = np.concatenate([base, -base], axis=0)
    rows: list[dict[str, Any]] = []
    for k in INJECTION_LAYERS:
        y = forward_to_layer(x, weights, k)
        base_down = continue_collect(y, weights, k + 1)
        layer_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        top_dir, orth_dir = top_and_orth_dirs(y, rng)
        perturbations = {
            "iid_gaussian": rng.standard_normal(y.shape, dtype=np.float32) * (EPS_NOISE * layer_rms),
            "bias_all": np.full((1, WIDTH), EPS_BIAS * layer_rms, dtype=np.float32),
            "top2_bias": (EPS_BIAS * layer_rms * top_dir)[None, :],
            "orthogonal_bias": (EPS_BIAS * layer_rms * orth_dir)[None, :],
        }
        for ptype, delta in perturbations.items():
            yp = np.maximum(y + delta, 0.0).astype(np.float32, copy=False)
            pert_down = continue_collect(yp, weights, k + 1)
            diff = pert_down - base_down
            mse = np.mean(diff * diff, axis=1)
            layers = np.arange(k + 1, DEPTH)
            amp_s = slope(layers, np.sqrt(np.maximum(mse, 1e-300)))
            mse_s = slope(layers, np.maximum(mse, 1e-300))
            rows.append(
                {
                    "bank_index": bank_index,
                    "seed": int(seed),
                    "injection_layer": int(k),
                    "perturbation_type": ptype,
                    "mse_factor_per_layer": float(math.exp(mse_s)),
                    "mse_log_slope": float(mse_s),
                    "amplitude_factor_per_layer": float(math.exp(amp_s)),
                    "amplitude_log_slope": float(amp_s),
                    "terminal_over_first_mse": float(mse[-1] / max(mse[0], 1e-300)),
                    "mse_by_layer": [float(v) for v in mse],
                }
            )
    return rows


def rank_reproject(y: np.ndarray, rank: int) -> np.ndarray:
    yd = y.astype(np.float64, copy=False)
    mu = yd.mean(axis=0, keepdims=True)
    yc = yd - mu
    u, s, vt = np.linalg.svd(yc, full_matrices=False)
    z = mu + (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    return np.maximum(z, 0.0).astype(np.float32)


def toy_row(seed: int, weights: list[np.ndarray], truth: np.ndarray, bank_index: int, toy: str) -> dict[str, Any]:
    rng = np.random.default_rng(12_300_000 + 97 * bank_index + (0 if toy == "plain_particles_n512" else 1))
    half = Q2_SAMPLES // 2
    x0 = rng.standard_normal((half, WIDTH), dtype=np.float32)
    y = np.concatenate([x0, -x0], axis=0)
    preds = []
    for w in weights:
        y = np.maximum(y @ w, 0.0).astype(np.float32, copy=False)
        if toy == "rank2_reproject_n512":
            y = rank_reproject(y, 2)
        preds.append(y.astype(np.float64).mean(axis=0))
    pred = np.asarray(preds, dtype=np.float64)
    mse = np.mean((pred - truth) ** 2, axis=1)
    hidden_layers = np.arange(2, 31)
    hidden_slope = slope(hidden_layers, mse[2:31])
    return {
        "bank_index": bank_index,
        "seed": int(seed),
        "toy": toy,
        "hidden_log_mse_slope_layers_2_30": float(hidden_slope),
        "hidden_mse_factor_per_layer": float(math.exp(hidden_slope)),
        "terminal_layer31_over_layer30": float(mse[31] / max(mse[30], 1e-300)),
        "final_mse": float(mse[31]),
        "all_layer_mse": float(np.mean(mse)),
        "mse_by_layer": [float(v) for v in mse],
    }


def process_one(
    bank_index: int,
    original_index: int,
    seeds: np.ndarray,
    truths: np.ndarray,
    bank_checksums: np.ndarray,
) -> dict[str, Any]:
    seed = int(seeds[bank_index])
    weights = np_weights(seed)
    checksum = weights_sha256(weights)
    truth = truths[bank_index].astype(np.float64)
    started = time.monotonic()
    return {
        "bank_index": int(original_index),
        "shard_bank_index": int(bank_index),
        "seed": seed,
        "weights_sha256": checksum,
        "bank_weights_sha256": str(bank_checksums[bank_index]),
        "checksum_ok": checksum == str(bank_checksums[bank_index]),
        "q1": q1_rows(seed, weights, int(original_index)),
        "q2": [
            toy_row(seed, weights, truth, int(original_index), "plain_particles_n512"),
            toy_row(seed, weights, truth, int(original_index), "rank2_reproject_n512"),
        ],
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
    payload = {
        "task": "contraction_gate",
        "script_version": SCRIPT_VERSION,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "bank_rows": n_rows,
        "records": records,
        "failures": failures,
        "n_records": len(records),
        "n_failures": len(failures),
        "config": {
            "q1_samples": Q1_SAMPLES,
            "q2_samples": Q2_SAMPLES,
            "injection_layers": list(INJECTION_LAYERS),
            "perturbation_types": list(PERTURBATION_TYPES),
            "eps_noise": EPS_NOISE,
            "eps_bias": EPS_BIAS,
        },
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
