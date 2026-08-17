#!/usr/bin/env python3
"""Fly payload for the preregistered Sobol triangular Stage-A v2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = Path(__file__).resolve().parent
for path in (ROOT, PAYLOAD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sobol_runtime_feasibility_generator_20260710 import sobol_normal_rows

import flopscope.numpy as fnp
from estimator import (
    _DEEP_VARIANCE_MATCH_STRENGTH,
    _MIN_VARIANCE,
    _gaussian_relu_variance,
    _hadamard,
    _strassen_matmul,
    _zero_mean_relu_mean_cov,
)
from local_engine import build_mlp


WIDTH = 256
DEPTH = 32
POSITIVE_ROWS = 4096
BLOCKS = 16
PILOTS = 8
PROBES = 8
SOBOL_STREAM = 0x5B01_0710
PILOT_STREAM = 0x7A11_0710
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "sobol-triangular-lt-v2"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int) -> int:
    return int((seed ^ stream) % (1 << 32))


def _hadamard_rows(seed: int, width: int, blocks: int) -> np.ndarray:
    """Return independent positive half-bases, never interleaved pairs."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows: list[np.ndarray] = []
    for _ in range(blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        rows.append(base * flips[None, :])
    return np.concatenate(rows, axis=0)


def _recolor_and_propagate(weights: list[fnp.ndarray], rows: np.ndarray) -> np.ndarray:
    """Apply the fixed legal current-route transforms to one positive set."""
    positive_rows = fnp.array(rows, dtype=fnp.float32)
    pre_half = _strassen_matmul(positive_rows, weights[0], 3)
    y = fnp.concatenate(
        (fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0
    )
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3)
        / float(centered.shape[0])
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(weights[0].shape[1])
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = fnp.mean(pre, axis=0).astype(fnp.float32)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(
                pre_mean.astype(fnp.float64),
                fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
            )
            sample_mean_1 = fnp.mean(x, axis=0).astype(fnp.float32)
            centered_apply = x - sample_mean_1[None, :]
            sample_var = fnp.maximum(
                fnp.mean(centered_apply * centered_apply, axis=0).astype(fnp.float64),
                _MIN_VARIANCE,
            )
            scale = (
                1.0
                + _DEEP_VARIANCE_MATCH_STRENGTH
                * (fnp.sqrt(target_var / sample_var) - 1.0)
            ).astype(fnp.float32)
            x = centered_apply * scale[None, :] + sample_mean_1[None, :]
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def _pilot_importance(
    weights: list[fnp.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return squared-gradient importance and the stable descending order."""
    rng = np.random.default_rng(_derived_seed(seed, PILOT_STREAM))
    pilots = rng.standard_normal((PILOTS, WIDTH))
    pilots /= np.linalg.norm(pilots, axis=1, keepdims=True)
    probes = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(PROBES, WIDTH))
    importance = np.zeros(WIDTH, dtype=np.float64)
    for pilot in pilots.astype(np.float32):
        h = fnp.array(pilot[None, :], dtype=fnp.float32)
        pre_layers: list[fnp.ndarray] = []
        for weight in weights:
            pre = _strassen_matmul(h, weight, 3)
            pre_layers.append(pre)
            h = fnp.maximum(pre, 0.0)
        for probe in probes.astype(np.float32):
            grad = fnp.array(probe[None, :], dtype=fnp.float32)
            for layer_idx in range(DEPTH - 1, 0, -1):
                grad = grad * (pre_layers[layer_idx] > 0.0)
                grad = fnp.matmul(grad, weights[layer_idx].T)
            grad_np = np.asarray(grad[0], dtype=np.float64)
            importance += grad_np * grad_np
    importance /= float(PILOTS * PROBES)
    return importance, np.argsort(-importance, kind="stable")


def _triangular_rows(
    rows: np.ndarray, weights: list[np.ndarray], order: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    w0_order = np.asarray(weights[0][:, order], dtype=np.float64)
    q, r = np.linalg.qr(w0_order, mode="complete")
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    q = q * signs[None, :]
    r = signs[:, None] * r
    transformed = np.asarray(rows, dtype=np.float64) @ q.T
    check_rows = min(16, rows.shape[0])
    left = transformed[:check_rows].astype(np.float32) @ w0_order.astype(np.float32)
    right = rows[:check_rows].astype(np.float32) @ r.astype(np.float32)
    error = left.astype(np.float64) - right.astype(np.float64)
    max_abs = float(np.max(np.abs(error)))
    max_rel = float(max_abs / max(np.max(np.abs(right)), 1e-12))
    return transformed.astype(np.float32), {"max_abs": max_abs, "max_rel": max_rel}


def _importance_shares(importance: np.ndarray, order: np.ndarray) -> dict[str, float]:
    total = max(float(np.sum(importance)), 1e-300)
    return {str(k): float(np.sum(importance[order[:k]]) / total) for k in (1, 2, 4, 8, 16, 32)}


def run_one(shard_index: int, bank_path: Path) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")
    weights_f32 = [fnp.array(w, dtype=fnp.float32) for w in weights]

    sobol_rows = sobol_normal_rows(_derived_seed(seed, SOBOL_STREAM))
    importance, order = _pilot_importance(weights_f32, seed)
    triangular_rows, qr_error = _triangular_rows(sobol_rows, weights, order)
    estimates = {
        "current": _recolor_and_propagate(
            weights_f32,
            _hadamard_rows(_derived_seed(seed, CURRENT_STREAM), WIDTH, BLOCKS),
        ),
        "sobol_sphere": _recolor_and_propagate(weights_f32, sobol_rows),
        "sobol_triangular": _recolor_and_propagate(weights_f32, triangular_rows),
    }

    # Truth is read only after all route estimates and diagnostics are fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    mse = {name: float(np.mean((value - truth) ** 2)) for name, value in estimates.items()}
    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": True,
        "weights_sha256": actual,
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "positive_rows": POSITIVE_ROWS,
            "antithetic_rows": 2 * POSITIVE_ROWS,
            "blocks": BLOCKS,
            "pilots": PILOTS,
            "probes": PROBES,
            "sobol_transform": "reviewed sobol_normal_rows(seed)->shared 4096 sphere rows",
            "first_layer": "global exact ReLU mean/covariance recolor",
            "first_successor": "fp32 centered_apply, strength 1.5",
            "propagation": "fp32 L3 Strassen",
            "radial_multiplier": None,
        },
        "mse": mse,
        "estimates": {name: value.tolist() for name, value in estimates.items()},
        "truth_final": truth.tolist(),
        "importance_shares": _importance_shares(importance, order),
        "importance_total": float(np.sum(importance)),
        "qr_error": qr_error,
        "qr_order_head": [int(x) for x in order[:32]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank)
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
