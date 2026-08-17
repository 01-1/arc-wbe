#!/usr/bin/env python3
"""Machine-side spherical Stein Haar fold-CV truth-bank gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
BLOCKS = 8
REPS = 3
STEIN_STREAM = 0x51E1_4A7E_2026_0710
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "spherical-stein-foldcv-v1"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _haar_rows(rng: np.random.Generator, width: int, blocks: int) -> np.ndarray:
    rows = []
    for _ in range(blocks):
        gaussian = rng.standard_normal((width, width))
        q, r = np.linalg.qr(gaussian)
        signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
        rows.append((math.sqrt(width) * (q * signs[None, :]).T).astype(np.float32))
    return np.concatenate(rows, axis=0)


def _stein_matrix(weights: list[np.ndarray]) -> np.ndarray:
    d = weights[0].shape[0]
    w0 = weights[0].astype(np.float64)
    a0 = w0 @ w0.T
    a = a0 - np.trace(a0) / d * np.eye(d)
    norm = float(np.linalg.norm(a, ord="fro"))
    if not np.isfinite(norm) or norm <= 1e-30:
        return np.zeros((d, d), dtype=np.float32)
    return (a * (math.sqrt(d) / norm)).astype(np.float32)


def _raw_haar_stein(weights: list[np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paired G,K for positive rows and their exact antipodes."""
    d = rows.shape[1]
    x_pos = rows.astype(np.float32, copy=False)
    x_neg = (-x_pos).astype(np.float32)
    a = _stein_matrix(weights)
    q = np.sum((x_pos @ a) * x_pos, axis=1, dtype=np.float32) / np.float32(d)
    v = (x_pos @ a - q[:, None] * x_pos).astype(np.float32)
    h_pos = x_pos
    h_neg = x_neg
    dh_pos = v
    dh_neg = -v
    for weight in weights:
        w = weight.astype(np.float32, copy=False)
        pre_pos = h_pos @ w
        pre_neg = h_neg @ w
        dpre_pos = dh_pos @ w
        dpre_neg = dh_neg @ w
        mask_pos = pre_pos > 0.0
        mask_neg = pre_neg > 0.0
        h_pos = np.maximum(pre_pos, 0.0).astype(np.float32, copy=False)
        h_neg = np.maximum(pre_neg, 0.0).astype(np.float32, copy=False)
        dh_pos = (dpre_pos * mask_pos).astype(np.float32, copy=False)
        dh_neg = (dpre_neg * mask_neg).astype(np.float32, copy=False)
    correction = np.float32(np.trace(a)) - np.float32(d) * q
    h_control_pos = dh_pos + h_pos * correction[:, None]
    h_control_neg = dh_neg + h_neg * correction[:, None]
    g = 0.5 * (h_pos.astype(np.float64) + h_neg.astype(np.float64))
    k = 0.5 * (h_control_pos.astype(np.float64) + h_control_neg.astype(np.float64))
    return g, k


def _stein_fold_cv(g: np.ndarray, k: np.ndarray) -> np.ndarray:
    pairs_per_base = WIDTH
    block_ids = np.arange(BLOCKS).repeat(pairs_per_base)
    estimates = []
    for held_blocks in (np.arange(0, 4), np.arange(4, 8)):
        held = np.isin(block_ids, held_blocks)
        train = ~held
        g_train, k_train = g[train], k[train]
        g_test, k_test = g[held], k[held]
        mean_g = np.mean(g_train, axis=0)
        mean_k = np.mean(k_train, axis=0)
        centered_g = g_train - mean_g
        centered_k = k_train - mean_k
        covariance = np.mean(centered_k * centered_g, axis=0)
        variance = np.mean(centered_k * centered_k, axis=0)
        ridge = 1e-3 * float(np.mean(variance))
        beta = covariance / (variance + ridge)
        estimates.append(np.mean(g_test, axis=0) - beta * np.mean(k_test, axis=0))
    return 0.5 * (estimates[0] + estimates[1])


def _hadamard_rows(rng: np.random.Generator, blocks: int) -> np.ndarray:
    base = np.asarray(_hadamard(WIDTH), dtype=np.float32)
    return np.concatenate(
        [base * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=WIDTH)[None, :]
         for _ in range(blocks)], axis=0)


def _current_route_estimate(mlp, seed: int, rep: int) -> np.ndarray:
    rng = np.random.default_rng((seed ^ CURRENT_STREAM ^ (rep * 0x9E3779B9)) % (1 << 63))
    weights = [w.astype(fnp.float32) for w in mlp.weights]
    rows = fnp.array(_hadamard_rows(rng, 16), dtype=fnp.float32)
    pre_half = _strassen_matmul(rows, weights[0], 3)
    y = fnp.concatenate((fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0)
    target_mean, target_cov = _zero_mean_relu_mean_cov(mlp.weights[0].T @ mlp.weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (_strassen_matmul(centered.T.astype(fnp.float32), centered.astype(fnp.float32), 3) / float(centered.shape[0])).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(mlp.width)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(pre_mean, fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64))
            sample_mean_1 = fnp.mean(x, axis=0).astype(fnp.float64)
            centered_layer = x - sample_mean_1[None, :]
            sample_var = fnp.maximum(fnp.mean(centered_layer * centered_layer, axis=0).astype(fnp.float64), _MIN_VARIANCE)
            scale = 1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (fnp.sqrt(target_var / sample_var) - 1.0)
            x = ((x - sample_mean_1.astype(fnp.float32)[None, :]) * scale.astype(fnp.float32)[None, :] + sample_mean_1.astype(fnp.float32)[None, :])
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def run_one(shard_index: int, bank_path: Path) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")
    radial = math.exp(0.5 * math.log(2.0) + math.lgamma((WIDTH + 1.0) / 2.0) - 0.5 * math.log(WIDTH) - math.lgamma(WIDTH / 2.0))
    reps = []
    for rep in range(REPS):
        rng = np.random.default_rng((seed ^ STEIN_STREAM ^ (rep * 0x9E3779B9)) % (1 << 63))
        rows = _haar_rows(rng, WIDTH, BLOCKS)
        g, k = _raw_haar_stein(weights, rows)
        raw = radial * np.mean(g, axis=0)
        stein = radial * _stein_fold_cv(g, k)
        current = _current_route_estimate(mlp, seed, rep)
        estimates = {"current": current, "raw_haar": raw, "stein": stein}
        mse = {name: float(np.mean((value - truth) ** 2)) for name, value in estimates.items()}
        reps.append({"rep": rep, "mse": mse, "estimates": {name: value.tolist() for name, value in estimates.items()}})
    return {"ok": True, "script_version": SCRIPT_VERSION, "mlp_index": shard_index, "seed": seed, "checksum_ok": True, "weights_sha256": actual, "config": {"width": WIDTH, "depth": DEPTH, "blocks": BLOCKS, "reps": REPS, "folds": "4/4", "ridge": "1e-3*mean(train_var(K))", "radial_factor": radial}, "truth_final": truth.tolist(), "reps": reps}


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
