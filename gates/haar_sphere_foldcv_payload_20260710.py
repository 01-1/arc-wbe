#!/usr/bin/env python3
"""Truth-bank Fly payload for the preregistered Haar-sphere fold-CV gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Fly executes this file from the repository root while preserving its
# nested archive path; make root-level estimator/local_engine imports explicit.
ROOT = Path(__file__).resolve().parents[2]
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
BLOCKS = 16
REPS = 3
STREAM = 0xA11CE5EED
SCRIPT_VERSION = "haar-sphere-foldcv-v1"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _haar_rows(rng: np.random.Generator, width: int, blocks: int) -> np.ndarray:
    """Return `blocks * width` rows from independent scaled Haar bases."""
    rows: list[np.ndarray] = []
    for _ in range(blocks):
        gaussian = rng.standard_normal((width, width))
        q, r = np.linalg.qr(gaussian)
        signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
        q = q * signs[None, :]
        rows.append((math.sqrt(width) * q.T).astype(np.float32))
    return np.concatenate(rows, axis=0)


def _raw_pair_features(
    weights: list[fnp.ndarray], rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the exact antithetic orbit through the raw fp32 network."""
    x = fnp.array(rows, dtype=fnp.float32)
    pre_half = _strassen_matmul(x, weights[0], 3)
    c = 0.5 * fnp.abs(pre_half.astype(fnp.float64))
    positive = fnp.maximum(pre_half, 0.0)
    negative = fnp.maximum(-pre_half, 0.0)
    for weight in weights[1:]:
        positive = fnp.maximum(_strassen_matmul(positive, weight, 3), 0.0)
        negative = fnp.maximum(_strassen_matmul(negative, weight, 3), 0.0)
    g = 0.5 * (positive.astype(fnp.float64) + negative.astype(fnp.float64))
    return np.asarray(c, dtype=np.float64), np.asarray(g, dtype=np.float64)


def _fixed_sphere_mu_c(w0: np.ndarray) -> np.ndarray:
    d = w0.shape[0]
    log_coeff = (
        0.5 * math.log(d)
        + math.lgamma(d / 2.0)
        - math.log(2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((d + 1.0) / 2.0)
    )
    return math.exp(log_coeff) * np.linalg.norm(w0, axis=0)


def _radial_factor(width: int) -> float:
    return math.exp(
        0.5 * math.log(2.0)
        + math.lgamma((width + 1.0) / 2.0)
        - 0.5 * math.log(width)
        - math.lgamma(width / 2.0)
    )


def _foldcv(c: np.ndarray, g: np.ndarray, mu_c: np.ndarray) -> np.ndarray:
    rows_per_block = c.shape[0] // BLOCKS
    block_ids = np.arange(BLOCKS).repeat(rows_per_block)
    estimates: list[np.ndarray] = []
    for held_blocks in (np.arange(0, 8), np.arange(8, 16)):
        held = np.isin(block_ids, held_blocks)
        train = ~held
        c_train = c[train]
        g_train = g[train]
        c_test = c[held]
        g_test = g[held]
        c_centered = c_train - np.mean(c_train, axis=0, keepdims=True)
        g_centered = g_train - np.mean(g_train, axis=0, keepdims=True)
        gram = c_centered.T @ c_centered
        lam = 0.1 * float(np.trace(gram)) / float(c.shape[1])
        beta = np.linalg.solve(
            gram + lam * np.eye(c.shape[1], dtype=np.float64),
            c_centered.T @ g_centered,
        )
        estimates.append(np.mean(g_test, axis=0) + (mu_c - np.mean(c_test, axis=0)) @ beta)
    return 0.5 * (estimates[0] + estimates[1])


def _hadamard_rows(width: int, rng: np.random.Generator, blocks: int) -> np.ndarray:
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows: list[np.ndarray] = []
    for _ in range(blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        rows.append(base * flips[None, :])
    return np.concatenate(rows, axis=0)


def _current_route_estimate(mlp, seed: int, rep: int) -> np.ndarray:
    """Existing research helper for `hadamard_st3_b16`, kept fixed."""
    rng = np.random.default_rng(seed * 1_000_003 + 10_007 * rep + 41)
    weights_f32 = [w.astype(fnp.float32) for w in mlp.weights]
    rows = fnp.array(_hadamard_rows(mlp.width, rng, BLOCKS), dtype=fnp.float32)
    pre_half = _strassen_matmul(rows, weights_f32[0], 3)
    y = fnp.concatenate((fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0)

    target_mean, target_cov = _zero_mean_relu_mean_cov(mlp.weights[0].T @ mlp.weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.T.astype(fnp.float32), centered.astype(fnp.float32), 3)
        / float(centered.shape[0])
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(mlp.width)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]

    for layer_idx, weight in enumerate(weights_f32[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(
                pre_mean,
                fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
            )
            sample_mean_1 = fnp.mean(x, axis=0).astype(fnp.float64)
            centered_layer = x - sample_mean_1[None, :]
            sample_var = fnp.maximum(
                fnp.mean(centered_layer * centered_layer, axis=0).astype(fnp.float64),
                _MIN_VARIANCE,
            )
            scale = 1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (
                fnp.sqrt(target_var / sample_var) - 1.0
            )
            x = (
                (x - sample_mean_1.astype(fnp.float32)[None, :])
                * scale.astype(fnp.float32)[None, :]
                + sample_mean_1.astype(fnp.float32)[None, :]
            )
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def run_one(shard_index: int, bank_path: Path) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights = [w.astype(fnp.float32) for w in mlp.weights]
    actual = _weights_sha256([np.asarray(w) for w in weights])
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")
    w0 = np.asarray(weights[0], dtype=np.float64)
    mu_c = _fixed_sphere_mu_c(w0)
    radial = _radial_factor(WIDTH)
    reps: list[dict[str, object]] = []
    for rep in range(REPS):
        rng = np.random.default_rng((seed ^ STREAM ^ (rep * 0x9E3779B9)) % (1 << 63))
        rows = _haar_rows(rng, WIDTH, BLOCKS)
        c, g = _raw_pair_features(weights, rows)
        raw = radial * np.mean(g, axis=0)
        cv = radial * _foldcv(c, g, mu_c)
        current = _current_route_estimate(mlp, seed, rep)
        estimates = {"current": current, "raw_haar": raw, "haar_cv": cv}
        mse = {name: float(np.mean((value - truth) ** 2)) for name, value in estimates.items()}
        reps.append(
            {
                "rep": rep,
                "mse": mse,
                "estimates": {name: value.tolist() for name, value in estimates.items()},
            }
        )
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
            "blocks": BLOCKS,
            "reps": REPS,
            "ridge_factor": 0.1,
            "radial_factor": radial,
        },
        "truth_final": truth.tolist(),
        "reps": reps,
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
