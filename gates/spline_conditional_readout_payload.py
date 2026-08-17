#!/usr/bin/env python3
"""Fly payload for cross-fitted smooth latent conditional readout gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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


LAYERS = (24, 28, 30)
RANKS = (1, 2, 4)
FAMILIES = ("poly2", "poly3", "bins8")


def _block_rows(width: int, rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows = []
    for _ in range(n_blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        half = base * flips[None, :]
        rows.append(half)
        rows.append(-half)
    return np.concatenate(rows, axis=0)


def _basis(z: np.ndarray, family: str, knots: list[np.ndarray] | None = None) -> np.ndarray:
    if family == "poly2":
        cols = [np.ones((z.shape[0], 1)), z, z * z]
        if z.shape[1] >= 2:
            cols.append(np.column_stack([z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i + 1, z.shape[1])]))
        return np.concatenate(cols, axis=1)
    if family == "poly3":
        return np.concatenate([np.ones((z.shape[0], 1)), z, z * z, z * z * z], axis=1)
    if family == "bins8":
        # Piecewise-constant spline surrogate with fixed quantile knots from the train fold.
        cols = [np.ones((z.shape[0], 1))]
        for j in range(z.shape[1]):
            q = knots[j] if knots is not None else np.quantile(z[:, j], np.linspace(0.125, 0.875, 7))
            cols.append((z[:, j : j + 1] > q[None, :]).astype(np.float64))
        return np.concatenate(cols, axis=1)
    raise ValueError(f"unknown family {family}")


def _fit_predict(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray, family: str, ridge: float) -> np.ndarray:
    train_mu = train_z.mean(axis=0)
    train_sd = train_z.std(axis=0)
    train_sd[train_sd == 0.0] = 1.0
    z_train = (train_z - train_mu) / train_sd
    z_test = (test_z - train_mu) / train_sd
    knots = None
    if family == "bins8":
        knots = [np.quantile(z_train[:, j], np.linspace(0.125, 0.875, 7)) for j in range(z_train.shape[1])]
    x_train = _basis(z_train, family, knots)
    x_test = _basis(z_test, family, knots)
    penalty = np.eye(x_train.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    return x_test @ beta


def _crossfit_estimate(latent: np.ndarray, final: np.ndarray, rank: int, family: str, ridge: float) -> np.ndarray:
    centered = latent - latent.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    z = centered @ vt[:rank].T
    pred = np.empty_like(final, dtype=np.float64)
    folds = np.arange(final.shape[0]) % 4
    for fold in range(4):
        train = folds != fold
        test = ~train
        pred[test] = _fit_predict(z[train], final[train], z[test], family, ridge)
    return pred.mean(axis=0)


def _propagate_current_route(mlp, seed: int, rep: int, blocks: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed * 1_000_003 + 10_007 * rep + 41)
    weights_f32 = [w.astype(fnp.float32) for w in mlp.weights]
    rows = fnp.array(_block_rows(mlp.width, rng, blocks), dtype=fnp.float32)
    half = rows[: blocks * mlp.width]
    pre_half = _strassen_matmul(half, weights_f32[0], 3)
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

    saved: dict[int, np.ndarray] = {}
    for layer_idx, w_prop in enumerate(weights_f32[1:], start=1):
        pre = _strassen_matmul(x, w_prop, 3)
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
            sample_var = fnp.maximum(fnp.mean(centered_layer * centered_layer, axis=0).astype(fnp.float64), _MIN_VARIANCE)
            scale = 1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (fnp.sqrt(target_var / sample_var) - 1.0)
            centered_apply = x - sample_mean_1.astype(fnp.float32)[None, :]
            x = centered_apply * scale.astype(fnp.float32)[None, :] + sample_mean_1.astype(fnp.float32)[None, :]
        if layer_idx in LAYERS:
            saved[layer_idx] = np.asarray(x, dtype=np.float64)
    return saved, np.asarray(x, dtype=np.float64)


def run_one(shard_index: int, bank_path: Path, reps: int, blocks: int) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    truth_final = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    mlp = build_mlp(256, 32, seed)
    records = []
    for rep in range(reps):
        latents, final = _propagate_current_route(mlp, seed, rep, blocks)
        equal = final.mean(axis=0)
        equal_mse = float(np.mean((equal - truth_final) ** 2))
        for layer in LAYERS:
            for rank in RANKS:
                for family in FAMILIES:
                    for ridge in (1e-3, 1e-1):
                        estimate = _crossfit_estimate(latents[layer], final, rank, family, ridge)
                        mse = float(np.mean((estimate - truth_final) ** 2))
                        records.append(
                            {
                                "rep": rep,
                                "layer": layer,
                                "rank": rank,
                                "family": family,
                                "ridge": ridge,
                                "equal_mse": equal_mse,
                                "conditional_mse": mse,
                                "ratio": equal_mse / mse if mse > 0.0 else 0.0,
                                "bias_norm": float(np.mean((estimate - equal) ** 2)),
                            }
                        )
    return {
        "ok": True,
        "script": "spline_conditional_readout_payload_v1",
        "mlp_index": shard_index,
        "seed": seed,
        "reps": reps,
        "blocks": blocks,
        "layers": list(LAYERS),
        "ranks": list(RANKS),
        "families": list(FAMILIES),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    payload = run_one(args.shard_index, args.bank, args.reps, args.blocks)
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
