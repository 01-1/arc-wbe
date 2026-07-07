#!/usr/bin/env python3
"""Fly payload for the block predictability gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flopscope.numpy as fnp
from local_engine import build_mlp
from estimator import (
    _DEEP_VARIANCE_MATCH_STRENGTH,
    _MIN_VARIANCE,
    _gaussian_relu_variance,
    _hadamard,
    _strassen_matmul,
    _zero_mean_relu_mean_cov,
)


def _block_rows(width: int, rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows = []
    for _ in range(n_blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        half = base * flips[None, :]
        rows.append(half)
        rows.append(-half)
    return np.concatenate(rows, axis=0)


def _moments(block: np.ndarray) -> tuple[float, float, float]:
    centered = block - block.mean(axis=0, keepdims=True)
    var = np.maximum((centered * centered).mean(axis=0), 1e-30)
    z = centered / np.sqrt(var)[None, :]
    skew = np.mean(np.abs((z * z * z).mean(axis=0)))
    kurt = np.mean((z * z * z * z).mean(axis=0) - 3.0)
    radius = np.mean(np.sum(block * block, axis=1))
    return float(radius), float(skew), float(kurt)


def run_one(shard_index: int, bank_path: Path, reps: int, blocks: int) -> dict[str, object]:
    arrays = np.load(bank_path)
    seeds = arrays["seeds"]
    truths = arrays["truths"]
    seed = int(seeds[shard_index])
    truth_final = np.asarray(truths[shard_index, -1], dtype=np.float64)
    mlp = build_mlp(256, 32, seed)
    weights_f32 = [w.astype(fnp.float32) for w in mlp.weights]
    w0 = mlp.weights[0]
    w0_f32 = weights_f32[0]
    target_mean, target_cov = _zero_mean_relu_mean_cov(w0.T @ w0)
    target_cov_np = np.asarray(target_cov, dtype=np.float64)
    target_mean_np = np.asarray(target_mean, dtype=np.float64)
    target_diag = np.diag(target_cov_np)
    downstream0 = np.asarray(np.sum(mlp.weights[1] * mlp.weights[1], axis=1), dtype=np.float64)
    downstream0 = downstream0 / max(float(np.mean(downstream0)), 1e-30)
    rows_per_block = 512
    result_blocks = []
    rep_mse = []

    for rep in range(reps):
        rng = np.random.default_rng(seed * 1_000_003 + 97 * rep + 17)
        signs = fnp.array(_block_rows(mlp.width, rng, blocks), dtype=fnp.float32)
        pre_half = _strassen_matmul(signs[: blocks * mlp.width], w0_f32, 3)
        y_pos = np.asarray(fnp.maximum(pre_half, 0.0), dtype=np.float32).reshape(blocks, mlp.width, mlp.width)
        y_neg = np.asarray(fnp.maximum(-pre_half, 0.0), dtype=np.float32).reshape(blocks, mlp.width, mlp.width)
        y = fnp.array(np.concatenate((y_pos, y_neg), axis=1).reshape(blocks * rows_per_block, mlp.width))
        y_np = np.asarray(y, dtype=np.float64)
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
        x0_np = np.asarray(x, dtype=np.float64)

        for layer_idx, w_prop in enumerate(weights_f32[1:], start=1):
            pre = _strassen_matmul(x, w_prop, 3)
            x = fnp.maximum(pre, 0.0)
            if layer_idx == 1:
                pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
                pre_centered = pre - pre_mean[None, :]
                target_var = _gaussian_relu_variance(
                    pre_mean, fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64)
                )
                sample_mean_1 = fnp.mean(x, axis=0).astype(fnp.float64)
                centered_layer = x - sample_mean_1[None, :]
                sample_var = fnp.maximum(
                    fnp.mean(centered_layer * centered_layer, axis=0).astype(fnp.float64),
                    _MIN_VARIANCE,
                )
                scale = 1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (fnp.sqrt(target_var / sample_var) - 1.0)
                centered_apply = x - sample_mean_1.astype(fnp.float32)[None, :]
                x = centered_apply * scale.astype(fnp.float32)[None, :] + sample_mean_1.astype(fnp.float32)[None, :]
                vm_scale = np.asarray(scale, dtype=np.float64)
        final_np = np.asarray(x, dtype=np.float64)
        block_means = final_np.reshape(blocks, rows_per_block, mlp.width).mean(axis=1)
        equal_mean = block_means.mean(axis=0)
        rep_mse.append(float(np.mean((equal_mean - truth_final) ** 2)))

        y_blocks = y_np.reshape(blocks, rows_per_block, mlp.width)
        x0_blocks = x0_np.reshape(blocks, rows_per_block, mlp.width)
        final_blocks = final_np.reshape(blocks, rows_per_block, mlp.width)
        for block_idx in range(blocks):
            yb = y_blocks[block_idx]
            xb = x0_blocks[block_idx]
            fb = final_blocks[block_idx]
            y_center = yb - yb.mean(axis=0, keepdims=True)
            x_center = xb - xb.mean(axis=0, keepdims=True)
            y_var = np.mean(y_center * y_center, axis=0)
            x_var = np.mean(x_center * x_center, axis=0)
            f_radius, f_skew, f_kurt = _moments(fb)
            feature = [
                float(np.mean((yb.mean(axis=0) - target_mean_np) ** 2)),
                float(abs(np.sum(y_var) - np.trace(np.asarray(sample_cov))) / mlp.width),
                float(np.mean((x_var - target_diag) ** 2)),
                float(np.mean((xb.mean(axis=0) - target_mean_np) ** 2)),
                float(abs(np.sum(x_var) - np.sum(target_diag)) / mlp.width),
                float(np.mean((vm_scale - 1.0) ** 2)),
                f_radius,
                float(np.mean(np.sum((fb - fb.mean(axis=0)) ** 2 * downstream0[None, :], axis=1))),
                f_skew,
                f_kurt,
            ]
            err = block_means[block_idx] - truth_final
            result_blocks.append(
                {
                    "rep": rep,
                    "block": block_idx,
                    "features": feature,
                    "sqerr": float(np.mean(err * err)),
                    "dot_equal_error": float(np.mean(err * (equal_mean - truth_final))),
                }
            )

    return {
        "ok": True,
        "seed": seed,
        "mlp_index": shard_index,
        "reps": reps,
        "blocks": blocks,
        "feature_names": [
            "raw_mean_resid",
            "raw_trace_resid",
            "recolor_diag_resid",
            "recolor_mean_resid",
            "recolor_trace_resid",
            "varmatch_energy",
            "final_radius",
            "downstream_final_radius",
            "final_abs_skew",
            "final_excess_kurt",
        ],
        "rep_mse": rep_mse,
        "block_rows": result_blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    payload = run_one(args.shard_index, args.bank, args.reps, args.blocks)
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
