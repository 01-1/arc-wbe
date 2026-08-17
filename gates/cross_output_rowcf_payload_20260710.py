#!/usr/bin/env python3
"""Machine-side row-cross-fitted James–Stein research gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flopscope as flops
import flopscope.numpy as fnp
from estimator import (
    _DEEP_VARIANCE_MATCH_STRENGTH,
    _MIN_VARIANCE,
    _gaussian_relu_mean,
    _gaussian_relu_variance,
    _hadamard,
    _strassen_matmul,
    _zero_mean_relu_mean_cov,
)
from local_engine import build_mlp


WIDTH = 256
DEPTH = 32
BLOCKS = 16
ROWS = 2 * WIDTH * BLOCKS
FOLD_ROWS = ROWS // 2
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "cross-output-rowcf-v1"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _independent_hadamard_positive(seed: int) -> np.ndarray:
    """Sixteen independent full positive Hadamard bases, never interleaved."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hadamard(WIDTH), dtype=np.float32)
    blocks = []
    for _ in range(BLOCKS):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=WIDTH)
        blocks.append(base * flips[None, :])
    return np.concatenate(blocks, axis=0)


def _current_route_final_z(mlp, seed: int) -> tuple[np.ndarray, np.ndarray]:
    weights = [fnp.array(w, dtype=fnp.float32) for w in mlp.weights]
    x_half = fnp.array(_independent_hadamard_positive(seed), dtype=fnp.float32)
    pre_half = _strassen_matmul(x_half, weights[0], 3)
    y = fnp.concatenate((fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0)

    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3)
        / float(ROWS)
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(WIDTH)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]

    final_pre = None
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        final_pre = pre
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
    assert final_pre is not None
    z = np.asarray(final_pre, dtype=np.float64)
    return z, np.mean(np.maximum(z, 0.0), axis=0)


def _row_folds(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.arange(FOLD_ROWS)
    negative = positive + FOLD_ROWS
    block_rows = WIDTH
    a_positive = positive[: 8 * block_rows]
    b_positive = positive[8 * block_rows :]
    a_negative = negative[: 8 * block_rows]
    b_negative = negative[8 * block_rows :]
    return np.concatenate((a_positive, a_negative)), np.concatenate((b_positive, b_negative))


def _cross_output_predictor(z: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    z = np.asarray(z, dtype=np.float64)
    y = np.mean(np.maximum(z, 0.0), axis=0)
    a = np.mean(z, axis=0)
    variance = np.maximum(np.mean((z - a[None, :]) ** 2, axis=0), _MIN_VARIANCE)
    std = np.sqrt(variance)
    alpha = np.clip(a / std, -4.0, 4.0)
    standardized = (z - a[None, :]) / std[None, :]
    skew = np.mean(standardized**3, axis=0)
    excess = np.mean(standardized**4, axis=0) - 3.0
    g = np.asarray(
        _gaussian_relu_mean(fnp.array(a), fnp.array(variance)), dtype=np.float64
    )
    features = np.column_stack(
        (np.ones(WIDTH), alpha, alpha**2, alpha**3, skew, skew * alpha, excess, excess * alpha)
    )
    target = (y - g) / std
    prediction = np.empty(WIDTH, dtype=np.float64)
    ridge_lambdas = []
    discrepancies = []
    for fold in range(4):
        held = (np.arange(WIDTH) % 4) == fold
        train = ~held
        train_features = features[train, 1:]
        held_features = features[held, 1:]
        means = np.mean(train_features, axis=0)
        scales = np.maximum(np.std(train_features, axis=0), np.sqrt(_MIN_VARIANCE))
        train_standardized = (train_features - means[None, :]) / scales[None, :]
        held_standardized = (held_features - means[None, :]) / scales[None, :]
        centered_x = train_standardized - np.mean(train_standardized, axis=0)[None, :]
        centered_t = target[train] - np.mean(target[train])
        gram = centered_x.T @ centered_x
        ridge_lambda = 0.1 * float(np.trace(gram)) / 8.0
        beta = np.linalg.solve(gram + ridge_lambda * np.eye(7), centered_x.T @ centered_t)
        prediction[held] = np.mean(target[train]) + (
            held_standardized - np.mean(train_standardized, axis=0)[None, :]
        ) @ beta
        ridge_lambdas.append(ridge_lambda)
        discrepancies.append(float(np.mean((prediction[held] - target[held]) ** 2)))
    p = g + std * prediction
    return p, {
        "prediction_mse_vs_fold_mean": float(np.mean((p - y) ** 2)),
        "ridge_lambda_mean": float(np.mean(ridge_lambdas)),
        "ridge_prediction_discrepancy_mean": float(np.mean(discrepancies)),
        "feature_alpha_mean": float(np.mean(alpha)),
    }


def _candidate(z: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    fold_a, fold_b = _row_folds(z)
    za, zb = z[fold_a], z[fold_b]
    y_a = np.mean(np.maximum(za, 0.0), axis=0)
    y_b = np.mean(np.maximum(zb, 0.0), axis=0)
    p_a, diag_a = _cross_output_predictor(za)
    p_b, diag_b = _cross_output_predictor(zb)
    noise_a = np.var(np.maximum(za, 0.0), axis=0, ddof=1) / float(FOLD_ROWS)
    noise_b = np.var(np.maximum(zb, 0.0), axis=0, ddof=1) / float(FOLD_ROWS)
    lambda_a = float(np.clip((1.0 - 2.0 / WIDTH) * np.mean(noise_a) / max(np.mean((y_a - p_b) ** 2), _MIN_VARIANCE), 0.0, 1.0))
    lambda_b = float(np.clip((1.0 - 2.0 / WIDTH) * np.mean(noise_b) / max(np.mean((y_b - p_a) ** 2), _MIN_VARIANCE), 0.0, 1.0))
    candidate = 0.5 * (
        (1.0 - lambda_a) * y_a + lambda_a * p_b
        + (1.0 - lambda_b) * y_b + lambda_b * p_a
    )
    return candidate, {
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "lambda_mean": 0.5 * (lambda_a + lambda_b),
        "predictor_discrepancy": float(np.mean((p_a - p_b) ** 2)),
        "target_discrepancy_a": float(np.mean((y_a - p_b) ** 2)),
        "target_discrepancy_b": float(np.mean((y_b - p_a) ** 2)),
        "predictor_a_mse": diag_a["prediction_mse_vs_fold_mean"],
        "predictor_b_mse": diag_b["prediction_mse_vs_fold_mean"],
    }


def run_one(shard_index: int, bank_path: Path, reps: int) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")

    fixed = []
    with flops.BudgetContext(flop_budget=2_000_000_000_000):
        for rep in range(reps):
            z, direct = _current_route_final_z(mlp, seed ^ (rep * 0x9E3779B9))
            candidate, diagnostics = _candidate(z)
            fixed.append({"rep": rep, "direct": direct, "candidate": candidate, "diagnostics": diagnostics})

    # Truth is read only after current/candidate vectors are fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    candidates = []
    rep_results = []
    for record in fixed:
        direct = np.asarray(record["direct"], dtype=np.float64)
        candidate = np.asarray(record["candidate"], dtype=np.float64)
        candidates.append(candidate)
        rep_results.append({
            "rep": record["rep"],
            "current_mse": float(np.mean((direct - truth) ** 2)),
            "candidate_mse": float(np.mean((candidate - truth) ** 2)),
            **record["diagnostics"],
        })
    mean_candidate = np.mean(np.stack(candidates, axis=0), axis=0)
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
            "rows": ROWS,
            "fold_rows": FOLD_ROWS,
            "reps": reps,
            "row_folds": "positive blocks 0:8 + matching negatives; 8:16 + matching negatives",
            "feature_folds": "j mod 4",
            "ridge_lambda": "0.1*trace(Xt.T@Xt)/8",
            "lambda_factor": "(1-2/256)*mean(noise)/mean(opposite_target_error)",
        },
        "rep_results": rep_results,
        "three_rep_squared_bias_proxy": float(np.mean((mean_candidate - truth) ** 2)) if reps == 3 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--reps", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank, args.reps)
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
