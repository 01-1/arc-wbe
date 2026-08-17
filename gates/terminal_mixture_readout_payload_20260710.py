#!/usr/bin/env python3
"""Truth-bank payload for the preregistered terminal mixture readout gate."""

from __future__ import annotations

import argparse
import hashlib
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


SCRIPT_VERSION = "terminal-mixture-readout-v1"
WIDTH = 256
DEPTH = 32
BLOCKS = 16
REPS = 3
EM_UPDATES = 12
VARIANCE_FLOOR_MULTIPLIER = 1e-4
WEIGHT_FLOOR = 0.02


def weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _positive_block_rows(width: int, rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    """Return exactly one positive Hadamard half per independent block."""
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows = []
    for _ in range(n_blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        rows.append(base * flips[None, :])
    positive = np.concatenate(rows, axis=0)
    assert positive.shape == (n_blocks * width, width)
    return positive


def _propagate_current_route(mlp, seed: int, rep: int, blocks: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed * 1_000_003 + 10_007 * rep + 41)
    weights_f32 = [weight.astype(fnp.float32) for weight in mlp.weights]
    positive_rows = fnp.array(_positive_block_rows(mlp.width, rng, blocks), dtype=fnp.float32)
    pre_half = _strassen_matmul(positive_rows, weights_f32[0], 3)
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

    final_pre = None
    for layer_idx, weight in enumerate(weights_f32[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        final_pre = pre
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(
                pre_mean,
                fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
            )
            successor_mean = fnp.mean(x, axis=0).astype(fnp.float64)
            centered_layer = x - successor_mean[None, :]
            sample_var = fnp.maximum(
                fnp.mean(centered_layer * centered_layer, axis=0).astype(fnp.float64),
                _MIN_VARIANCE,
            )
            scale = 1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (
                fnp.sqrt(target_var / sample_var) - 1.0
            )
            centered_apply = x - successor_mean.astype(fnp.float32)[None, :]
            x = (
                centered_apply * scale.astype(fnp.float32)[None, :]
                + successor_mean.astype(fnp.float32)[None, :]
            )
    assert final_pre is not None
    return np.asarray(final_pre, dtype=np.float64), np.asarray(x, dtype=np.float64)


def _normal_relu_mean(mean: float, variance: float) -> float:
    variance = max(float(variance), 1e-30)
    sigma = math.sqrt(variance)
    alpha = mean / sigma
    phi = math.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))
    return sigma * phi + mean * Phi


def _logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(np.sum(np.exp(values - maximum[:, None]), axis=1))


def _fit_two_gaussian_readout(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen per-output 2-Gaussian EM readout and a Gaussian control."""
    n_rows, width = z.shape
    mixture = np.empty(width, dtype=np.float64)
    gaussian = np.empty(width, dtype=np.float64)
    for output in range(width):
        values = np.asarray(z[:, output], dtype=np.float64)
        total_mean = float(np.mean(values))
        total_variance = float(np.mean((values - total_mean) ** 2))
        variance_floor = VARIANCE_FLOOR_MULTIPLIER * total_variance + 1e-30
        gaussian[output] = _normal_relu_mean(total_mean, total_variance)

        ordered = np.sort(values, kind="mergesort")
        split = n_rows // 2
        lower = ordered[:split]
        upper = ordered[split:]
        means = np.array([np.mean(lower), np.mean(upper)], dtype=np.float64)
        variances = np.array(
            [np.mean((lower - means[0]) ** 2), np.mean((upper - means[1]) ** 2)],
            dtype=np.float64,
        )
        variances = np.maximum(variances, variance_floor)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        for _ in range(EM_UPDATES):
            log_prob = np.empty((n_rows, 2), dtype=np.float64)
            for component in range(2):
                log_prob[:, component] = (
                    math.log(weights[component])
                    - 0.5 * math.log(2.0 * math.pi * variances[component])
                    - 0.5 * (values - means[component]) ** 2 / variances[component]
                )
            responsibilities = np.exp(log_prob - _logsumexp(log_prob)[:, None])
            effective = np.sum(responsibilities, axis=0)
            weights = np.maximum(effective / float(n_rows), WEIGHT_FLOOR)
            weights /= np.sum(weights)
            means = np.sum(responsibilities * values[:, None], axis=0) / np.maximum(effective, 1e-30)
            variances = np.sum(
                responsibilities * (values[:, None] - means[None, :]) ** 2,
                axis=0,
            ) / np.maximum(effective, 1e-30)
            variances = np.maximum(variances, variance_floor)
        sigma = np.sqrt(variances)
        alpha = means / sigma
        phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + np.array([math.erf(float(value) / math.sqrt(2.0)) for value in alpha]))
        mixture[output] = float(np.sum(weights * (sigma * phi + means * Phi)))
    return mixture, gaussian


def _pair_variance(estimates: np.ndarray) -> float:
    pairs = []
    for first in range(estimates.shape[0]):
        for second in range(first + 1, estimates.shape[0]):
            pairs.append(np.mean((estimates[first] - estimates[second]) ** 2) / 2.0)
    return float(np.mean(pairs))


def run_one(shard_index: int, bank_path: Path, reps: int, blocks: int) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    checksum = weights_sha256(mlp.weights)
    expected_checksum = str(bank["weights_sha256"][shard_index])
    if checksum != expected_checksum:
        raise ValueError(f"weight checksum mismatch for bank index {shard_index}")

    replicate_records = []
    for rep in range(reps):
        final_pre, final_relu = _propagate_current_route(mlp, seed, rep, blocks)
        baseline = np.mean(final_relu, axis=0)
        mixture, gaussian = _fit_two_gaussian_readout(final_pre)
        replicate_records.append(
            {
                "rep": rep,
                "baseline_estimate": baseline.tolist(),
                "mixture_estimate": mixture.tolist(),
                "gaussian_estimate": gaussian.tolist(),
            }
        )

    truth_final = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    baseline_estimates = np.asarray([record["baseline_estimate"] for record in replicate_records], dtype=np.float64)
    mixture_estimates = np.asarray([record["mixture_estimate"] for record in replicate_records], dtype=np.float64)
    gaussian_estimates = np.asarray([record["gaussian_estimate"] for record in replicate_records], dtype=np.float64)
    baseline_mses = np.mean((baseline_estimates - truth_final[None, :]) ** 2, axis=1)
    mixture_mses = np.mean((mixture_estimates - truth_final[None, :]) ** 2, axis=1)
    gaussian_mses = np.mean((gaussian_estimates - truth_final[None, :]) ** 2, axis=1)
    mixture_mean = np.mean(mixture_estimates, axis=0)
    mixture_mean_mse = float(np.mean((mixture_mean - truth_final) ** 2))
    mixture_pair_variance = _pair_variance(mixture_estimates)
    mixture_bias_proxy = max(mixture_mean_mse - mixture_pair_variance / float(reps), 0.0)
    baseline_pair_variance = _pair_variance(baseline_estimates)
    gaussian_pair_variance = _pair_variance(gaussian_estimates)
    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": True,
        "weights_sha256": checksum,
        "reps": reps,
        "blocks": blocks,
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "em_updates": EM_UPDATES,
            "variance_floor_multiplier": VARIANCE_FLOOR_MULTIPLIER,
            "weight_floor": WEIGHT_FLOOR,
        },
        "replicates": replicate_records,
        "baseline_mses": baseline_mses.tolist(),
        "mixture_mses": mixture_mses.tolist(),
        "gaussian_mses": gaussian_mses.tolist(),
        "baseline_mean_mse": float(np.mean(baseline_mses)),
        "mixture_mean_mse": float(np.mean(mixture_mses)),
        "gaussian_mean_mse": float(np.mean(gaussian_mses)),
        "baseline_three_rep_mean_mse": float(np.mean((np.mean(baseline_estimates, axis=0) - truth_final) ** 2)),
        "mixture_three_rep_mean_mse": mixture_mean_mse,
        "gaussian_three_rep_mean_mse": float(np.mean((np.mean(gaussian_estimates, axis=0) - truth_final) ** 2)),
        "baseline_pair_variance": baseline_pair_variance,
        "mixture_pair_variance": mixture_pair_variance,
        "gaussian_pair_variance": gaussian_pair_variance,
        "mixture_squared_bias_proxy": mixture_bias_proxy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    if args.reps != REPS or args.blocks != BLOCKS:
        raise ValueError("the preregistered gate requires exactly 3 reps and 16 blocks")
    payload = run_one(args.shard_index, args.bank, args.reps, args.blocks)
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
