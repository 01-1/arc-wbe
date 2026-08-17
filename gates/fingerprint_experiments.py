#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

WIDTH = 256
DEPTH = 32
BLOCKS = 16
START_SEED = 1000
MIN_VARIANCE = 1e-30
ROOT = Path(__file__).resolve().parent


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def build_mlp(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32) for _ in range(DEPTH)]


def hadamard(width: int = WIDTH) -> np.ndarray:
    rows = [[1.0]]
    while len(rows) < width:
        rows = [row + row for row in rows] + [row + [-value for value in row] for row in rows]
    return np.array(rows, dtype=np.float64)


HADAMARD = hadamard()


def zero_mean_relu_mean_cov(cov_pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    var = np.maximum(np.diag(cov_pre), MIN_VARIANCE)
    std = np.sqrt(var)
    denom = std[:, None] * std[None, :]
    rho = np.clip(cov_pre / denom, -1.0, 1.0)
    second = denom * (
        np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) + (math.pi - np.arccos(rho)) * rho
    ) / (2.0 * math.pi)
    mean = std / math.sqrt(2.0 * math.pi)
    return mean, second - np.outer(mean, mean)


def gaussian_relu_mean_var(mean_pre: np.ndarray, var_pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    var_pre = np.maximum(var_pre, MIN_VARIANCE)
    sigma = np.sqrt(var_pre)
    alpha = mean_pre / sigma
    phi = normal_pdf(alpha)
    Phi = normal_cdf(alpha)
    relu_mean = sigma * phi + mean_pre * Phi
    second = (mean_pre * mean_pre + var_pre) * Phi + mean_pre * sigma * phi
    return relu_mean, np.maximum(second - relu_mean * relu_mean, MIN_VARIANCE)


def smooth_relu_mean(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    alpha = x / sigma
    return sigma * normal_pdf(alpha) + x * normal_cdf(alpha)


def truth_final(weights: list[np.ndarray], n_total: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    batch_pairs = 10_000
    final_sum = np.zeros(WIDTH, dtype=np.float64)
    final_sum_sq = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < n_total:
        pairs = min(batch_pairs, (n_total - done) // 2)
        x0 = rng.standard_normal((pairs, WIDTH)).astype(np.float32)
        x = np.concatenate((x0, -x0), axis=0).astype(np.float64)
        for w in weights:
            x = np.maximum(x @ w.astype(np.float64), 0.0)
        final_sum += x.sum(axis=0)
        final_sum_sq += (x * x).sum(axis=0)
        done += 2 * pairs
    mean = final_sum / n_total
    var = final_sum_sq / n_total - mean * mean
    return mean, float(np.mean(var) / n_total)


def randomized_hadamard_half_blocks(rng: np.random.Generator) -> np.ndarray:
    blocks = []
    for _ in range(BLOCKS):
        flips = 2.0 * rng.integers(0, 2, size=WIDTH) - 1.0
        blocks.append(HADAMARD * flips[None, :])
    return np.concatenate(blocks, axis=0)


def first_recolor(weights: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    w0 = weights[0].astype(np.float64)
    x_half = randomized_hadamard_half_blocks(rng)
    pre = x_half @ w0
    y = np.concatenate((np.maximum(pre, 0.0), np.maximum(-pre, 0.0)), axis=0)
    cov_pre = w0.T @ w0
    target_mean, target_cov = zero_mean_relu_mean_cov(cov_pre)
    sample_mean = y.mean(axis=0)
    centered = y - sample_mean[None, :]
    sample_cov = (centered.T @ centered) / centered.shape[0]
    jitter = max(float(np.mean(np.diag(target_cov))), MIN_VARIANCE) * 1e-6
    eye = np.eye(WIDTH)
    sample_chol = np.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = np.linalg.cholesky(target_cov + jitter * eye)
    recolor = np.linalg.inv(sample_chol.T) @ target_chol.T
    return centered @ recolor + target_mean[None, :]


def run_variant(weights: list[np.ndarray], seed: int, variant: str, x0: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = first_recolor(weights, rng) if x0 is None else x0.copy()
    for layer_idx, w in enumerate(weights[1:], start=1):
        pre = x @ w.astype(np.float64)
        if variant.startswith("smooth_unbiased") and layer_idx >= 16:
            frac = float(variant.rsplit("_", 1)[1])
            sigma = frac * np.maximum(pre.std(axis=0), 1e-12)
            eps = rng.standard_normal(pre.shape)
            # Exactly unbiased conditional on the current preactivation ensemble.
            x = np.maximum(pre + eps * sigma[None, :], 0.0)
            x -= smooth_relu_mean(pre, sigma[None, :]) - np.maximum(pre, 0.0)
        elif variant.startswith("smooth_biased") and layer_idx >= 16:
            frac = float(variant.rsplit("_", 1)[1])
            sigma = frac * np.maximum(pre.std(axis=0), 1e-12)
            x = smooth_relu_mean(pre, sigma[None, :])
        else:
            x = np.maximum(pre, 0.0)

        if layer_idx == 1:
            pre_mean = pre.mean(axis=0)
            pre_centered = pre - pre_mean[None, :]
            _, target_var = gaussian_relu_mean_var(pre_mean, np.mean(pre_centered * pre_centered, axis=0))
            sample_mean2 = x.mean(axis=0)
            centered_layer = x - sample_mean2[None, :]
            sample_var = np.maximum(np.mean(centered_layer * centered_layer, axis=0), MIN_VARIANCE)
            scale = 1.0 + 1.5 * (np.sqrt(target_var / sample_var) - 1.0)
            x = centered_layer * scale[None, :] + sample_mean2[None, :]

        if variant.startswith("late_shrink"):
            _, _, start, alpha = variant.split("_")
            if layer_idx >= int(start):
                mean = x.mean(axis=0)
                x = mean[None, :] + float(alpha) * (x - mean[None, :])

        if variant.startswith("late_gauss"):
            _, _, start = variant.split("_")
            if layer_idx >= int(start):
                mean_pre = pre.mean(axis=0)
                var_pre = np.var(pre, axis=0)
                mean, _ = gaussian_relu_mean_var(mean_pre, var_pre)
                x = np.broadcast_to(mean[None, :], x.shape).copy()
    return x.mean(axis=0)


def main() -> None:
    mlp_seeds = [11]
    R = 12
    truth_samples = 120_000
    variants = [
        "baseline",
        "smooth_unbiased_0.05",
        "late_shrink_20_0.90",
        "late_shrink_24_0.85",
    ]
    started = time.perf_counter()
    rows = []
    for mlp_seed in mlp_seeds:
        weights = build_mlp(mlp_seed)
        truth, truth_noise = truth_final(weights, truth_samples, 800_000 + mlp_seed)
        finals = {v: [] for v in variants}
        for r in range(R):
            seed = START_SEED + r
            x0 = first_recolor(weights, np.random.default_rng(seed))
            for v in variants:
                finals[v].append(run_variant(weights, seed, v, x0))
        base_errors = np.stack(finals["baseline"]) - truth[None, :]
        base_var = float(np.mean(np.var(base_errors, axis=0, ddof=1)))
        base_mse = float(np.mean(base_errors * base_errors))
        for v in variants:
            errors = np.stack(finals[v]) - truth[None, :]
            var = float(np.mean(np.var(errors, axis=0, ddof=1)))
            mse = float(np.mean(errors * errors))
            bias = np.mean(errors, axis=0)
            rows.append({
                "mlp_seed": mlp_seed,
                "variant": v,
                "R": R,
                "truth_samples": truth_samples,
                "truth_noise_mse": truth_noise,
                "mse": mse,
                "seed_variance": var,
                "bias_mse": float(np.mean(bias * bias)),
                "variance_ratio_vs_baseline": var / base_var,
                "mse_ratio_vs_baseline": mse / base_mse,
            })
            print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for v in variants:
        selected = [r for r in rows if r["variant"] == v]
        summary[v] = {
            "mean_mse": float(np.mean([r["mse"] for r in selected])),
            "mean_seed_variance": float(np.mean([r["seed_variance"] for r in selected])),
            "mean_variance_ratio_vs_baseline": float(np.mean([r["variance_ratio_vs_baseline"] for r in selected])),
            "mean_mse_ratio_vs_baseline": float(np.mean([r["mse_ratio_vs_baseline"] for r in selected])),
        }
    out = {"rows": rows, "summary": summary, "elapsed_seconds": time.perf_counter() - started}
    (ROOT / "fingerprint_experiment_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"summary": summary, "elapsed_seconds": out["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
