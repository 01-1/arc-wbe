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
ROWS = 2 * BLOCKS * WIDTH
MLP_SEEDS = (11, 22)
ESTIMATOR_SEEDS = tuple(range(1000, 1060))
BRANCH_LAYERS = (16, 20, 24)
RANKS = (32, 64, 128)
PROJ_DIM = 128
MIN_VARIANCE = 1e-30
OUTDIR = Path(__file__).resolve().parent


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def hadamard(width: int = WIDTH) -> np.ndarray:
    rows = [[1.0]]
    while len(rows) < width:
        rows = [row + row for row in rows] + [row + [-value for value in row] for row in rows]
    return np.array(rows, dtype=np.float64)


HADAMARD = hadamard()


def build_mlp(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32) for _ in range(DEPTH)]


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


def randomized_hadamard_half_blocks(rng: np.random.Generator) -> np.ndarray:
    blocks = []
    for _ in range(BLOCKS):
        flips = 2.0 * rng.integers(0, 2, size=WIDTH) - 1.0
        blocks.append(HADAMARD * flips[None, :])
    return np.concatenate(blocks, axis=0)


def route_snapshots(weights: list[np.ndarray], seed: int) -> dict[int | str, np.ndarray]:
    rng = np.random.default_rng(seed)
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
    x = centered @ recolor + target_mean[None, :]

    snapshots: dict[int | str, np.ndarray] = {}
    if 1 in BRANCH_LAYERS:
        snapshots[1] = x.copy()
    for layer_idx, w in enumerate(weights[1:], start=2):
        pre = x @ w.astype(np.float64)
        x = np.maximum(pre, 0.0)
        if layer_idx == 2:
            pre_mean = pre.mean(axis=0)
            pre_centered = pre - pre_mean[None, :]
            _, target_var = gaussian_relu_mean_var(pre_mean, np.mean(pre_centered * pre_centered, axis=0))
            sample_mean2 = x.mean(axis=0)
            centered_layer = x - sample_mean2[None, :]
            sample_var = np.maximum(np.mean(centered_layer * centered_layer, axis=0), MIN_VARIANCE)
            scale = 1.0 + 1.5 * (np.sqrt(target_var / sample_var) - 1.0)
            x = centered_layer * scale[None, :] + sample_mean2[None, :]
        if layer_idx in BRANCH_LAYERS:
            snapshots[layer_idx] = x.copy()
    snapshots["final"] = x.mean(axis=0)
    return snapshots


def propagate_full(x: np.ndarray, weights: list[np.ndarray], branch_layer: int) -> np.ndarray:
    y = x.copy()
    for w in weights[branch_layer:]:
        y = np.maximum(y @ w.astype(np.float64), 0.0)
    return y.mean(axis=0)


def svd_suffixes(weights: list[np.ndarray]) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {r: [] for r in RANKS}
    for w in weights:
        u, s, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
        for r in RANKS:
            out[r].append((u[:, :r] * s[:r]) @ vt[:r, :])
    return out


def propagate_rank(x: np.ndarray, svd_weights: list[np.ndarray], branch_layer: int) -> np.ndarray:
    y = x.copy()
    for w in svd_weights[branch_layer:]:
        y = np.maximum(y @ w, 0.0)
    return y.mean(axis=0)


def propagate_diag_gaussian(x: np.ndarray, weights: list[np.ndarray], branch_layer: int) -> np.ndarray:
    mean = x.mean(axis=0)
    var = np.var(x, axis=0)
    for w in weights[branch_layer:]:
        wf = w.astype(np.float64)
        pre_mean = mean @ wf
        pre_var = var @ (wf * wf)
        mean, var = gaussian_relu_mean_var(pre_mean, pre_var)
    return mean


def propagate_subsample(x: np.ndarray, weights: list[np.ndarray], branch_layer: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=x.shape[0] // 4, replace=False)
    y = x[idx].copy()
    for w in weights[branch_layer:]:
        y = np.maximum(y @ w.astype(np.float64), 0.0)
    return y.mean(axis=0)


def propagate_projected(x: np.ndarray, weights: list[np.ndarray], branch_layer: int) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    p = vt[:PROJ_DIM].T
    y = x @ p
    suffix = weights[branch_layer:]
    for w in suffix[:-1]:
        wp = p.T @ w.astype(np.float64) @ p
        y = np.maximum(y @ wp, 0.0)
    y = np.maximum(y @ (p.T @ suffix[-1].astype(np.float64)), 0.0)
    return y.mean(axis=0)


def pooled_rho2(full: np.ndarray, cheap: np.ndarray) -> dict[str, float]:
    xf = full - full.mean(axis=0, keepdims=True)
    yc = cheap - cheap.mean(axis=0, keepdims=True)
    cov = np.mean(xf * yc, axis=0)
    vx = np.mean(xf * xf, axis=0)
    vy = np.mean(yc * yc, axis=0)
    rho2 = cov * cov / np.maximum(vx * vy, MIN_VARIANCE)
    weights = vx
    pooled = float(np.sum(weights * rho2) / np.sum(weights))
    scalar_cov = float(np.mean(np.sum(xf, axis=1) * np.sum(yc, axis=1)))
    scalar_vx = float(np.mean(np.sum(xf, axis=1) ** 2))
    scalar_vy = float(np.mean(np.sum(yc, axis=1) ** 2))
    return {
        "variance_weighted_pooled_rho2": pooled,
        "mean_coordinate_rho2": float(np.mean(rho2)),
        "scalar_sum_rho2": scalar_cov * scalar_cov / max(scalar_vx * scalar_vy, MIN_VARIANCE),
        "full_variance_weight_sum": float(np.sum(weights)),
    }


def cost_ratio(candidate: str, branch_layer: int) -> float:
    suffix_layers = DEPTH - branch_layer
    if candidate.startswith("rank"):
        r = int(candidate.split("_r")[1])
        return 2.0 * r / WIDTH
    if candidate == "diag_gaussian":
        return 2.0 / ROWS
    if candidate == "row_subsample_q25":
        return 0.25
    if candidate == "projected_128":
        if suffix_layers <= 1:
            return 0.5
        return ((suffix_layers - 1) * PROJ_DIM * PROJ_DIM + PROJ_DIM * WIDTH) / (
            suffix_layers * WIDTH * WIDTH
        )
    if candidate in {"full_control", "noise_degraded_full"}:
        return 1.0
    raise KeyError(candidate)


def main() -> None:
    started = time.perf_counter()
    candidate_names = [
        "full_control",
        "noise_degraded_full",
        *(f"rank_r{r}" for r in RANKS),
        "diag_gaussian",
        "row_subsample_q25",
        "projected_128",
    ]
    accum: dict[str, dict[str, list[np.ndarray]]] = {}
    rows = []
    for mlp_seed in MLP_SEEDS:
        weights = build_mlp(mlp_seed)
        svds = svd_suffixes(weights)
        for branch in BRANCH_LAYERS:
            key = f"mlp{mlp_seed}_k{branch}"
            accum[key] = {"full": []}
            for name in candidate_names:
                accum[key][name] = []
        for i, est_seed in enumerate(ESTIMATOR_SEEDS, start=1):
            snaps = route_snapshots(weights, est_seed)
            for branch in BRANCH_LAYERS:
                x = snaps[branch]
                full = propagate_full(x, weights, branch)
                key = f"mlp{mlp_seed}_k{branch}"
                accum[key]["full"].append(full)
                accum[key]["full_control"].append(full.copy())
                for r in RANKS:
                    accum[key][f"rank_r{r}"].append(propagate_rank(x, svds[r], branch))
                accum[key]["diag_gaussian"].append(propagate_diag_gaussian(x, weights, branch))
                accum[key]["row_subsample_q25"].append(
                    propagate_subsample(x, weights, branch, seed=9_000_000 + mlp_seed * 10_000 + branch * 100 + est_seed)
                )
                accum[key]["projected_128"].append(propagate_projected(x, weights, branch))
            print(f"mlp={mlp_seed} seed {i}/{len(ESTIMATOR_SEEDS)}", flush=True)

    for mlp_seed in MLP_SEEDS:
        for branch in BRANCH_LAYERS:
            key = f"mlp{mlp_seed}_k{branch}"
            full = np.stack(accum[key]["full"], axis=0)
            sigma = np.sqrt(np.var(full, axis=0, ddof=0))
            degraded = []
            for est_seed, row in zip(ESTIMATOR_SEEDS, full):
                rng = np.random.default_rng(7_000_000 + mlp_seed * 10_000 + branch * 100 + est_seed)
                degraded.append(row + rng.standard_normal(WIDTH) * sigma)
            accum[key]["noise_degraded_full"] = degraded

    per_mlp = {}
    pooled_by_branch = {}
    within_mlp_weighted_by_branch = {}
    for branch in BRANCH_LAYERS:
        pooled_by_branch[str(branch)] = {}
        within_mlp_weighted_by_branch[str(branch)] = {}
        for name in candidate_names:
            all_full = []
            all_cheap = []
            weighted = []
            weight_sum = 0.0
            for mlp_seed in MLP_SEEDS:
                key = f"mlp{mlp_seed}_k{branch}"
                full = np.stack(accum[key]["full"], axis=0)
                cheap = np.stack(accum[key][name], axis=0)
                one = pooled_rho2(full, cheap)
                per_mlp[f"{key}_{name}"] = one
                w = one["full_variance_weight_sum"]
                weighted.append((w, one))
                weight_sum += w
                all_full.append(full)
                all_cheap.append(cheap)
            metrics = pooled_rho2(np.concatenate(all_full, axis=0), np.concatenate(all_cheap, axis=0))
            h = cost_ratio(name, branch)
            metrics["cost_ratio_h"] = h
            metrics["memo_variance_factor"] = (1.0 - metrics["variance_weighted_pooled_rho2"]) * (1.0 + h)
            pooled_by_branch[str(branch)][name] = metrics
            within = {}
            for field in ("variance_weighted_pooled_rho2", "mean_coordinate_rho2", "scalar_sum_rho2"):
                within[field] = float(sum(w * one[field] for w, one in weighted) / weight_sum)
            within["cost_ratio_h"] = h
            within["memo_variance_factor"] = (1.0 - within["variance_weighted_pooled_rho2"]) * (1.0 + h)
            within_mlp_weighted_by_branch[str(branch)][name] = within

    gate_survivors = []
    for branch, items in within_mlp_weighted_by_branch.items():
        for name, metrics in items.items():
            if name in {"full_control", "noise_degraded_full", "row_subsample_q25"}:
                continue
            if metrics["variance_weighted_pooled_rho2"] >= 0.45 and metrics["memo_variance_factor"] <= 0.645:
                gate_survivors.append({"branch_layer": int(branch), "candidate": name, **metrics})

    result = {
        "config": {
            "mlp_seeds": MLP_SEEDS,
            "estimator_seeds": ESTIMATOR_SEEDS,
            "branch_layers": BRANCH_LAYERS,
            "rows": ROWS,
            "width": WIDTH,
            "depth": DEPTH,
        },
        "pooled_by_branch": pooled_by_branch,
        "within_mlp_weighted_by_branch": within_mlp_weighted_by_branch,
        "per_mlp": per_mlp,
        "gate_survivors": gate_survivors,
        "seconds": time.perf_counter() - started,
    }
    out = OUTDIR / "suffix_cv_pretest_results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_survivors": gate_survivors, "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
