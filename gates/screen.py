#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


WIDTH = 256
DEPTH = 32
MLP_SEEDS = (11, 22, 33)
DEFAULT_R = 100
MIN_R = 60
TRUTH_SAMPLES = 400_000
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


def gaussian_relu_variance(mean_pre: np.ndarray, var_pre: np.ndarray) -> np.ndarray:
    var_pre = np.maximum(var_pre, MIN_VARIANCE)
    sigma = np.sqrt(var_pre)
    alpha = mean_pre / sigma
    phi = normal_pdf(alpha)
    Phi = normal_cdf(alpha)
    relu_mean = sigma * phi + mean_pre * Phi
    second = (mean_pre * mean_pre + var_pre) * Phi + mean_pre * sigma * phi
    return np.maximum(second - relu_mean * relu_mean, MIN_VARIANCE)


def truth_layer_means(weights: list[np.ndarray], n_total: int, seed: int) -> tuple[np.ndarray, float]:
    if n_total % 2:
        raise ValueError("truth samples must be even for antithetic pairs")
    rng = np.random.default_rng(seed)
    batch_pairs = 10_000
    rows_sum = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    final_sum = np.zeros(WIDTH, dtype=np.float64)
    final_sum_sq = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < n_total:
        pairs = min(batch_pairs, (n_total - done) // 2)
        x0 = rng.standard_normal((pairs, WIDTH)).astype(np.float32)
        x = np.concatenate((x0, -x0), axis=0).astype(np.float64)
        for i, w in enumerate(weights):
            x = np.maximum(x @ w.astype(np.float64), 0.0)
            rows_sum[i] += x.sum(axis=0)
        final_sum += x.sum(axis=0)
        final_sum_sq += (x * x).sum(axis=0)
        done += 2 * pairs
    means = rows_sum / n_total
    final_mean = final_sum / n_total
    final_var = final_sum_sq / n_total - final_mean * final_mean
    truth_noise_mse = float(np.mean(final_var) / n_total)
    return means, truth_noise_mse


def randomized_hadamard_half_blocks(rng: np.random.Generator) -> np.ndarray:
    blocks = []
    for _ in range(BLOCKS):
        flips = 2.0 * rng.integers(0, 2, size=WIDTH) - 1.0
        blocks.append(HADAMARD * flips[None, :])
    return np.concatenate(blocks, axis=0)


def estimator_run(weights: list[np.ndarray], seed: int, validate: bool = False) -> dict[str, np.ndarray | float]:
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
    x1 = x.copy()

    recolor_mean_rel = float(np.linalg.norm(x.mean(axis=0) - target_mean) / max(np.linalg.norm(target_mean), MIN_VARIANCE))
    xc = x - x.mean(axis=0)
    xcov = (xc.T @ xc) / x.shape[0]
    recolor_cov_rel = float(np.linalg.norm(xcov - target_cov) / max(np.linalg.norm(target_cov), MIN_VARIANCE))

    rows = [target_mean.copy()]
    mid = None
    pre2 = None
    for layer_idx, w in enumerate(weights[1:], start=1):
        pre = x @ w.astype(np.float64)
        if layer_idx == 1:
            pre2 = pre.copy()
        x = np.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = pre.mean(axis=0)
            pre_centered = pre - pre_mean[None, :]
            target_var = gaussian_relu_variance(pre_mean, np.mean(pre_centered * pre_centered, axis=0))
            sample_mean2 = x.mean(axis=0)
            centered_layer = x - sample_mean2[None, :]
            sample_var = np.maximum(np.mean(centered_layer * centered_layer, axis=0), MIN_VARIANCE)
            scale = 1.0 + 1.5 * (np.sqrt(target_var / sample_var) - 1.0)
            x = centered_layer * scale[None, :] + sample_mean2[None, :]
        rows.append(x.mean(axis=0))
        if layer_idx == 15:
            mid = rows[-1].copy()

    block_means = []
    for b in range(BLOCKS):
        pos = x[b * WIDTH : (b + 1) * WIDTH]
        neg = x[BLOCKS * WIDTH + b * WIDTH : BLOCKS * WIDTH + (b + 1) * WIDTH]
        block_means.append(np.concatenate((pos, neg), axis=0).mean(axis=0))
    features = anchored_features(weights, x1, target_mean, target_cov, cov_pre, pre2)
    out = {
        "means": np.stack(rows, axis=0),
        "mid16": mid,
        "final": rows[-1].copy(),
        "block_means": np.stack(block_means, axis=0),
        "recolor_mean_rel": recolor_mean_rel,
        "recolor_cov_rel": recolor_cov_rel,
    }
    out.update(features)
    return out


def anchored_features(
    weights: list[np.ndarray],
    x1: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    cov_pre: np.ndarray,
    pre2: np.ndarray,
) -> dict[str, np.ndarray | float]:
    sigma = np.sqrt(np.maximum(np.diag(cov_pre), MIN_VARIANCE))
    tstd = np.sqrt(np.maximum(np.diag(target_cov), MIN_VARIANCE))
    x_center = x1 - x1.mean(axis=0)
    m3_sample = np.mean(x_center ** 3, axis=0)
    c3 = math.sqrt(2.0 / math.pi) - 1.5 / math.sqrt(2.0 * math.pi) + 2.0 / ((2.0 * math.pi) ** 1.5)
    m3_target = (sigma ** 3) * c3
    v1 = (m3_sample - m3_target) / (tstd ** 3)
    m1 = sigma / math.sqrt(2.0 * math.pi)
    r2 = sigma * sigma / 2.0
    r3 = sigma ** 3 * math.sqrt(2.0 / math.pi)
    r4 = 1.5 * sigma ** 4
    mu4 = r4 - 4.0 * m1 * r3 + 6.0 * m1 * m1 * r2 - 3.0 * m1 ** 4
    v2 = (np.mean(x_center ** 4, axis=0) - mu4) / (tstd ** 4)

    w1 = weights[1].astype(np.float64)
    m2 = target_mean @ w1
    s2 = np.sqrt(np.maximum(np.diag(w1.T @ target_cov @ w1), MIN_VARIANCE))
    f3_vec = np.mean(pre2 > 0.0, axis=0) - normal_cdf(m2 / s2)
    radii = np.sum(x1 * x1, axis=1)
    sample_radius_var = np.var(radii)
    closure_radius_var = 2.0 * np.trace(target_cov @ target_cov) + 4.0 * float(target_mean @ target_cov @ target_mean)
    f4 = (sample_radius_var - closure_radius_var) / max(closure_radius_var, MIN_VARIANCE)
    return {
        "F": np.array([np.mean(v1), np.mean(v2), np.mean(f3_vec), f4], dtype=np.float64),
        "V1": v1.astype(np.float64),
        "V2": v2.astype(np.float64),
    }


def adjusted_r2_scalar_cv(errors: np.ndarray, F: np.ndarray) -> float:
    n, p = F.shape
    X = np.column_stack((np.ones(n), F))
    y = errors
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ betas
    sse = np.sum(resid * resid, axis=0)
    centered = y - y.mean(axis=0)
    sst = np.sum(centered * centered, axis=0)
    r2_adj = 1.0 - (sse / max(n - p - 1, 1)) / np.maximum(sst / max(n - 1, 1), MIN_VARIANCE)
    weights = np.var(y, axis=0, ddof=1)
    return float(np.sum(weights * r2_adj) / np.sum(weights))


def cv_r2_generous(errors: np.ndarray, F: np.ndarray, V: np.ndarray, folds: int = 5) -> float:
    n = errors.shape[0]
    Vc = V - V.mean(axis=0)
    _, _, vt = np.linalg.svd(Vc, full_matrices=False)
    pcs = Vc @ vt[:8].T
    X0 = np.column_stack((F, pcs))
    preds = np.zeros_like(errors)
    indices = np.arange(n)
    for fold in range(folds):
        test = indices[fold::folds]
        train = np.setdiff1d(indices, test)
        mean = X0[train].mean(axis=0)
        std = np.maximum(X0[train].std(axis=0), 1e-12)
        Xtr = np.column_stack((np.ones(len(train)), (X0[train] - mean) / std))
        Xte = np.column_stack((np.ones(len(test)), (X0[test] - mean) / std))
        beta = np.linalg.lstsq(Xtr, errors[train], rcond=None)[0]
        preds[test] = Xte @ beta
    resid = errors - preds
    sse = np.sum(resid * resid, axis=0)
    centered = errors - errors.mean(axis=0)
    sst = np.sum(centered * centered, axis=0)
    r2 = 1.0 - sse / np.maximum(sst, MIN_VARIANCE)
    weights = np.var(errors, axis=0, ddof=1)
    return float(np.sum(weights * r2) / np.sum(weights))


def analyze_mlp(seed: int, R: int, truth_samples: int) -> dict:
    weights = build_mlp(seed)
    t0 = time.perf_counter()
    truth, truth_noise = truth_layer_means(weights, truth_samples, seed=900_000 + seed)
    truth_seconds = time.perf_counter() - t0
    first = estimator_run(weights, START_SEED, validate=True)
    first_mse = float(np.mean((first["final"] - truth[-1]) ** 2))
    if not (first["recolor_mean_rel"] <= 1e-6 and first["recolor_cov_rel"] <= 1e-6):
        raise RuntimeError(f"recolor validation failed for mlp {seed}: {first['recolor_mean_rel']=} {first['recolor_cov_rel']=}")
    if not (1e-6 <= first_mse <= 8e-6):
        raise RuntimeError(f"single-seed MSE gate failed for mlp {seed}: {first_mse}")

    finals, mids, Fs, Vs, blocks = [], [], [], [], []
    t1 = time.perf_counter()
    for r in range(R):
        run = first if r == 0 else estimator_run(weights, START_SEED + r)
        finals.append(run["final"])
        mids.append(run["mid16"])
        Fs.append(run["F"])
        Vs.append(np.concatenate((run["V1"], run["V2"])))
        blocks.append(run["block_means"])
    est_seconds = time.perf_counter() - t1
    finals = np.stack(finals)
    mids = np.stack(mids)
    Fs = np.stack(Fs)
    Vs = np.stack(Vs)
    blocks = np.stack(blocks)
    errors = finals - truth[-1][None, :]
    mid_errors = mids - truth[15][None, :]
    honest = adjusted_r2_scalar_cv(errors, Fs)
    generous = cv_r2_generous(errors, Fs, Vs)
    bias = errors.mean(axis=0)
    seed_var = np.var(errors, axis=0, ddof=1)
    block_var = np.var(blocks, axis=1, ddof=1)
    decomp_num = float(np.mean(seed_var))
    decomp_den = float(np.mean(block_var) / BLOCKS)
    bias_mse = float(np.mean(bias * bias))
    seed_mean_var_over_R = float(np.mean(seed_var) / R)
    bias_net = float(bias_mse - truth_noise - seed_mean_var_over_R)
    return {
        "mlp_seed": seed,
        "R": R,
        "truth_samples": truth_samples,
        "truth_seconds": truth_seconds,
        "estimator_seconds": est_seconds,
        "single_seed_mse": first_mse,
        "recolor_mean_rel": float(first["recolor_mean_rel"]),
        "recolor_cov_rel": float(first["recolor_cov_rel"]),
        "honest_scalar_adjusted_r2": honest,
        "generous_cv_r2": generous,
        "bias_mse": bias_mse,
        "truth_noise_mse": truth_noise,
        "seed_mean_variance_over_R": seed_mean_var_over_R,
        "bias_mse_net_of_noise": bias_net,
        "variance_decomposition_ratio": decomp_num / decomp_den,
        "sanity_mean_seed_mse": float(np.mean(errors * errors)),
        "mid16_mean_seed_mse": float(np.mean(mid_errors * mid_errors)),
        "_errors": errors,
        "_F": Fs,
        "_V": Vs,
    }


def strip_arrays(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def pooled_metrics(per_mlp: list[dict]) -> dict:
    errors = np.concatenate([d["_errors"] for d in per_mlp], axis=0)
    F = np.concatenate([d["_F"] for d in per_mlp], axis=0)
    V = np.concatenate([d["_V"] for d in per_mlp], axis=0)
    return {
        "honest_scalar_adjusted_r2": adjusted_r2_scalar_cv(errors, F),
        "generous_cv_r2": cv_r2_generous(errors, F, V),
        "bias_mse_mean": float(np.mean([d["bias_mse"] for d in per_mlp])),
        "bias_mse_net_of_noise_mean": float(np.mean([d["bias_mse_net_of_noise"] for d in per_mlp])),
        "variance_decomposition_ratio_mean": float(np.mean([d["variance_decomposition_ratio"] for d in per_mlp])),
        "sanity_mean_seed_mse": float(np.mean([d["sanity_mean_seed_mse"] for d in per_mlp])),
    }


def write_report(results: dict) -> None:
    lines = ["# Offline anchored-CV screen", ""]
    lines.append(f"R per MLP: {results['R']}; truth samples per MLP: {results['truth_samples']:,}")
    lines.append("")
    lines.append("| MLP seed | honest adj R^2 | generous CV R^2 | bias MSE | truth-noise MSE | seed-var/R | net bias MSE | decomp ratio | sanity MSE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in results["per_mlp"]:
        lines.append(
            f"| {d['mlp_seed']} | {d['honest_scalar_adjusted_r2']:.4f} | {d['generous_cv_r2']:.4f} | "
            f"{d['bias_mse']:.4e} | {d['truth_noise_mse']:.4e} | {d['seed_mean_variance_over_R']:.4e} | "
            f"{d['bias_mse_net_of_noise']:.4e} | {d['variance_decomposition_ratio']:.3f} | {d['sanity_mean_seed_mse']:.4e} |"
        )
    p = results["pooled"]
    lines.extend([
        "",
        "## Pooled",
        "",
        f"- Honest scalar adjusted R^2: {p['honest_scalar_adjusted_r2']:.4f}",
        f"- Generous CV R^2: {p['generous_cv_r2']:.4f}",
        f"- Mean bias MSE: {p['bias_mse_mean']:.4e}",
        f"- Mean net bias MSE: {p['bias_mse_net_of_noise_mean']:.4e}",
        f"- Mean variance-decomposition ratio: {p['variance_decomposition_ratio_mean']:.3f}",
        f"- Mean sanity MSE: {p['sanity_mean_seed_mse']:.4e}",
        "",
        "## Verdict",
        "",
    ])
    threshold = 0.40
    if p["honest_scalar_adjusted_r2"] >= threshold or p["generous_cv_r2"] >= threshold:
        verdict = "At least one anchored-feature ceiling meets the 40% decision threshold."
    else:
        verdict = "Anchored features do not explain >= 40% of final-error variance in this offline screen."
    lines.append(verdict)
    (ROOT / "screen_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=DEFAULT_R)
    parser.add_argument("--truth-samples", type=int, default=TRUTH_SAMPLES)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.R < MIN_R and not args.probe:
        raise SystemExit(f"R must be at least {MIN_R}")
    if args.probe:
        weights = build_mlp(MLP_SEEDS[0])
        t0 = time.perf_counter()
        truth, truth_noise = truth_layer_means(weights, args.truth_samples, seed=900_000 + MLP_SEEDS[0])
        truth_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        run = estimator_run(weights, START_SEED)
        est_seconds = time.perf_counter() - t1
        mse = float(np.mean((run["final"] - truth[-1]) ** 2))
        print(json.dumps({
            "truth_seconds": truth_seconds,
            "estimator_seconds_one_seed": est_seconds,
            "single_seed_mse": mse,
            "truth_noise_mse": truth_noise,
            "recolor_mean_rel": run["recolor_mean_rel"],
            "recolor_cov_rel": run["recolor_cov_rel"],
        }, indent=2))
        return
    started = time.perf_counter()
    per = []
    for seed in MLP_SEEDS:
        print(f"running mlp_seed={seed} R={args.R}", flush=True)
        d = analyze_mlp(seed, args.R, args.truth_samples)
        print(json.dumps(strip_arrays(d), indent=2), flush=True)
        per.append(d)
    results = {
        "R": args.R,
        "truth_samples": args.truth_samples,
        "elapsed_seconds": time.perf_counter() - started,
        "per_mlp": [strip_arrays(d) for d in per],
        "pooled": pooled_metrics(per),
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results)
    print(json.dumps({"pooled": results["pooled"], "elapsed_seconds": results["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
