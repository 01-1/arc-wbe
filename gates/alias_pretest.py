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


def truth_final(weights: list[np.ndarray], n_total: int, seed: int) -> tuple[np.ndarray, float]:
    if n_total % 2:
        raise ValueError("truth samples must be even for antithetic pairs")
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


def randomized_hadamard_half_blocks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    flips = []
    for _ in range(BLOCKS):
        d = (2.0 * rng.integers(0, 2, size=WIDTH) - 1.0).astype(np.float64)
        flips.append(d)
        blocks.append(HADAMARD * d[None, :])
    return np.concatenate(blocks, axis=0), np.stack(flips)


def estimator_run(weights: list[np.ndarray], seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    w0 = weights[0].astype(np.float64)
    x_half, flips = randomized_hadamard_half_blocks(rng)
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

    for layer_idx, w in enumerate(weights[1:], start=1):
        pre = x @ w.astype(np.float64)
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
    return {"final": x.mean(axis=0), "flips": flips}


def make_quadruples(rng: np.random.Generator, n: int) -> np.ndarray:
    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    while len(out) < n:
        abc = rng.choice(WIDTH, size=3, replace=False)
        a, b, c = (int(v) for v in abc)
        d = a ^ b ^ c
        if d in (a, b, c):
            continue
        q = tuple(sorted((a, b, c, d)))
        if len(set(q)) == 4 and q not in seen:
            seen.add(q)
            out.append(q)
    return np.array(out, dtype=np.int16)


def make_sextuples(rng: np.random.Generator, n: int) -> np.ndarray:
    out: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    while len(out) < n:
        vals = [int(v) for v in rng.choice(WIDTH, size=5, replace=False)]
        last = vals[0] ^ vals[1] ^ vals[2] ^ vals[3] ^ vals[4]
        if last in vals:
            continue
        s = tuple(sorted(vals + [last]))
        if len(set(s)) == 6 and s not in seen:
            seen.add(s)
            out.append(s)
    return np.array(out, dtype=np.int16)


def make_pairs(rng: np.random.Generator, n: int) -> np.ndarray:
    out = set()
    while len(out) < n:
        a, b = (int(v) for v in rng.choice(WIDTH, size=2, replace=False))
        out.add(tuple(sorted((a, b))))
    return np.array(sorted(out), dtype=np.int16)


def sketch_library(weights: list[np.ndarray], seed: int, quad_n: int, sext_n: int, pair_n: int) -> dict:
    rng = np.random.default_rng(seed)
    w0 = weights[0].astype(np.float64)
    col_norm = np.linalg.norm(w0, axis=0)
    gram = w0.T @ w0
    families = []
    quad_total = (WIDTH * (WIDTH - 1) * (WIDTH - 2)) // 24
    sext_total_approx = (WIDTH**5) // 720

    for i in range(12):
        q = make_quadruples(rng, quad_n)
        families.append({"name": f"q4_unweighted_{i:02d}", "kind": "quad", "idx": q, "coeff": np.ones(len(q))})

    for name in ("q4_w0_colnorm", "q4_w0_gram_chain", "q4_w0_gram_star", "q4_w0_absgram_pair"):
        q = make_quadruples(rng, quad_n)
        a, b, c, d = (q[:, i] for i in range(4))
        if name == "q4_w0_colnorm":
            coeff = col_norm[a] * col_norm[b] * col_norm[c] * col_norm[d]
        elif name == "q4_w0_gram_chain":
            coeff = gram[a, b] * gram[c, d] + gram[a, c] * gram[b, d] + gram[a, d] * gram[b, c]
        elif name == "q4_w0_gram_star":
            coeff = np.sum(np.abs(w0[:, a] * w0[:, b] * w0[:, c] * w0[:, d]), axis=0)
        else:
            coeff = np.abs(gram[a, b] * gram[c, d]) + np.abs(gram[a, c] * gram[b, d]) + np.abs(gram[a, d] * gram[b, c])
        families.append({"name": name, "kind": "quad", "idx": q, "coeff": coeff})

    for i in range(4):
        s = make_sextuples(rng, sext_n)
        families.append({"name": f"q6_unweighted_{i:02d}", "kind": "sext", "idx": s, "coeff": np.ones(len(s))})

    for i in range(4):
        p = make_pairs(rng, pair_n)
        families.append({"name": f"q2_control_{i:02d}", "kind": "pair", "idx": p, "coeff": np.ones(len(p))})

    return {
        "families": families,
        "coverage": {
            "quad_total_xor_closed_distinct_estimate": quad_total,
            "quad_per_sketch": quad_n,
            "quad_union_requested": 16 * quad_n,
            "quad_union_fraction_requested": (16 * quad_n) / quad_total,
            "sext_total_xor_closed_rough_estimate": sext_total_approx,
            "sext_per_sketch": sext_n,
            "pair_per_control": pair_n,
        },
    }


def features_from_flips(flips: np.ndarray, lib: dict) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    cols = []
    names = []
    groups: dict[str, list[int]] = {"q4_unweighted": [], "q4_weighted": [], "q6_unweighted": [], "q2_control": []}
    for fam in lib["families"]:
        idx = fam["idx"]
        prod = np.prod(flips[:, idx], axis=2)
        coeff = np.asarray(fam["coeff"], dtype=np.float64)
        value = float(np.sum(prod * coeff[None, :]))
        scale = math.sqrt(BLOCKS * float(np.sum(coeff * coeff)))
        cols.append(value / max(scale, 1e-12))
        names.append(fam["name"])
        if fam["name"].startswith("q4_unweighted"):
            groups["q4_unweighted"].append(len(cols) - 1)
        elif fam["name"].startswith("q4_w0"):
            groups["q4_weighted"].append(len(cols) - 1)
        elif fam["name"].startswith("q6"):
            groups["q6_unweighted"].append(len(cols) - 1)
        elif fam["name"].startswith("q2"):
            groups["q2_control"].append(len(cols) - 1)
    groups["all_q4"] = groups["q4_unweighted"] + groups["q4_weighted"]
    groups["all_alias"] = groups["all_q4"] + groups["q6_unweighted"]
    groups["all_with_controls"] = list(range(len(cols)))
    return np.array(cols, dtype=np.float64), names, groups


def standardize_train(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = Xtr.mean(axis=0)
    std = np.maximum(Xtr.std(axis=0), 1e-12)
    return (Xtr - mean) / std, (Xte - mean) / std


def cv_predict(Y: np.ndarray, X: np.ndarray, folds: int = 5) -> np.ndarray:
    n = Y.shape[0]
    pred = np.zeros_like(Y)
    indices = np.arange(n)
    for fold in range(folds):
        test = indices[fold::folds]
        train = np.setdiff1d(indices, test)
        Xtr, Xte = standardize_train(X[train], X[test])
        Xtr = np.column_stack((np.ones(len(train)), Xtr))
        Xte = np.column_stack((np.ones(len(test)), Xte))
        beta = np.linalg.lstsq(Xtr, Y[train], rcond=None)[0]
        pred[test] = Xte @ beta
    return pred


def variance_weighted_r2(Y: np.ndarray, pred: np.ndarray) -> float:
    resid = Y - pred
    sse = np.sum(resid * resid, axis=0)
    centered = Y - Y.mean(axis=0)
    sst = np.sum(centered * centered, axis=0)
    r2 = 1.0 - sse / np.maximum(sst, MIN_VARIANCE)
    weights = np.var(Y, axis=0, ddof=1)
    return float(np.sum(weights * r2) / np.maximum(np.sum(weights), MIN_VARIANCE))


def scalar_mean_r2(Y: np.ndarray, X: np.ndarray) -> float:
    y = Y.mean(axis=1, keepdims=True)
    return variance_weighted_r2(y, cv_predict(y, X))


def score_groups(errors: np.ndarray, features: np.ndarray, groups: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    out = {}
    for name, cols in groups.items():
        X = features[:, cols]
        pred = cv_predict(errors, X)
        out[name] = {
            "n_features": len(cols),
            "cv_r2_var_weighted_per_coord": variance_weighted_r2(errors, pred),
            "cv_r2_scalar_mean_error": scalar_mean_r2(errors, X),
            "mean_feature_std": float(np.mean(np.std(X, axis=0, ddof=1))),
            "min_feature_std": float(np.min(np.std(X, axis=0, ddof=1))),
        }
    return out


def positive_control(features: np.ndarray, groups: dict[str, list[int]], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    cols = groups["all_alias"]
    if len(cols) > 6:
        cols = list(rng.choice(cols, size=6, replace=False))
    X = features[:, cols]
    Xs = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    beta = rng.standard_normal((len(cols), 8))
    signal = Xs @ beta
    signal = signal / np.maximum(signal.std(axis=0, keepdims=True), 1e-12)
    noise = rng.standard_normal(signal.shape)
    target_r2 = 0.50
    y = signal + noise * math.sqrt((1.0 - target_r2) / target_r2)
    pred = cv_predict(y, X)
    return {
        "target_r2": target_r2,
        "recovered_cv_r2": variance_weighted_r2(y, pred),
        "n_features": len(cols),
        "n_synthetic_outputs": y.shape[1],
    }


def analyze_mlp(mlp_seed: int, R: int, truth_samples: int, args: argparse.Namespace) -> dict:
    weights = build_mlp(mlp_seed)
    truth, truth_noise = truth_final(weights, truth_samples, 900_000 + mlp_seed)
    lib = sketch_library(weights, 700_000 + mlp_seed, args.quad_per_sketch, args.sext_per_sketch, args.pair_per_control)
    finals = []
    feats = []
    names = None
    groups = None
    t0 = time.perf_counter()
    for r in range(R):
        run = estimator_run(weights, START_SEED + r)
        f, names, groups = features_from_flips(run["flips"], lib)
        finals.append(run["final"])
        feats.append(f)
    finals_a = np.stack(finals)
    feats_a = np.stack(feats)
    errors = finals_a - truth[None, :]
    assert names is not None and groups is not None
    return {
        "mlp_seed": mlp_seed,
        "R": R,
        "truth_samples": truth_samples,
        "truth_noise_mse": truth_noise,
        "estimator_seconds": time.perf_counter() - t0,
        "mean_seed_mse": float(np.mean(errors * errors)),
        "mean_seed_variance": float(np.mean(np.var(errors, axis=0, ddof=1))),
        "feature_names": names,
        "feature_groups": groups,
        "feature_std_by_name": {name: float(std) for name, std in zip(names, np.std(feats_a, axis=0, ddof=1))},
        "coverage": lib["coverage"],
        "scores": score_groups(errors, feats_a, groups),
        "positive_control": positive_control(feats_a, groups, 800_000 + mlp_seed),
        "_errors": errors,
        "_features": feats_a,
    }


def strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def pooled(per: list[dict]) -> dict:
    errors = np.concatenate([d["_errors"] for d in per], axis=0)
    features = np.concatenate([d["_features"] for d in per], axis=0)
    groups = per[0]["feature_groups"]
    return {
        "scores": score_groups(errors, features, groups),
        "positive_control": positive_control(features, groups, 880_000),
        "mean_seed_mse": float(np.mean([d["mean_seed_mse"] for d in per])),
        "mean_seed_variance": float(np.mean([d["mean_seed_variance"] for d in per])),
    }


def write_report(results: dict) -> None:
    p = results["pooled"]
    lines = [
        "# Alias-Correlation Pre-Test for High-Order-Even Cubature",
        "",
        f"Run date: 2026-07-05. Offline only; local self-generated MLPs and MC truth.",
        f"MLP seeds: {results['mlp_seeds']}; R per MLP: {results['R']}; truth samples: {results['truth_samples']:,}.",
        "",
        "## Pooled CV R^2",
        "",
        "| Feature family | n features | per-coordinate variance-weighted CV R^2 | scalar mean-error CV R^2 | mean feature std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in p["scores"].items():
        lines.append(
            f"| {name} | {row['n_features']} | {row['cv_r2_var_weighted_per_coord']:.4f} | "
            f"{row['cv_r2_scalar_mean_error']:.4f} | {row['mean_feature_std']:.3f} |"
        )
    lines.extend(["", "## Per MLP", ""])
    for d in results["per_mlp"]:
        lines.append(f"### MLP seed {d['mlp_seed']}")
        lines.append("")
        lines.append("| Feature family | per-coordinate CV R^2 | scalar mean-error CV R^2 |")
        lines.append("|---|---:|---:|")
        for name, row in d["scores"].items():
            lines.append(f"| {name} | {row['cv_r2_var_weighted_per_coord']:.4f} | {row['cv_r2_scalar_mean_error']:.4f} |")
        lines.append("")
    pc = p["positive_control"]
    cov = results["per_mlp"][0]["coverage"]
    gate_r2 = p["scores"]["all_alias"]["cv_r2_var_weighted_per_coord"]
    verdict = "SURVIVES" if gate_r2 >= 0.35 else "DEAD"
    lines.extend([
        "## Power and Coverage",
        "",
        f"Positive control target R^2: {pc['target_r2']:.2f}; recovered pooled CV R^2: {pc['recovered_cv_r2']:.4f}.",
        f"Quadruple sketches used {cov['quad_per_sketch']:,} tuples each across 16 requested degree-4 sketches, "
        f"covering about {100.0 * cov['quad_union_fraction_requested']:.2f}% of the distinct XOR-closed quadruple count before duplicate overlap.",
        f"Degree-6 sketches used {cov['sext_per_sketch']:,} sextuples each; this is sparse relative to the rough XOR-closed sextuple count.",
        "",
        "## Gate Verdict",
        "",
        f"Pre-registered pooled all-alias per-coordinate CV R^2: {gate_r2:.4f}. Candidate 3 is **{verdict}** under the 0.35 gate.",
    ])
    if verdict == "DEAD":
        lines.append(
            "The tested alias sketches do not resolve the fingerprint; alias-targeted sign designs should not be implemented from this evidence, and external information remains the remaining lever."
        )
    else:
        lines.append("The alias signal clears the gate; inspect the carrying family before designing a sign schedule.")
    (ROOT / "alias_pretest_20260705.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-seeds", type=int, nargs="+", default=[11, 22])
    parser.add_argument("--R", type=int, default=80)
    parser.add_argument("--truth-samples", type=int, default=200_000)
    parser.add_argument("--quad-per-sketch", type=int, default=20_000)
    parser.add_argument("--sext-per-sketch", type=int, default=8_000)
    parser.add_argument("--pair-per-control", type=int, default=20_000)
    args = parser.parse_args()
    started = time.perf_counter()
    per = []
    for seed in args.mlp_seeds:
        print(f"alias pretest mlp_seed={seed} R={args.R}", flush=True)
        d = analyze_mlp(seed, args.R, args.truth_samples, args)
        print(json.dumps(strip_private(d), indent=2), flush=True)
        per.append(d)
    results = {
        "mlp_seeds": args.mlp_seeds,
        "R": args.R,
        "truth_samples": args.truth_samples,
        "quad_per_sketch": args.quad_per_sketch,
        "sext_per_sketch": args.sext_per_sketch,
        "pair_per_control": args.pair_per_control,
        "elapsed_seconds": time.perf_counter() - started,
        "per_mlp": [strip_private(d) for d in per],
        "pooled": pooled(per),
    }
    (ROOT / "alias_pretest_results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results)
    summary = {
        "pooled_scores": results["pooled"]["scores"],
        "positive_control": results["pooled"]["positive_control"],
        "gate_all_alias_r2": results["pooled"]["scores"]["all_alias"]["cv_r2_var_weighted_per_coord"],
        "elapsed_seconds": results["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
