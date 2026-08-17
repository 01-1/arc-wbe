#!/usr/bin/env python3
"""Stage-1 deterministic filament-grid propagation gate.

Research-only harness.  It deliberately uses the same 400k Monte Carlo sample
set both as "truth" and as a near-perfect branch-layer initializer, so the
reported error isolates the deterministic propagation machinery rather than
estimator-legal initialization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_engine import build_mlp  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
JSON_PATH = OUT_DIR / "filament_stage1_20260706_results.json"
MD_PATH = OUT_DIR / "filament_stage1_20260706.md"

WIDTH = 256
DEPTH = 32
SEEDS = (11, 22)
KS = (16, 24)
GS = (9, 17, 33, 65)
TRUTH_N = 400_000
TRUTH_BATCH = 10_000
MIN_VARIANCE = 1e-12
FINE_R1_BINS = 513
FINE_R2_BINS = 31

GL16_NODES = np.array(
    [
        -0.9894009349916499,
        -0.9445750230732326,
        -0.8656312023878318,
        -0.7554044083550030,
        -0.6178762444026438,
        -0.4580167776572274,
        -0.2816035507792589,
        -0.0950125098376374,
        0.0950125098376374,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    ],
    dtype=np.float64,
)
GL16_WEIGHTS = np.array(
    [
        0.0271524594117541,
        0.0622535239386479,
        0.0951585116824928,
        0.1246289712555339,
        0.1495959888165767,
        0.1691565193950025,
        0.1826034150449236,
        0.1894506104550685,
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    ],
    dtype=np.float64,
)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def mlp_weights_np(seed: int) -> list[np.ndarray]:
    mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    return [np.asarray(w, dtype=np.float64) for w in mlp.weights]


@dataclass
class TruthPack:
    means: dict[int, np.ndarray]
    branches: dict[int, np.ndarray]
    wall_time_s: float


def collect_truth_and_branches(
    weights: list[np.ndarray],
    *,
    seed: int,
    n_samples: int,
    batch: int,
) -> TruthPack:
    if n_samples % 2:
        raise ValueError("truth sample count must be even for antithetic generation")
    rng = np.random.default_rng(100_000 + seed)
    layers = set(range(min(KS), DEPTH + 1))
    sums = {layer: np.zeros(WIDTH, dtype=np.float64) for layer in layers}
    branches = {
        k: np.empty((n_samples, WIDTH), dtype=np.float32)
        for k in KS
    }
    done = 0
    started = time.time()
    while done < n_samples:
        b = min(batch, n_samples - done)
        half = (b + 1) // 2
        x0 = rng.standard_normal((half, WIDTH), dtype=np.float32)
        x = np.concatenate([x0, -x0], axis=0)[:b].astype(np.float64, copy=False)
        for layer_idx, w in enumerate(weights, start=1):
            x = np.maximum(x @ w, 0.0)
            if layer_idx in sums:
                sums[layer_idx] += x.sum(axis=0)
            if layer_idx in branches:
                branches[layer_idx][done : done + b] = x.astype(np.float32)
        done += b
        print(f"  seed={seed} truth {done:,}/{n_samples:,}", flush=True)
    means = {layer: total / n_samples for layer, total in sums.items()}
    return TruthPack(means=means, branches=branches, wall_time_s=time.time() - started)


def symmetrize_cov(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    diag = np.maximum(np.diag(cov), MIN_VARIANCE)
    cov = cov.copy()
    np.fill_diagonal(cov, diag)
    return cov


def relu_mean_cov(mean_pre: np.ndarray, cov_pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    var = np.maximum(np.diag(cov_pre), MIN_VARIANCE)
    sigma = np.sqrt(var)
    beta = mean_pre / sigma
    phi = norm_pdf(beta)
    Phi = norm_cdf(beta)
    relu_mean = sigma * phi + mean_pre * Phi
    marginal_second = (mean_pre * mean_pre + var) * Phi + mean_pre * sigma * phi

    denom = sigma[:, None] * sigma[None, :]
    rho = np.clip(cov_pre / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    alpha_i = -beta[:, None]
    alpha_j = -beta[None, :]
    tail_i = Phi[:, None]
    tail_j = Phi[None, :]
    rho_int = np.zeros_like(rho)
    for node, weight in zip(GL16_NODES, GL16_WEIGHTS):
        r = 0.5 * rho * (node + 1.0)
        one_minus = np.maximum(1.0 - r * r, MIN_VARIANCE)
        exponent = -(
            alpha_i * alpha_i - 2.0 * r * alpha_i * alpha_j + alpha_j * alpha_j
        ) / (2.0 * one_minus)
        phi2 = np.exp(exponent) / (2.0 * math.pi * np.sqrt(one_minus))
        rho_int += weight * (rho - r) * phi2
    second = relu_mean[:, None] * relu_mean[None, :] + denom * (
        rho * tail_i * tail_j + 0.5 * rho * rho_int
    )
    np.fill_diagonal(second, marginal_second)
    return relu_mean, symmetrize_cov(second - relu_mean[:, None] * relu_mean[None, :])


def fit_shared_residual(y: np.ndarray, rank: int) -> dict[str, Any]:
    yd = y.astype(np.float64, copy=False)
    mean = yd.mean(axis=0)
    yc = yd - mean
    cov = symmetrize_cov((yc.T @ yc) / yd.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    vals = evals[order[:rank]]
    vecs = evecs[:, order[:rank]]
    scores = yc @ vecs
    resid = yc - scores @ vecs.T
    residual_cov = symmetrize_cov((resid.T @ resid) / yd.shape[0])
    residual_cov_r1_curve = conditional_residual_cov(yd, scores[:, :1], (FINE_R1_BINS,))
    residual_cov_r2_curve = conditional_residual_cov(yd, scores[:, :2], (FINE_R2_BINS, FINE_R2_BINS))
    return {
        "mean": mean,
        "cov": cov,
        "evals": evals[order],
        "vecs": vecs,
        "scores": scores,
        "residual_cov_linear": residual_cov,
        "residual_cov_r1_curve": residual_cov_r1_curve,
        "residual_cov_r2_curve": residual_cov_r2_curve,
        "top_share": float(np.sum(vals) / max(float(np.trace(cov)), MIN_VARIANCE)),
    }


def quantile_codes(scores: np.ndarray, bins_by_dim: tuple[int, ...]) -> tuple[np.ndarray, int]:
    codes = np.zeros(scores.shape[0], dtype=np.int64)
    mult = 1
    for dim, bins in enumerate(bins_by_dim):
        edges = np.quantile(scores[:, dim], np.linspace(0.0, 1.0, bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        b = np.searchsorted(edges[1:-1], scores[:, dim], side="right")
        codes += mult * b
        mult *= bins
    return codes, mult


def conditional_residual_cov(
    yd: np.ndarray,
    scores: np.ndarray,
    bins_by_dim: tuple[int, ...],
) -> np.ndarray:
    codes, total_codes = quantile_codes(scores, bins_by_dim)
    order = np.argsort(codes)
    sorted_codes = codes[order]
    cov_sum = np.zeros((WIDTH, WIDTH), dtype=np.float64)
    start = 0
    for code in np.flatnonzero(np.bincount(codes, minlength=total_codes)):
        end = start
        while end < sorted_codes.shape[0] and sorted_codes[end] == code:
            end += 1
        cell = yd[order[start:end]]
        centered = cell - cell.mean(axis=0)
        cov_sum += centered.T @ centered
        start = end
    return symmetrize_cov(cov_sum / yd.shape[0])


@dataclass
class Mixture:
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    construction: dict[str, Any]


def make_r1_grid(y: np.ndarray, g: int, fit: dict[str, Any]) -> Mixture:
    a = fit["scores"][:, 0]
    u = fit["vecs"][:, 0]
    edges = np.quantile(a, np.linspace(0.0, 1.0, g + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    codes = np.searchsorted(edges[1:-1], a, side="right")
    counts = np.bincount(codes, minlength=g).astype(np.float64)

    yd = y.astype(np.float64, copy=False)
    weights = counts / yd.shape[0]
    means = np.zeros((g, WIDTH), dtype=np.float64)
    covs = np.zeros((g, WIDTH, WIDTH), dtype=np.float64)
    uu = np.outer(u, u)
    for cell in range(g):
        idx = codes == cell
        means[cell] = yd[idx].mean(axis=0)
        var_a = float(np.var(a[idx])) if np.count_nonzero(idx) > 1 else 0.0
        covs[cell] = symmetrize_cov(fit["residual_cov_r1_curve"] + var_a * uu)
    construction = {
        "rank": 1,
        "nodes": g,
        "cell_rule": "empirical equal-mass latent quantile cells",
        "node_mean": "empirical conditional mean per cell from all 400k truth samples",
        "node_covariance": "shared fine-curve residual covariance plus empirical within-cell latent variance uu^T",
        "fine_residual_bins": FINE_R1_BINS,
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "top1_share": fit["top_share"],
    }
    return Mixture(weights=weights, means=means, covs=covs, construction=construction)


def make_r2_grid(y: np.ndarray, bins: int, fit: dict[str, Any]) -> Mixture:
    scores = fit["scores"][:, :2]
    codes = np.zeros(scores.shape[0], dtype=np.int64)
    mult = 1
    for dim in range(2):
        edges = np.quantile(scores[:, dim], np.linspace(0.0, 1.0, bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        b = np.searchsorted(edges[1:-1], scores[:, dim], side="right")
        codes += mult * b
        mult *= bins

    counts = np.bincount(codes, minlength=bins * bins).astype(np.float64)
    active = np.flatnonzero(counts > 0)
    yd = y.astype(np.float64, copy=False)
    u = fit["vecs"][:, :2]
    weights = counts[active] / yd.shape[0]
    means = np.zeros((len(active), WIDTH), dtype=np.float64)
    covs = np.zeros((len(active), WIDTH, WIDTH), dtype=np.float64)
    for out_idx, code in enumerate(active):
        idx = codes == code
        means[out_idx] = yd[idx].mean(axis=0)
        if np.count_nonzero(idx) > 1:
            latent_cov = np.cov(scores[idx].T, bias=True)
        else:
            latent_cov = np.zeros((2, 2), dtype=np.float64)
        covs[out_idx] = symmetrize_cov(fit["residual_cov_r2_curve"] + u @ latent_cov @ u.T)
    construction = {
        "rank": 2,
        "nodes": int(len(active)),
        "bins_per_axis": bins,
        "cell_rule": "Cartesian empirical quantile cells on the top-2 latent coordinates",
        "node_covariance": "shared fine-grid residual covariance plus empirical within-cell 2D latent covariance",
        "fine_residual_bins_by_axis": FINE_R2_BINS,
        "min_count": int(counts[active].min()),
        "max_count": int(counts[active].max()),
        "top2_share": fit["top_share"],
    }
    return Mixture(weights=weights, means=means, covs=covs, construction=construction)


def make_single_gaussian(y: np.ndarray, fit: dict[str, Any]) -> Mixture:
    return Mixture(
        weights=np.array([1.0], dtype=np.float64),
        means=fit["mean"][None, :],
        covs=fit["cov"][None, :, :],
        construction={
            "rank": 0,
            "nodes": 1,
            "cell_rule": "single Gaussian matched to branch-layer empirical mean/covariance",
        },
    )


def propagate_mixture(
    mix: Mixture,
    weights: list[np.ndarray],
    *,
    start_layer: int,
    truth_means: dict[int, np.ndarray],
) -> dict[str, Any]:
    node_w = mix.weights
    means = mix.means.copy()
    covs = mix.covs.copy()
    layer_rows = []

    pred = node_w @ means
    err = pred - truth_means[start_layer]
    layer_rows.append(
        {
            "layer": start_layer,
            "mse": float(np.mean(err * err)),
            "rmse": float(math.sqrt(np.mean(err * err))),
            "max_abs": float(np.max(np.abs(err))),
            "mean_bias": float(np.mean(err)),
        }
    )
    started = time.time()
    for next_layer in range(start_layer + 1, DEPTH + 1):
        w = weights[next_layer - 1]
        pre_means = means @ w
        tmp = np.matmul(covs, w)
        pre_covs = np.matmul(w.T[None, :, :], tmp)

        next_means = np.empty_like(pre_means)
        next_covs = np.empty_like(pre_covs)
        for node_idx in range(means.shape[0]):
            next_means[node_idx], next_covs[node_idx] = relu_mean_cov(
                pre_means[node_idx],
                pre_covs[node_idx],
            )
        means = next_means
        covs = next_covs
        pred = node_w @ means
        err = pred - truth_means[next_layer]
        layer_rows.append(
            {
                "layer": next_layer,
                "mse": float(np.mean(err * err)),
                "rmse": float(math.sqrt(np.mean(err * err))),
                "max_abs": float(np.max(np.abs(err))),
                "mean_bias": float(np.mean(err)),
            }
        )
        print(
            f"    K={start_layer} nodes={len(node_w)} propagated layer {next_layer:02d} "
            f"mse={layer_rows[-1]['mse']:.3e}",
            flush=True,
        )

    return {
        "prediction_final": pred,
        "layer_diagnostics": layer_rows,
        "final_mse": layer_rows[-1]["mse"],
        "final_rmse": layer_rows[-1]["rmse"],
        "final_max_abs": layer_rows[-1]["max_abs"],
        "wall_time_s": time.time() - started,
    }


def fit_convergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gs = np.array([row["G"] for row in rows if row["G"] > 1], dtype=np.float64)
    ys = np.array([row["final_mse"] for row in rows if row["G"] > 1], dtype=np.float64)
    raw_coef = np.polyfit(np.log(gs), np.log(ys), 1)
    raw_p = float(-raw_coef[0])

    best = None
    ymin = float(ys.min())
    for floor in np.linspace(0.0, ymin * 0.95, 500):
        resid = ys - floor
        if np.any(resid <= 0.0):
            continue
        coef = np.polyfit(np.log(gs), np.log(resid), 1)
        pred = np.polyval(coef, np.log(gs))
        sse = float(np.mean((np.log(resid) - pred) ** 2))
        cand = (sse, floor, float(-coef[0]), float(math.exp(coef[1])))
        if best is None or cand < best:
            best = cand
    assert best is not None
    return {
        "raw_loglog_p": raw_p,
        "floor_fit_grid": {
            "floor": best[1],
            "p": best[2],
            "coef": best[3],
            "mean_log_sse": best[0],
            "model": "mse ~= floor + coef * G^-p; floor grid-searched over [0, 0.95*min(mse)]",
        },
        "largest_G_mse": float(ys[np.argmax(gs)]),
    }


def verdict(final_k24_g65: float, p: float) -> str:
    if final_k24_g65 <= 0.3e-6 and p >= 1.5:
        return "MACHINERY CONFIRMED"
    if final_k24_g65 <= 1.5e-6:
        return "PARTIALLY CONFIRMED"
    if final_k24_g65 > 3.0e-6 or p <= 0.0:
        return "MACHINERY INSUFFICIENT AS CONSTRUCTED"
    return "INCONCLUSIVE / BETWEEN REGISTERED BANDS"


def fmt_sci(x: float) -> str:
    return f"{x:.3e}"


def write_markdown(results: dict[str, Any]) -> None:
    lines = [
        "# Filament Stage-1 Deterministic Grid Propagation Gate (2026-07-06)",
        "",
        "Offline analysis-only run under `paired_fly_logs/fingerprint_theory/`. No Fly, network, pytest, or tracked-file edits. MLPs use `local_engine.build_mlp`, width 256, depth 32, seeds 11 and 22. Truth and initialization both use the same 400k antithetic sample set by design, so this gate removes initialization error and tests deterministic propagation machinery only.",
        "",
        "## Grid Construction",
        "",
        "For each branch layer K, the full branch activation sample is centered and diagonalized. The r=1 grid uses empirical equal-mass quantile cells along the top eigen-score `a = u.(y_K - mean)`. Each node stores its empirical cell mass, empirical conditional mean, and covariance `C_resid + Var(a|cell) uu^T`, where `C_resid` is pooled residual covariance after subtracting a 513-bin fine empirical conditional-mean curve. Thus the mixture mean exactly matches the sample mean at K up to roundoff, and the within-cell latent variance tiles the filament between node means.",
        "",
        "Propagation uses exact linear Gaussian moment propagation per node, then nonzero-mean Gaussian ReLU marginal moments plus the GL16 Price-identity bivariate covariance closure ported from the existing estimator/pretest code.",
        "",
        "## Bias-MSE Table",
        "",
        "| K | G | seed 11 | seed 22 | mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_kg: dict[tuple[int, int], list[float]] = {}
    for row in results["cases"]:
        if row["rank"] == 1:
            by_kg.setdefault((row["K"], row["G"]), []).append(row["final_mse"])
    for k in KS:
        for g in (1, *GS):
            vals = by_kg.get((k, g), [])
            if len(vals) == len(SEEDS):
                lines.append(
                    f"| {k} | {g} | {fmt_sci(vals[0])} | {fmt_sci(vals[1])} | {fmt_sci(float(np.mean(vals)))} |"
                )
    lines += ["", "## Convergence", "", "| K | raw p | floor-fit p | floor | largest-G MSE |", "|---:|---:|---:|---:|---:|"]
    for k in KS:
        conv = results["convergence"][str(k)]
        lines.append(
            f"| {k} | {conv['raw_loglog_p']:.3f} | {conv['floor_fit_grid']['p']:.3f} | {fmt_sci(conv['floor_fit_grid']['floor'])} | {fmt_sci(conv['largest_G_mse'])} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- K=24, G=65 mean final bias-MSE: `{fmt_sci(results['summary']['k24_g65_mean_mse'])}`",
        f"- Selected convergence order p: `{results['summary']['k24_floor_fit_p']:.3f}`",
        f"- Verdict: **{results['summary']['verdict']}**",
        "",
        "## Per-Layer Diagnostics",
        "",
        "Mean MSE across seeds for selected anchors.",
        "",
        "| K | G | layer | mean MSE |",
        "|---:|---:|---:|---:|",
    ]
    selected = {(16, 1), (16, 65), (24, 1), (24, 65)}
    diag_acc: dict[tuple[int, int, int], list[float]] = {}
    for row in results["cases"]:
        key = (row["K"], row["G"])
        if row["rank"] == 1 and key in selected:
            for layer_row in row["layer_diagnostics"]:
                diag_acc.setdefault((row["K"], row["G"], layer_row["layer"]), []).append(layer_row["mse"])
    for key in sorted(diag_acc):
        vals = diag_acc[key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt_sci(float(np.mean(vals)))} |")

    if results.get("r2_optional"):
        lines += ["", "## Optional r=2 Check", "", "| seed | K | grid | nodes | final MSE |", "|---:|---:|---|---:|---:|"]
        for row in results["r2_optional"]:
            lines.append(
                f"| {row['seed']} | {row['K']} | {row['grid']} | {row['nodes']} | {fmt_sci(row['final_mse'])} |"
            )

    lines += [
        "",
        "## Recommended Next Action",
        "",
        results["summary"]["recommended_next_action"],
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    started = time.time()
    for seed in SEEDS:
        print(f"seed={seed}: building MLP and collecting truth", flush=True)
        weights = mlp_weights_np(seed)
        truth = collect_truth_and_branches(
            weights,
            seed=seed,
            n_samples=args.truth_n,
            batch=args.batch,
        )
        print(f"seed={seed}: truth collection wall={truth.wall_time_s:.1f}s", flush=True)
        fits: dict[int, dict[str, Any]] = {}
        for k in KS:
            print(f"seed={seed} K={k}: fitting branch PCA/residual", flush=True)
            fits[k] = fit_shared_residual(truth.branches[k], rank=2)
            for g in (1, *GS):
                print(f"seed={seed} K={k} G={g}: constructing mixture", flush=True)
                mix = make_single_gaussian(truth.branches[k], fits[k]) if g == 1 else make_r1_grid(truth.branches[k], g, fits[k])
                print(f"seed={seed} K={k} G={g}: propagating {len(mix.weights)} nodes", flush=True)
                prop = propagate_mixture(mix, weights, start_layer=k, truth_means=truth.means)
                cases.append(
                    {
                        "seed": seed,
                        "K": k,
                        "G": g,
                        "rank": 1,
                        "final_mse": prop["final_mse"],
                        "final_rmse": prop["final_rmse"],
                        "final_max_abs": prop["final_max_abs"],
                        "wall_time_s": prop["wall_time_s"],
                        "construction": mix.construction,
                        "layer_diagnostics": prop["layer_diagnostics"],
                    }
                )
        if not args.skip_r2:
            k = 24
            print(f"seed={seed} K={k}: optional r=2 9x9 grid", flush=True)
            mix2 = make_r2_grid(truth.branches[k], 9, fits[k])
            prop2 = propagate_mixture(mix2, weights, start_layer=k, truth_means=truth.means)
            r2_rows.append(
                {
                    "seed": seed,
                    "K": k,
                    "grid": "9x9",
                    "nodes": int(len(mix2.weights)),
                    "final_mse": prop2["final_mse"],
                    "final_rmse": prop2["final_rmse"],
                    "wall_time_s": prop2["wall_time_s"],
                    "construction": mix2.construction,
                    "layer_diagnostics": prop2["layer_diagnostics"],
                }
            )
        del truth

    convergence = {}
    for k in KS:
        averaged = []
        for g in GS:
            vals = [row["final_mse"] for row in cases if row["K"] == k and row["G"] == g and row["rank"] == 1]
            averaged.append({"G": g, "final_mse": float(np.mean(vals))})
        convergence[str(k)] = fit_convergence(averaged)
    k24_g65 = float(np.mean([row["final_mse"] for row in cases if row["K"] == 24 and row["G"] == 65]))
    k24_p = convergence["24"]["floor_fit_grid"]["p"]
    got_verdict = verdict(k24_g65, k24_p)
    if got_verdict == "MACHINERY CONFIRMED":
        next_action = "Proceed to stage 2: deterministic initialization through the transition layers, keeping this grid propagation/readout as the isolated machinery target."
    elif got_verdict == "PARTIALLY CONFIRMED":
        next_action = "Localize the residual floor before stage 2: compare exact sampled node propagation on a few cells against the Gaussian closure, and test whether r=2 or a-dependent residual covariance lowers the floor."
    else:
        next_action = "Do not invest in estimator initialization yet. Use the per-layer diagnostics to isolate whether error enters immediately after K from node closure, accumulates smoothly from Gaussian covariance closure, or remains from r=1 truncation."

    results = {
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "seeds": list(SEEDS),
            "K": list(KS),
            "G": list(GS),
            "truth_n": args.truth_n,
            "truth_protocol": "antithetic normal inputs; same full sample set used for truth means and near-perfect branch initialization",
            "batch": args.batch,
            "skip_r2": args.skip_r2,
        },
        "cases": cases,
        "convergence": convergence,
        "r2_optional": r2_rows,
        "summary": {
            "k24_g65_mean_mse": k24_g65,
            "k24_floor_fit_p": k24_p,
            "verdict": got_verdict,
            "recommended_next_action": next_action,
            "wall_time_s": time.time() - started,
        },
    }
    JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results)
    return results


def print_summary(results: dict[str, Any]) -> None:
    print("\nBias-MSE table (mean across seeds)")
    print("K   G      mean_final_mse")
    for k in KS:
        for g in (1, *GS):
            vals = [row["final_mse"] for row in results["cases"] if row["K"] == k and row["G"] == g]
            print(f"{k:<3} {g:<5} {np.mean(vals):.6e}")
    print("\nConvergence")
    for k in KS:
        conv = results["convergence"][str(k)]
        print(
            f"K={k}: raw_p={conv['raw_loglog_p']:.3f} "
            f"floor_fit_p={conv['floor_fit_grid']['p']:.3f} "
            f"floor={conv['floor_fit_grid']['floor']:.3e} "
            f"largestG={conv['largest_G_mse']:.3e}"
        )
    print(
        f"\nVerdict: {results['summary']['verdict']} "
        f"(K=24,G=65 mean={results['summary']['k24_g65_mean_mse']:.3e}, "
        f"p={results['summary']['k24_floor_fit_p']:.3f})"
    )
    print(f"Recommended next action: {results['summary']['recommended_next_action']}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-n", type=int, default=TRUTH_N)
    parser.add_argument("--batch", type=int, default=TRUTH_BATCH)
    parser.add_argument("--skip-r2", action="store_true")
    args = parser.parse_args()
    results = run(args)
    print_summary(results)


if __name__ == "__main__":
    main()
