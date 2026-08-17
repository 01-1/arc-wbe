from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_engine import build_mlp


WIDTH = 256
DEPTH = 32
SEEDS = (11, 22)
TARGET_LAYERS = (2, 4, 8)
OUTDIR = REPO / "paired_fly_logs" / "fingerprint_theory"
JSON_PATH = OUTDIR / "k4_ladder_20260706_results.json"
MD_PATH = OUTDIR / "k4_ladder_20260706.md"

MIN_VARIANCE = 1e-12
MASS_FLOOR = 1e-14

ROUTES = {
    "A": {"label": "K=2 anchor", "track_k3": False, "track_k4": False},
    "B": {"label": "+kappa3 diagonal only", "track_k3": True, "track_k4": False},
    "C": {"label": "+kappa4 diagonal only", "track_k3": False, "track_k4": True},
    "D": {"label": "+kappa3 + kappa4", "track_k3": True, "track_k4": True},
}

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

GH64_X, GH64_W = np.polynomial.hermite.hermgauss(64)
STD_NORMAL_NODES = np.sqrt(2.0) * GH64_X.astype(np.float64)
STD_NORMAL_WEIGHTS = (GH64_W / math.sqrt(math.pi)).astype(np.float64)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def mlp_weights(seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    weights32 = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    weights64 = [w.astype(np.float64) for w in weights32]
    return weights32, weights64


class RawMomentAccumulator:
    def __init__(self, width: int) -> None:
        self.n = 0
        self.s1 = np.zeros(width, dtype=np.float64)
        self.s2 = np.zeros(width, dtype=np.float64)
        self.s3 = np.zeros(width, dtype=np.float64)
        self.s4 = np.zeros(width, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        vals = x.astype(np.float64, copy=False)
        v2 = vals * vals
        self.n += vals.shape[0]
        self.s1 += vals.sum(axis=0)
        self.s2 += v2.sum(axis=0)
        self.s3 += (v2 * vals).sum(axis=0)
        self.s4 += (v2 * v2).sum(axis=0)

    def finish(self) -> dict[str, Any]:
        if self.n == 0:
            raise ValueError("empty moment accumulator")
        m1 = self.s1 / self.n
        raw2 = self.s2 / self.n
        raw3 = self.s3 / self.n
        raw4 = self.s4 / self.n
        var = np.maximum(raw2 - m1 * m1, 0.0)
        k3 = raw3 - 3.0 * m1 * raw2 + 2.0 * m1**3
        central4 = raw4 - 4.0 * m1 * raw3 + 6.0 * m1 * m1 * raw2 - 3.0 * m1**4
        k4 = central4 - 3.0 * var * var
        return {
            "n": self.n,
            "mean": m1.tolist(),
            "variance": var.tolist(),
            "kappa3": k3.tolist(),
            "kappa4": k4.tolist(),
        }


def antithetic_truth_with_preactivation_cumulants(
    weights: list[np.ndarray],
    *,
    n_samples: int,
    seed: int,
    batch_pairs: int,
    cumulant_samples: int,
) -> dict[str, Any]:
    if n_samples % 2:
        raise ValueError("n_samples must be even")
    if cumulant_samples % 2:
        raise ValueError("cumulant_samples must be even")
    if cumulant_samples > n_samples:
        raise ValueError("cumulant_samples cannot exceed n_samples")

    n_pairs = n_samples // 2
    cumulant_pairs = cumulant_samples // 2
    rng = np.random.default_rng(seed)
    total = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    total2 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    pre_acc = {layer: RawMomentAccumulator(WIDTH) for layer in TARGET_LAYERS}
    done_pairs = 0
    started = time.time()

    while done_pairs < n_pairs:
        b = min(batch_pairs, n_pairs - done_pairs)
        x0 = rng.standard_normal((b, WIDTH)).astype(np.float32)
        x = np.concatenate((x0, -x0), axis=0)
        take_cumulant_pairs = max(0, min(b, cumulant_pairs - done_pairs))

        for layer_idx, w in enumerate(weights, start=1):
            pre = x @ w
            if layer_idx in pre_acc and take_cumulant_pairs:
                vals = np.concatenate(
                    (pre[:take_cumulant_pairs], pre[b : b + take_cumulant_pairs]),
                    axis=0,
                )
                pre_acc[layer_idx].update(vals)
            x = np.maximum(pre, 0.0)
            pair = 0.5 * (x[:b] + x[b:])
            pair64 = pair.astype(np.float64, copy=False)
            total[layer_idx - 1] += pair64.sum(axis=0)
            total2[layer_idx - 1] += (pair64 * pair64).sum(axis=0)

        done_pairs += b

    mean = total / n_pairs
    var_pair = np.maximum(total2 / n_pairs - mean * mean, 0.0)
    noise = var_pair.mean(axis=1) / n_pairs
    return {
        "mean": mean,
        "truth_noise_mse_by_layer": noise,
        "truth_noise_final_mse": float(noise[-1]),
        "truth_noise_all_layer_mse": float(noise.mean()),
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "cumulant_samples": cumulant_samples,
        "cumulant_pairs": cumulant_pairs,
        "seed": seed,
        "batch_pairs": batch_pairs,
        "wall_time_s": time.time() - started,
        "empirical_preactivation_cumulants": {
            str(layer): acc.finish() for layer, acc in pre_acc.items()
        },
    }


def gaussian_relu_mean_cov(mean_pre: np.ndarray, cov_pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    cov = second - relu_mean[:, None] * relu_mean[None, :]
    cov = 0.5 * (cov + cov.T)
    np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE))
    return relu_mean, cov


def gaussian_relu_raw_moments(
    mean_pre: np.ndarray,
    var_pre: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    var = np.maximum(var_pre, MIN_VARIANCE)
    sigma = np.sqrt(var)
    beta = mean_pre / sigma
    threshold = -beta
    phi = norm_pdf(beta)
    tail = norm_cdf(beta)
    ints = [tail, phi]
    for order in range(2, 5):
        ints.append((threshold ** (order - 1)) * phi + (order - 1) * ints[order - 2])

    raw = []
    for power in range(1, 5):
        moment = np.zeros_like(mean_pre)
        for z_power in range(power + 1):
            moment += (
                math.comb(power, z_power)
                * mean_pre ** (power - z_power)
                * sigma**z_power
                * ints[z_power]
            )
        raw.append(moment)
    return raw[0], raw[1], raw[2], raw[3]


def edgeworth_relu_marginal_moments(
    mean_pre: np.ndarray,
    var_pre: np.ndarray,
    k3_pre: np.ndarray,
    k4_pre: np.ndarray,
    *,
    track_k3: bool,
    track_k4: bool,
    clip_stats: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    var = np.maximum(var_pre, MIN_VARIANCE)
    sigma = np.sqrt(var)
    lam3 = k3_pre / (var * sigma) if track_k3 else np.zeros_like(var)
    lam4 = k4_pre / (var * var) if track_k4 else np.zeros_like(var)

    t = STD_NORMAL_NODES[:, None]
    he3 = t**3 - 3.0 * t
    he4 = t**4 - 6.0 * t**2 + 3.0
    he6 = t**6 - 15.0 * t**4 + 45.0 * t**2 - 15.0
    correction = np.ones((STD_NORMAL_NODES.shape[0], mean_pre.shape[0]), dtype=np.float64)
    if track_k3:
        l3 = lam3[None, :]
        correction += (l3 / 6.0) * he3 + (l3 * l3 / 72.0) * he6
    if track_k4:
        correction += (lam4[None, :] / 24.0) * he4

    negative = correction < 0.0
    clip_stats["density_evals"] += int(correction.size)
    clip_stats["coordinate_layers"] += int(mean_pre.shape[0])
    if np.any(negative):
        clip_stats["negative_density_evals"] += int(np.count_nonzero(negative))
        clip_stats["coordinate_layers_with_clipping"] += int(np.count_nonzero(np.any(negative, axis=0)))

    correction = np.maximum(correction, 0.0)
    delta_weights = STD_NORMAL_WEIGHTS[:, None] * (correction - 1.0)
    mass = 1.0 + delta_weights.sum(axis=0)
    bad_mass = mass <= MASS_FLOOR
    if np.any(bad_mass):
        clip_stats["mass_floor_coordinates"] += int(np.count_nonzero(bad_mass))
        mass = np.where(bad_mass, MASS_FLOOR, mass)

    u = mean_pre[None, :] + sigma[None, :] * t
    relu = np.maximum(u, 0.0)
    base1, base2, base3, base4 = gaussian_relu_raw_moments(mean_pre, var)
    r1 = (base1 + (delta_weights * relu).sum(axis=0)) / mass
    r2 = (base2 + (delta_weights * relu**2).sum(axis=0)) / mass
    r3 = (base3 + (delta_weights * relu**3).sum(axis=0)) / mass
    r4 = (base4 + (delta_weights * relu**4).sum(axis=0)) / mass

    var_post = np.maximum(r2 - r1 * r1, MIN_VARIANCE)
    k3_post = r3 - 3.0 * r1 * r2 + 2.0 * r1**3
    central4 = r4 - 4.0 * r1 * r3 + 6.0 * r1 * r1 * r2 - 3.0 * r1**4
    k4_post = central4 - 3.0 * var_post * var_post
    if not track_k3:
        k3_post = np.zeros_like(k3_post)
    if not track_k4:
        k4_post = np.zeros_like(k4_post)
    return r1, var_post, k3_post, k4_post


def analytic_ladder(weights: list[np.ndarray], route: str) -> dict[str, Any]:
    cfg = ROUTES[route]
    m = np.zeros(WIDTH, dtype=np.float64)
    s = np.eye(WIDTH, dtype=np.float64)
    g3 = np.zeros(WIDTH, dtype=np.float64)
    g4 = np.zeros(WIDTH, dtype=np.float64)
    rows = []
    pre_cumulants: dict[str, dict[str, list[float]]] = {}
    clip_stats = {
        "density_evals": 0,
        "negative_density_evals": 0,
        "coordinate_layers": 0,
        "coordinate_layers_with_clipping": 0,
        "mass_floor_coordinates": 0,
    }
    started = time.time()

    for layer_idx, w in enumerate(weights, start=1):
        mz = m @ w
        sz = w.T @ s @ w
        sz = 0.5 * (sz + sz.T)
        g3z = (w**3).T @ g3 if cfg["track_k3"] else np.zeros(WIDTH, dtype=np.float64)
        g4z = (w**4).T @ g4 if cfg["track_k4"] else np.zeros(WIDTH, dtype=np.float64)
        if layer_idx in TARGET_LAYERS:
            pre_cumulants[str(layer_idx)] = {
                "kappa3": g3z.tolist(),
                "kappa4": g4z.tolist(),
            }

        _gauss_mean, gauss_cov = gaussian_relu_mean_cov(mz, sz)
        varz = np.maximum(np.diag(sz), MIN_VARIANCE)
        m, var_post, g3, g4 = edgeworth_relu_marginal_moments(
            mz,
            varz,
            g3z,
            g4z,
            track_k3=cfg["track_k3"],
            track_k4=cfg["track_k4"],
            clip_stats=clip_stats,
        )
        s = gauss_cov
        np.fill_diagonal(s, var_post)
        s = 0.5 * (s + s.T)
        rows.append(m.copy())

    clip_stats["negative_density_eval_fraction"] = (
        clip_stats["negative_density_evals"] / clip_stats["density_evals"]
        if clip_stats["density_evals"]
        else 0.0
    )
    clip_stats["coordinate_layer_clip_fraction"] = (
        clip_stats["coordinate_layers_with_clipping"] / clip_stats["coordinate_layers"]
        if clip_stats["coordinate_layers"]
        else 0.0
    )
    return {
        "route": route,
        "label": cfg["label"],
        "prediction": np.stack(rows, axis=0),
        "analytic_preactivation_cumulants": pre_cumulants,
        "clip_stats": clip_stats,
        "wall_time_s": time.time() - started,
    }


def metrics(pred: np.ndarray, truth: np.ndarray, noise_by_layer: np.ndarray) -> dict[str, Any]:
    layer_mse = ((pred.astype(np.float64) - truth.astype(np.float64)) ** 2).mean(axis=1)
    return {
        "final_layer_mse": float(layer_mse[-1]),
        "all_layer_mse": float(layer_mse.mean()),
        "net_bias_final_mse": float(layer_mse[-1] - noise_by_layer[-1]),
        "net_bias_all_layer_mse": float(layer_mse.mean() - noise_by_layer.mean()),
        "layer_mse": layer_mse.tolist(),
    }


def validation_stats(analytic: np.ndarray, empirical: np.ndarray) -> dict[str, float]:
    a = analytic.astype(np.float64)
    e = empirical.astype(np.float64)
    a0 = a - a.mean()
    e0 = e - e.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(e0))
    corr = float(np.dot(a0, e0) / denom) if denom > 0.0 else 0.0
    ee = float(np.dot(e, e))
    ls_scale = float(np.dot(a, e) / ee) if ee > 0.0 else 0.0
    rms_e = float(np.sqrt(np.mean(e * e)))
    rms_ratio = float(np.sqrt(np.mean(a * a)) / rms_e) if rms_e > 0.0 else 0.0
    return {"corr": corr, "ls_scale": ls_scale, "rms_ratio": rms_ratio}


def route_validation(
    analytic_run: dict[str, Any],
    empirical_cumulants: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    analytic_layers = analytic_run["analytic_preactivation_cumulants"]
    for layer in TARGET_LAYERS:
        key = str(layer)
        emp = empirical_cumulants[key]
        ana = analytic_layers[key]
        out[key] = {
            "kappa3": validation_stats(
                np.asarray(ana["kappa3"], dtype=np.float64),
                np.asarray(emp["kappa3"], dtype=np.float64),
            ),
            "kappa4": validation_stats(
                np.asarray(ana["kappa4"], dtype=np.float64),
                np.asarray(emp["kappa4"], dtype=np.float64),
            ),
        }
    return out


def serializable_truth(truth: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in truth.items()
        if k not in ("mean", "truth_noise_mse_by_layer")
    } | {"truth_noise_mse_by_layer": truth["truth_noise_mse_by_layer"].tolist()}


def serializable_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": run["route"],
        "label": run["label"],
        "clip_stats": run["clip_stats"],
        "wall_time_s": run["wall_time_s"],
        "analytic_preactivation_cumulants": run["analytic_preactivation_cumulants"],
    }


def pooled_metrics(
    route_preds: dict[str, list[np.ndarray]],
    truths: list[np.ndarray],
    noises: list[np.ndarray],
) -> dict[str, Any]:
    truth_stack = np.stack(truths)
    noise_stack = np.stack(noises)
    pooled: dict[str, Any] = {}
    for route, preds in route_preds.items():
        layer_mse = ((np.stack(preds) - truth_stack) ** 2).mean(axis=(0, 2))
        noise_by_layer = noise_stack.mean(axis=0)
        pooled[route] = {
            "final_layer_mse": float(layer_mse[-1]),
            "all_layer_mse": float(layer_mse.mean()),
            "truth_noise_final_mse": float(noise_by_layer[-1]),
            "truth_noise_all_layer_mse": float(noise_by_layer.mean()),
            "net_bias_final_mse": float(layer_mse[-1] - noise_by_layer[-1]),
            "net_bias_all_layer_mse": float(layer_mse.mean() - noise_by_layer.mean()),
            "layer_mse": layer_mse.tolist(),
        }
    return pooled


def verdict_from_pooled(pooled: dict[str, Any]) -> tuple[str, str]:
    a = pooled["A"]["net_bias_final_mse"]
    best = min(pooled["C"]["net_bias_final_mse"], pooled["D"]["net_bias_final_mse"])
    improvement = a / best if best > 0.0 else float("inf")
    if best <= 7e-6 and improvement >= 20.0:
        return (
            "MECHANISM FAMILY CONFIRMED",
            "Engineer a scorer-path version of the diagonal kappa4 route first; the result clears the finite-width mechanism gate.",
        )
    if improvement <= 5.0:
        return (
            "AMBIGUOUS-to-NEGATIVE",
            "Use the shallow cumulant validation to decide whether joint kappa4 structure is the next cost question or whether the Edgeworth marginal closure is the failure point.",
        )
    return (
        "PARTIAL / BELOW GATE",
        "The even-cumulant direction helped but did not clear the pre-registered 20x gate; inspect validation before spending scorer-path engineering effort.",
    )


def cost_accounting(width: int = WIDTH, depth: int = DEPTH) -> dict[str, Any]:
    n = width
    l = depth
    return {
        "dense_covariance_multiply_adds": int(2 * l * n**3),
        "mean_matvec_multiply_adds": int(l * n**2),
        "per_enabled_cumulant_matvec_multiply_adds": int(l * n**2),
        "gl16_pair_evaluations": int(l * 16 * n * n),
        "gh64_marginal_evaluations": int(l * 64 * n),
        "summary": (
            "One route is about 1.1e9 dense covariance multiply-adds plus "
            "O(32*16*256^2) GL16 pair-density evaluations and at most two "
            "extra 32*256^2 cumulant matvecs, i.e. low-1e9-class before "
            "implementation constants."
        ),
    }


def write_outputs(results: dict[str, Any]) -> None:
    JSON_PATH.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    pooled = results["pooled"]
    verdict = results["verdict"]
    lines = [
        "# Edgeworth-kappa4 analytic ladder (2026-07-06)",
        "",
        "Offline local MLPs from `local_engine.build_mlp(width=256, depth=32)`, seeds 11 and 22. MC truth used fresh antithetic `N(0,I)` samples. No Fly, network, pytest, tracked-file edits, public suites, or grader internals.",
        "",
        "Method notes: off-diagonal covariance uses the nonzero-mean Gaussian GL16 Price-identity bivariate ReLU closure ported from `estimator.py`/`gaussian_sum_pretest.py`; Edgeworth-corrected marginal variances replace only the diagonal. Marginal ReLU moments use exact Gaussian raw moments as a control variate, with 64-node Gauss-Hermite applied to the clipped and renormalized Edgeworth correction.",
        "",
        "## Verdict",
        "",
        f"- Verdict: **{verdict}**",
        f"- Route A pooled net final-layer bias-MSE: `{pooled['A']['net_bias_final_mse']:.9e}`",
        f"- Best of C/D pooled net final-layer bias-MSE: `{min(pooled['C']['net_bias_final_mse'], pooled['D']['net_bias_final_mse']):.9e}`",
        f"- C/D improvement over A: `{pooled['A']['net_bias_final_mse'] / max(min(pooled['C']['net_bias_final_mse'], pooled['D']['net_bias_final_mse']), 1e-300):.3f}x`",
        "",
        "## Ladder Bias",
        "",
        "| route | label | seed 11 net final MSE | seed 22 net final MSE | pooled net final MSE | pooled final MSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    per_seed = results["per_mlp"]
    for route, cfg in ROUTES.items():
        seed_vals = {
            row["seed"]: row["routes"][route]["metrics"]["net_bias_final_mse"] for row in per_seed
        }
        lines.append(
            f"| {route} | {cfg['label']} | "
            f"{seed_vals[11]:.9e} | {seed_vals[22]:.9e} | "
            f"{pooled[route]['net_bias_final_mse']:.9e} | {pooled[route]['final_layer_mse']:.9e} |"
        )

    lines += [
        "",
        "## Clipping",
        "",
        "| route | negative eval fraction | coord-layer clip fraction | mass-floor coords |",
        "|---|---:|---:|---:|",
    ]
    for route in ROUTES:
        stats = results["clip_summary"][route]
        lines.append(
            f"| {route} | {stats['negative_density_eval_fraction']:.6e} | "
            f"{stats['coordinate_layer_clip_fraction']:.6e} | {stats['mass_floor_coordinates']} |"
        )

    lines += [
        "",
        "## Cumulant Validation",
        "",
        "Correlation is across coordinates. Scale is the least-squares analytic/empirical ratio. This table uses route D, the full diagonal kappa3+kappa4 propagation; route-specific validation is in the JSON.",
        "",
        "| layer | k3 corr | k3 scale | k4 corr | k4 scale |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in TARGET_LAYERS:
        vals = results["pooled_validation"]["D"][str(layer)]
        lines.append(
            f"| {layer} | {vals['kappa3']['corr']:.4f} | {vals['kappa3']['ls_scale']:.4f} | "
            f"{vals['kappa4']['corr']:.4f} | {vals['kappa4']['ls_scale']:.4f} |"
        )

    lines += [
        "",
        "## Cost Accounting",
        "",
        results["cost_accounting"]["summary"],
        "",
        "## Recommended Next Action",
        "",
        results["recommended_next_action"],
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--cumulant-samples", type=int, default=400_000)
    ap.add_argument("--batch-pairs", type=int, default=4096)
    ap.add_argument("--truth-seed-base", type=int, default=30_000)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.cumulant_samples > args.samples:
        args.cumulant_samples = args.samples

    results: dict[str, Any] = {
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "seeds": list(SEEDS),
            "samples": args.samples,
            "cumulant_samples": args.cumulant_samples,
            "batch_pairs": args.batch_pairs,
            "truth_seed_base": args.truth_seed_base,
            "offdiagonal_covariance_closure": "nonzero-mean Gaussian GL16 Price-identity bivariate ReLU closure ported from estimator.py/gaussian_sum_pretest.py; Edgeworth variances replace only the diagonal.",
            "edgeworth_quadrature": "64-node Gauss-Hermite against clipped and renormalized diagonal Edgeworth correction, with the pure Gaussian raw-moment part evaluated exactly as a control variate.",
        },
        "per_mlp": [],
        "cost_accounting": cost_accounting(),
    }

    route_preds: dict[str, list[np.ndarray]] = {route: [] for route in ROUTES}
    truths: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    validation_by_route: dict[str, list[dict[str, Any]]] = {route: [] for route in ROUTES}
    clip_summary: dict[str, dict[str, int | float]] = {
        route: {
            "density_evals": 0,
            "negative_density_evals": 0,
            "coordinate_layers": 0,
            "coordinate_layers_with_clipping": 0,
            "mass_floor_coordinates": 0,
        }
        for route in ROUTES
    }

    for seed in SEEDS:
        print(f"[k4] seed {seed}: building MLP and MC truth ({args.samples} samples)", flush=True)
        weights32, weights64 = mlp_weights(seed)
        truth = antithetic_truth_with_preactivation_cumulants(
            weights32,
            n_samples=args.samples,
            seed=args.truth_seed_base + seed,
            batch_pairs=args.batch_pairs,
            cumulant_samples=args.cumulant_samples,
        )
        print(
            f"[k4] seed {seed}: truth done in {truth['wall_time_s']:.1f}s, noise final={truth['truth_noise_final_mse']:.3e}",
            flush=True,
        )

        row: dict[str, Any] = {
            "seed": seed,
            "truth": serializable_truth(truth),
            "routes": {},
        }
        truths.append(truth["mean"].astype(np.float64))
        noises.append(truth["truth_noise_mse_by_layer"].astype(np.float64))

        for route in ROUTES:
            print(f"[k4] seed {seed}: analytic route {route} ({ROUTES[route]['label']})", flush=True)
            run = analytic_ladder(weights64, route)
            m = metrics(run["prediction"], truth["mean"], truth["truth_noise_mse_by_layer"])
            route_preds[route].append(run["prediction"].astype(np.float64))
            row["routes"][route] = {
                "run": serializable_run(run),
                "metrics": m,
                "validation": route_validation(run, truth["empirical_preactivation_cumulants"]),
            }
            validation_by_route[route].append(row["routes"][route]["validation"])
            for key in (
                "density_evals",
                "negative_density_evals",
                "coordinate_layers",
                "coordinate_layers_with_clipping",
                "mass_floor_coordinates",
            ):
                clip_summary[route][key] = int(clip_summary[route][key]) + int(run["clip_stats"][key])
            print(
                f"[k4] seed {seed} route {route}: final={m['final_layer_mse']:.9e} net={m['net_bias_final_mse']:.9e} clip={run['clip_stats']['negative_density_eval_fraction']:.3e}",
                flush=True,
            )

        results["per_mlp"].append(row)

    for route, stats in clip_summary.items():
        stats["negative_density_eval_fraction"] = (
            stats["negative_density_evals"] / stats["density_evals"]
            if stats["density_evals"]
            else 0.0
        )
        stats["coordinate_layer_clip_fraction"] = (
            stats["coordinate_layers_with_clipping"] / stats["coordinate_layers"]
            if stats["coordinate_layers"]
            else 0.0
        )
    results["clip_summary"] = clip_summary
    results["pooled"] = pooled_metrics(route_preds, truths, noises)

    pooled_validation: dict[str, Any] = {}
    for route in ROUTES:
        pooled_validation[route] = {}
        for layer in TARGET_LAYERS:
            key = str(layer)
            pooled_validation[route][key] = {}
            for name in ("kappa3", "kappa4"):
                corr_vals = [v[key][name]["corr"] for v in validation_by_route[route]]
                scale_vals = [v[key][name]["ls_scale"] for v in validation_by_route[route]]
                rms_vals = [v[key][name]["rms_ratio"] for v in validation_by_route[route]]
                pooled_validation[route][key][name] = {
                    "corr": float(np.nanmean(corr_vals)),
                    "ls_scale": float(np.nanmean(scale_vals)),
                    "rms_ratio": float(np.nanmean(rms_vals)),
                }
    results["pooled_validation"] = pooled_validation

    verdict, rec = verdict_from_pooled(results["pooled"])
    if verdict == "AMBIGUOUS-to-NEGATIVE":
        d_k4 = results["pooled_validation"]["D"]["8"]["kappa4"]
        if abs(d_k4["ls_scale"]) < 0.1:
            rec = (
                "Do not scorer-engineer this diagonal one-loop kappa4 route. "
                "The layer-8 kappa4 validation is essentially gone "
                f"(corr={d_k4['corr']:.3f}, scale={d_k4['ls_scale']:.3f}), "
                "so the failure localizes to independence/diagonal cumulant propagation; "
                "only an oracle empirical-cumulant closure check or a joint-kappa4 design would be a meaningful next gate."
            )
    results["verdict"] = verdict
    results["recommended_next_action"] = rec

    write_outputs(results)

    print("\nA/B/C/D pooled net final-layer bias-MSE:")
    for route in ROUTES:
        print(
            f"  {route} {ROUTES[route]['label']}: "
            f"{results['pooled'][route]['net_bias_final_mse']:.9e}"
        )
    print("\nCumulant validation, route D pooled:")
    for layer in TARGET_LAYERS:
        vals = results["pooled_validation"]["D"][str(layer)]
        print(
            f"  layer {layer}: "
            f"k3 corr={vals['kappa3']['corr']:.4f} scale={vals['kappa3']['ls_scale']:.4f}; "
            f"k4 corr={vals['kappa4']['corr']:.4f} scale={vals['kappa4']['ls_scale']:.4f}"
        )
    print("\nClipping summary:")
    for route in ROUTES:
        stats = results["clip_summary"][route]
        print(
            f"  {route}: neg_eval_frac={stats['negative_density_eval_fraction']:.6e}, "
            f"coord_layer_frac={stats['coordinate_layer_clip_fraction']:.6e}, "
            f"mass_floor={stats['mass_floor_coordinates']}"
        )
    print(f"\nVerdict: {verdict}")
    print(f"Recommended next action: {rec}")
    print(f"JSON: {JSON_PATH}")
    print(f"Markdown: {MD_PATH}")


if __name__ == "__main__":
    main()
