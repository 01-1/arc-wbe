#!/usr/bin/env python3
"""Fly payload for the positive-homogeneity angular-importance gate."""

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

from local_engine import build_mlp


SCRIPT_VERSION = "angular-importance-gate-v1"
RIDGE = 1e-3
FIT_STEPS = 400
UNIFORM_FLOOR = 0.10
PILOT_PAIRS = 512
HOLDOUT_PAIRS = 2048
DIRECT_PAIRS = 1024
GATE_STREAM = 0xA691_5EED_2026_0709


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def _sphere(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    z = rng.standard_normal((n, d))
    norms = np.linalg.norm(z, axis=1)
    while np.any(norms == 0.0):
        bad = norms == 0.0
        z[bad] = rng.standard_normal((int(np.sum(bad)), d))
        norms[bad] = np.linalg.norm(z[bad], axis=1)
    return z / norms[:, None]


def _folded_terminal(weights: list[np.ndarray], u: np.ndarray, radius: float) -> np.ndarray:
    n = u.shape[0]
    x0 = (radius * u).astype(np.float32)
    x = np.concatenate((x0, -x0), axis=0)
    for weight in weights:
        x = x @ weight
        np.maximum(x, 0.0, out=x)
    return 0.5 * (x[:n].astype(np.float64) + x[n:].astype(np.float64))


def _features(u: np.ndarray, w0: np.ndarray, radius: float) -> np.ndarray:
    return np.column_stack(
        (
            np.full(u.shape[0], radius, dtype=np.float64),
            np.abs((radius * u) @ w0.astype(np.float64)),
        )
    )


def _fit_nonnegative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    x_scale = np.sqrt(np.mean(x * x, axis=0))
    x_scale[x_scale == 0.0] = 1.0
    y_scale = max(float(np.mean(y)), 1e-30)
    xs = x / x_scale[None, :]
    ys = y / y_scale
    gram = (xs.T @ xs) / float(xs.shape[0])
    cross = (xs.T @ ys) / float(xs.shape[0])
    penalty = np.ones(xs.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    lipschitz = float(np.linalg.eigvalsh(gram + RIDGE * np.diag(penalty))[-1])
    beta = np.zeros(xs.shape[1], dtype=np.float64)
    step = 1.0 / max(lipschitz, 1e-30)
    for _ in range(FIT_STEPS):
        grad = gram @ beta - cross + RIDGE * penalty * beta
        beta = np.maximum(beta - step * grad, 0.0)
    coeff = beta * y_scale / x_scale
    pred = x @ coeff
    resid = y - pred
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(resid * resid)) / max(denom, 1e-30)
    return coeff, {
        "train_r2": r2,
        "active_slopes": int(np.sum(coeff[1:] > np.max(coeff) * 1e-10)),
        "iterations": FIT_STEPS,
        "ridge": RIDGE,
    }


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _normalizers(width: int) -> tuple[float, float]:
    mean_radius = math.exp(
        0.5 * math.log(2.0)
        + math.lgamma((width + 1.0) / 2.0)
        - math.lgamma(width / 2.0)
    )
    sphere_abs_mean = math.sqrt(2.0 / math.pi) / mean_radius
    return mean_radius, sphere_abs_mean


def _normalized_coefficients(
    coeff: np.ndarray,
    column_norms: np.ndarray,
    sphere_abs_mean: float,
    uniform_floor: float,
) -> tuple[np.ndarray, float, float]:
    angular_normalizer = float(
        coeff[0] + sphere_abs_mean * np.dot(coeff[1:], column_norms)
    )
    if not np.isfinite(angular_normalizer) or angular_normalizer <= 0.0:
        fallback = np.zeros_like(coeff)
        fallback[0] = 1.0
        return fallback, 1.0, angular_normalizer
    normalized = coeff / angular_normalizer
    safe = (1.0 - uniform_floor) * normalized
    safe[0] += uniform_floor
    safe_normalizer = float(
        safe[0] + sphere_abs_mean * np.dot(safe[1:], column_norms)
    )
    return safe, safe_normalizer, angular_normalizer


def _ratio(
    u: np.ndarray,
    coeff: np.ndarray,
    w0: np.ndarray,
    sphere_abs_mean: float,
    column_norms: np.ndarray,
) -> np.ndarray:
    normalizer = float(coeff[0] + sphere_abs_mean * np.dot(coeff[1:], column_norms))
    score = coeff[0] + np.abs(u @ w0.astype(np.float64)) @ coeff[1:]
    return score / normalizer


def _sample_proposal(
    rng: np.random.Generator,
    n: int,
    coeff: np.ndarray,
    w0: np.ndarray,
    sphere_abs_mean: float,
    column_norms: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    width = w0.shape[0]
    masses = np.concatenate(
        (
            np.array([coeff[0]], dtype=np.float64),
            sphere_abs_mean * coeff[1:] * column_norms,
        )
    )
    masses = np.maximum(masses, 0.0)
    masses /= np.sum(masses)
    component = rng.choice(width + 1, size=n, p=masses)
    u = np.empty((n, width), dtype=np.float64)
    uniform = component == 0
    if np.any(uniform):
        u[uniform] = _sphere(rng, int(np.sum(uniform)), width)
    for selected in np.unique(component[~uniform]):
        mask = component == selected
        count = int(np.sum(mask))
        idx = int(selected - 1)
        e = w0[:, idx].astype(np.float64) / column_norms[idx]
        z = rng.standard_normal((count, width))
        z -= (z @ e)[:, None] * e[None, :]
        z_norm = np.linalg.norm(z, axis=1)
        while np.any(z_norm <= 1e-14):
            bad = z_norm <= 1e-14
            z[bad] = rng.standard_normal((int(np.sum(bad)), width))
            z[bad] -= (z[bad] @ e)[:, None] * e[None, :]
            z_norm[bad] = np.linalg.norm(z[bad], axis=1)
        v = z / z_norm[:, None]
        y = rng.beta(1.0, (width - 1.0) / 2.0, size=count)
        sign = rng.choice(np.array([-1.0, 1.0]), size=count)
        u[mask] = sign[:, None] * np.sqrt(y)[:, None] * e[None, :] + np.sqrt(
            1.0 - y
        )[:, None] * v
    return u, {
        "mixture_mass_sum": float(np.sum(masses)),
        "uniform_mass": float(masses[0]),
        "active_folded_components": int(np.sum(masses[1:] > 1e-12)),
        "largest_component_mass": float(np.max(masses)),
    }


def _identity_metrics(h: np.ndarray, ratio: np.ndarray) -> dict[str, float]:
    n = h.shape[0]
    split = n // 2
    mean_sq = float(np.dot(np.mean(h[:split], axis=0), np.mean(h[split:], axis=0)))
    norm_sq = np.sum(h * h, axis=1)
    baseline_second = float(np.mean(norm_sq))
    proposal_second = float(np.mean(norm_sq / ratio))
    norm_mean = float(np.mean(np.sqrt(norm_sq)))
    baseline_var = baseline_second - mean_sq
    proposal_var = proposal_second - mean_sq
    oracle_var = norm_mean * norm_mean - mean_sq
    return {
        "mean_sq_crossfit": mean_sq,
        "baseline_second_moment": baseline_second,
        "proposal_second_moment": proposal_second,
        "baseline_variance": baseline_var,
        "proposal_variance": proposal_var,
        "proposal_ratio": baseline_var / proposal_var if proposal_var > 0.0 else 0.0,
        "projected_total_fraction": (
            0.125 + 0.875 * proposal_var / baseline_var if baseline_var > 0.0 else math.inf
        ),
        "oracle_variance": oracle_var,
        "oracle_ratio": baseline_var / oracle_var if oracle_var > 0.0 else 0.0,
    }


def run_one(
    shard_index: int,
    bank_path: Path,
    metadata_path: Path,
    pilot_pairs: int,
    holdout_pairs: int,
    direct_pairs: int,
) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = str(metadata["rows"][shard_index]["weights_sha256"])
    mlp = build_mlp(256, 32, seed)
    weights = [np.ascontiguousarray(np.asarray(w), dtype=np.float32) for w in mlp.weights]
    actual_hash = _weights_sha256(weights)
    if actual_hash != expected_hash:
        raise RuntimeError(f"weight checksum mismatch at shard {shard_index}")

    width = 256
    radius = math.sqrt(width)
    mean_radius, sphere_abs_mean = _normalizers(width)
    w0 = weights[0]
    column_norms = np.linalg.norm(w0.astype(np.float64), axis=0)
    rng = np.random.default_rng((seed ^ GATE_STREAM) % (1 << 64))

    pilot_u = _sphere(rng, pilot_pairs, width)
    pilot_g = _folded_terminal(weights, pilot_u, radius)
    pilot_target = np.linalg.norm(pilot_g, axis=1)
    pilot_x = _features(pilot_u, w0, radius)
    fitted_coeff, fit = _fit_nonnegative(pilot_x, pilot_target)
    safe_coeff, safe_normalizer, raw_normalizer = _normalized_coefficients(
        fitted_coeff, column_norms, sphere_abs_mean, UNIFORM_FLOOR
    )

    holdout_u = _sphere(rng, holdout_pairs, width)
    holdout_g = _folded_terminal(weights, holdout_u, radius)
    holdout_h = (mean_radius / radius) * holdout_g
    holdout_target = np.linalg.norm(holdout_g, axis=1)
    raw_ratio = _ratio(
        holdout_u, fitted_coeff, w0, sphere_abs_mean, column_norms
    )
    safe_ratio = _ratio(holdout_u, safe_coeff, w0, sphere_abs_mean, column_norms)
    raw_metrics = _identity_metrics(holdout_h, raw_ratio)
    primary = _identity_metrics(holdout_h, safe_ratio)

    holdout_pred = _features(holdout_u, w0, radius) @ fitted_coeff
    fit.update(
        {
            "holdout_pearson": _corr(holdout_pred, holdout_target),
            "holdout_spearman": _corr(
                _rankdata(holdout_pred), _rankdata(holdout_target)
            ),
        }
    )

    direct_u, mixture = _sample_proposal(
        rng, direct_pairs, safe_coeff, w0, sphere_abs_mean, column_norms
    )
    direct_g = _folded_terminal(weights, direct_u, radius)
    direct_h = (mean_radius / radius) * direct_g
    direct_ratio = _ratio(
        direct_u, safe_coeff, w0, sphere_abs_mean, column_norms
    )
    direct_y = direct_h / direct_ratio[:, None]
    baseline_var_direct = float(np.sum(np.var(holdout_h, axis=0, ddof=1)))
    proposal_var_direct = float(np.sum(np.var(direct_y, axis=0, ddof=1)))
    direct_second = float(np.mean(np.sum(direct_y * direct_y, axis=1)))
    baseline_mean = np.mean(holdout_h, axis=0)
    direct_mean = np.mean(direct_y, axis=0)
    baseline_var_coord = np.var(holdout_h, axis=0, ddof=1)
    direct_var_coord = np.var(direct_y, axis=0, ddof=1)
    se_sq = baseline_var_coord / holdout_pairs + direct_var_coord / direct_pairs
    live = se_sq > 1e-30
    mean_stat = float(np.mean((direct_mean[live] - baseline_mean[live]) ** 2 / se_sq[live]))
    importance_weight = 1.0 / direct_ratio
    ess_fraction = float(
        np.sum(importance_weight) ** 2
        / (direct_pairs * np.sum(importance_weight * importance_weight))
    )

    return {
        "ok": True,
        "script": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "weights_sha256": actual_hash,
        "checksum_ok": True,
        "width": width,
        "depth": 32,
        "pilot_pairs": pilot_pairs,
        "holdout_pairs": holdout_pairs,
        "direct_pairs": direct_pairs,
        "pilot_fraction": 0.125,
        "radius": radius,
        "mean_chi_radius": mean_radius,
        "sphere_abs_mean": sphere_abs_mean,
        "uniform_floor": UNIFORM_FLOOR,
        "normalizer": {
            "raw_angular": raw_normalizer,
            "safe_angular": safe_normalizer,
            "safe_gaussian": mean_radius * safe_normalizer,
            "heldout_raw_ratio_mean": float(np.mean(raw_ratio)),
            "heldout_safe_ratio_mean": float(np.mean(safe_ratio)),
            "heldout_safe_ratio_min": float(np.min(safe_ratio)),
        },
        "fit": fit,
        "primary": primary,
        "raw": raw_metrics,
        "mixture": mixture,
        "direct": {
            "baseline_variance": baseline_var_direct,
            "proposal_variance": proposal_var_direct,
            "proposal_ratio": (
                baseline_var_direct / proposal_var_direct if proposal_var_direct > 0.0 else 0.0
            ),
            "proposal_second_moment": direct_second,
            "identity_second_moment_ratio": (
                direct_second / primary["proposal_second_moment"]
                if primary["proposal_second_moment"] > 0.0
                else 0.0
            ),
            "identity_variance_ratio": (
                proposal_var_direct / primary["proposal_variance"]
                if primary["proposal_variance"] > 0.0
                else 0.0
            ),
            "mean_difference_stat": mean_stat,
            "mean_relative_l2": float(
                np.linalg.norm(direct_mean - baseline_mean)
                / max(np.linalg.norm(baseline_mean), 1e-30)
            ),
            "weight_min": float(np.min(importance_weight)),
            "weight_median": float(np.median(importance_weight)),
            "weight_q99": float(np.quantile(importance_weight, 0.99)),
            "weight_max": float(np.max(importance_weight)),
            "ess_fraction": ess_fraction,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0"))
    )
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("analysis/truth_bank/metadata.json"))
    parser.add_argument("--pilot-pairs", type=int, default=PILOT_PAIRS)
    parser.add_argument("--holdout-pairs", type=int, default=HOLDOUT_PAIRS)
    parser.add_argument("--direct-pairs", type=int, default=DIRECT_PAIRS)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    payload = run_one(
        args.shard_index,
        args.bank,
        args.metadata,
        args.pilot_pairs,
        args.holdout_pairs,
        args.direct_pairs,
    )
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
