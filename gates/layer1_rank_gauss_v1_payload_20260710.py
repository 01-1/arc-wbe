#!/usr/bin/env python3
"""Truth-bank payload for exact first-layer Gaussian marginal rank transport."""

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

import flopscope.numpy as fnp
from estimator import (
    _DEEP_VARIANCE_MATCH_STRENGTH,
    _MIN_VARIANCE,
    _gaussian_relu_variance,
    _hadamard,
    _strassen_matmul,
    _zero_mean_relu_mean_cov,
)
from local_engine import build_mlp
from paired_fly_logs.fingerprint_theory.sobol_runtime_feasibility_generator_20260710 import (
    ndtri_dependency_free,
)


WIDTH = 256
DEPTH = 32
POSITIVE_ROWS = 4096
TOTAL_ROWS = 8192
BLOCKS = 16
REPS = 1
HADAMARD_STREAM = 0x484144_0710
SCRIPT_VERSION = "layer1-rank-gauss-v1"

# This is MLP-independent and is made once from the reviewed dependency-free
# Acklam implementation.  Symmetric assignment ensures exact pair signs even
# when the dependency-free inverse has tiny asymmetric roundoff.
_q_low = ndtri_dependency_free(
    (np.arange(TOTAL_ROWS // 2, dtype=np.float64) + 0.5) / float(TOTAL_ROWS)
)
QUANTILE_TABLE = np.concatenate((_q_low, -_q_low[::-1])).astype(np.float32)


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int, rep: int) -> int:
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _hadamard_rows(seed: int, width: int, blocks: int) -> fnp.ndarray:
    """Sixteen independent randomized positive Hadamard half-bases."""
    if width != WIDTH or blocks != BLOCKS:
        raise ValueError("this gate has fixed width 256 and 16 blocks")
    rng = fnp.random.default_rng(seed)
    base = _hadamard(width).astype(fnp.float32)
    pieces = []
    for _ in range(blocks):
        flips = (2.0 * rng.integers(0, 2, size=width) - 1.0).astype(fnp.float32)
        pieces.append(base * flips[None, :])
    return fnp.concatenate(tuple(pieces), axis=0).astype(fnp.float32)


def _stats(values: fnp.ndarray) -> dict[str, float]:
    """Reduce a flopscope vector; only scalar results cross the boundary."""
    x = values.astype(fnp.float64)
    ordered = fnp.sort(x)
    middle = (ordered[x.shape[0] // 2 - 1] + ordered[x.shape[0] // 2]) * 0.5
    return {
        "min": float(np.asarray(fnp.min(x))),
        "median": float(np.asarray(middle)),
        "max": float(np.asarray(fnp.max(x))),
        "rms": float(np.asarray(fnp.sqrt(fnp.mean(x * x)))),
    }


def _pre_diagnostics(
    pre: fnp.ndarray,
    transported: fnp.ndarray,
    sigma: fnp.ndarray,
    magnitude_order: fnp.ndarray,
) -> dict[str, object]:
    """Diagnostics are computed before truth is read and after rank inversion."""
    n = pre.shape[0]
    pair_error_raw = fnp.max(fnp.abs(pre[n // 2 :] + pre[: n // 2]))
    pair_error_transport = fnp.max(
        fnp.abs(transported[n // 2 :] + transported[: n // 2])
    )
    positive_pre = pre[: POSITIVE_ROWS]
    magnitudes = fnp.abs(positive_pre)
    sorted_magnitudes = fnp.take_along_axis(magnitudes, magnitude_order, axis=0)
    inverse_order = fnp.argsort(magnitude_order, axis=0, stable=True)
    q = fnp.array(QUANTILE_TABLE, dtype=fnp.float32)
    transported_sorted = q[:, None] * sigma[None, :]
    positive_rank_roundtrip = fnp.take_along_axis(
        magnitude_order, inverse_order, axis=0
    )
    rank_expected = fnp.arange(POSITIVE_ROWS, dtype=fnp.int64)[:, None]
    sorted_transport = fnp.sort(transported, axis=0)
    magnitude_tie_count = fnp.sum(
        sorted_magnitudes[1:] == sorted_magnitudes[:-1]
    )
    zero_value_count = fnp.sum(magnitudes == 0.0)

    sigma2 = sigma * sigma
    transported_mean = fnp.mean(transported, axis=0)
    transported_var = fnp.mean(
        (transported - transported_mean[None, :])
        * (transported - transported_mean[None, :]),
        axis=0,
    )
    mean_rel = transported_mean / fnp.maximum(sigma, _MIN_VARIANCE)
    var_rel = transported_var / fnp.maximum(sigma2, _MIN_VARIANCE) - 1.0
    return {
        "raw_antipode_max_abs": float(np.asarray(pair_error_raw)),
        "transported_antipode_max_abs": float(np.asarray(pair_error_transport)),
        "raw_magnitude_tie_count": int(np.asarray(magnitude_tie_count)),
        "raw_magnitude_tie_fraction": float(
            np.asarray(
                magnitude_tie_count
                / float((POSITIVE_ROWS - 1) * WIDTH)
            )
        ),
        "zero_value_count": int(np.asarray(zero_value_count)),
        "magnitude_rank_roundtrip_max_abs": float(
            np.asarray(
                fnp.max(fnp.abs(positive_rank_roundtrip - rank_expected))
            )
        ),
        "transport_sorted_target_max_abs": float(
            np.asarray(fnp.max(fnp.abs(sorted_transport - transported_sorted)))
        ),
        "transported_mean_relative": _stats(mean_rel),
        "transported_variance_relative_error": _stats(var_rel),
    }


def _relu_diagnostics(
    weights: list[fnp.ndarray],
    pre: fnp.ndarray,
    transported: fnp.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    target_second = fnp.diag(target_cov) + target_mean * target_mean
    results: dict[str, object] = {}
    recolored: dict[str, fnp.ndarray] = {}
    for name, values in (("current", pre), ("rank_gaussian", transported)):
        y = fnp.maximum(values, 0.0)
        mean = fnp.mean(y, axis=0).astype(fnp.float64)
        second = fnp.mean(y * y, axis=0).astype(fnp.float64)
        centered = y - mean[None, :]
        cov = (
            _strassen_matmul(
                centered.astype(fnp.float32).T,
                centered.astype(fnp.float32),
                3,
            )
            / float(y.shape[0])
        ).astype(fnp.float64)
        mean_error = mean - target_mean
        second_error = second - target_second
        cov_error = cov - target_cov
        jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
        eye = fnp.eye(weights[0].shape[1])
        sample_chol = fnp.linalg.cholesky(cov + jitter * eye)
        target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
        recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
        recolored[name] = _strassen_matmul(
            centered.astype(fnp.float32), recolor.astype(fnp.float32), 3
        ) + target_mean.astype(fnp.float32)[None, :]
        results[name] = {
            "pre_recolor_relu_mean_error": _stats(mean_error),
            "pre_recolor_relu_second_moment_error": _stats(second_error),
            "sample_relu_covariance_relative_fro": float(
                np.asarray(fnp.linalg.norm(cov_error, ord="fro")
                / fnp.maximum(fnp.linalg.norm(target_cov, ord="fro"), _MIN_VARIANCE))
            ),
            "recolor_frobenius_from_identity": float(
                np.asarray(fnp.linalg.norm(recolor - fnp.eye(WIDTH), ord="fro"))
            ),
        }
    return results, recolored


def _propagate(weights: list[fnp.ndarray], recolored: fnp.ndarray) -> np.ndarray:
    x = recolored
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(
                pre_mean,
                fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
            )
            sample_mean_1 = fnp.mean(x, axis=0).astype(fnp.float64)
            centered_layer = x - sample_mean_1[None, :]
            sample_var = fnp.maximum(
                fnp.mean(centered_layer * centered_layer, axis=0).astype(fnp.float64),
                _MIN_VARIANCE,
            )
            centered_apply = x - sample_mean_1.astype(fnp.float32)[None, :]
            scale = (
                1.0
                + _DEEP_VARIANCE_MATCH_STRENGTH
                * (fnp.sqrt(target_var / sample_var) - 1.0)
            ).astype(fnp.float32)
            x = centered_apply * scale[None, :] + sample_mean_1.astype(fnp.float32)[None, :]
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def _estimate_from_shared_pre(
    weights: list[fnp.ndarray], pre: fnp.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    gram = weights[0].T @ weights[0]
    sigma = fnp.sqrt(fnp.diag(gram)).astype(fnp.float32)
    positive_pre = pre[:POSITIVE_ROWS]
    magnitudes = fnp.abs(positive_pre)
    magnitude_order = fnp.argsort(magnitudes, axis=0, stable=True)
    inverse_magnitude_order = fnp.argsort(magnitude_order, axis=0, stable=True)
    q_positive = fnp.array(QUANTILE_TABLE[POSITIVE_ROWS:], dtype=fnp.float32)
    sorted_target = q_positive[:, None] * sigma[None, :]
    positive_ranked = fnp.take_along_axis(
        sorted_target, inverse_magnitude_order, axis=0
    )
    sign = fnp.where(positive_pre < 0.0, -1.0, 1.0).astype(fnp.float32)
    transported_positive = positive_ranked * sign
    transported = fnp.concatenate(
        (transported_positive, -transported_positive), axis=0
    )
    diagnostics = _pre_diagnostics(pre, transported, sigma, magnitude_order)
    relu_diag, recolored = _relu_diagnostics(weights, pre, transported)
    diagnostics["relu"] = relu_diag
    estimates = {
        "current": _propagate(weights, recolored["current"]),
        "rank_gaussian": _propagate(weights, recolored["rank_gaussian"]),
    }
    return estimates, diagnostics


def run_one(shard_index: int, bank_path: Path) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")
    weights_f32 = [fnp.array(w, dtype=fnp.float32) for w in weights]
    estimated_reps: list[dict[str, object]] = []
    for rep in range(REPS):
        rows = _hadamard_rows(
            _derived_seed(seed, HADAMARD_STREAM, rep), WIDTH, BLOCKS
        )
        pre_half = _strassen_matmul(rows, weights_f32[0], 3)
        # Both arms below consume this one shared raw preactivation tensor.
        pre = fnp.concatenate((pre_half, -pre_half), axis=0)
        estimates, diagnostics = _estimate_from_shared_pre(weights_f32, pre)
        estimated_reps.append(
            {"rep": rep, "estimates": estimates, "diagnostics": diagnostics}
        )

    # Truth is intentionally read only after all vectors and diagnostics are fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    reps = []
    for item in estimated_reps:
        estimates = item["estimates"]
        reps.append(
            {
                "rep": item["rep"],
                "mse": {
                    name: float(np.mean((value - truth) ** 2))
                    for name, value in estimates.items()
                },
                "estimates": {
                    name: value.tolist() for name, value in estimates.items()
                },
                "diagnostics": item["diagnostics"],
            }
        )
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
            "positive_rows": POSITIVE_ROWS,
            "total_rows": TOTAL_ROWS,
            "blocks": BLOCKS,
            "reps": REPS,
            "hadamard_bases": "16 independent randomized positive half-bases; exact antipodes",
            "rank_transport": "stable positive-half magnitude ranking, original-sign restoration, exact negative append",
            "quantile_source": "reviewed dependency-free ndtri; symmetric fixed table",
            "finite_fp32_tie_policy": "stable positive-half magnitude ranks, original signs, exact negative half; zero magnitudes use positive sign",
            "tie_diagnostics": "magnitude tie fraction, zero count, exact full sorted target audit, exact antipode audit",
            "first_layer": "exact global ReLU mean/covariance recolor for both arms",
            "first_successor": "fp64 statistics with fp32 centered_apply/scale/writeback, strength 1.5",
            "propagation": "fp32 L3 Strassen",
            "row_normalization": "none",
        },
        "truth_final": truth.tolist(),
        "reps": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank)
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
