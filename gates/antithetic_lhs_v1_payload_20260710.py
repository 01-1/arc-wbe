#!/usr/bin/env python3
"""Truth-bank Fly payload for antithetic Gaussian Latin-hypercube v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Fly invokes the manifest command from the repository root while preserving
# this nested archive path.  Make both root and generator imports explicit.
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
from paired_fly_logs.fingerprint_theory.sobol_runtime_feasibility_generator_20260710 import ndtri_dependency_free


WIDTH = 256
DEPTH = 32
POSITIVE_ROWS = 4096
TOTAL_ROWS = 8192
BLOCKS = 16
REPS = 1
UNIFORM_LO = 2.0 ** -53
UNIFORM_HI = 1.0 - UNIFORM_LO
LHS_STREAM = 0x4C4853_0710
IID_STREAM = 0x1D5A_0710
JITTER_LO = 2.0 ** -53
JITTER_HI = 1.0 - JITTER_LO
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "antithetic-gaussian-lhs-v1"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int, rep: int) -> int:
    """Keep the original gate's stable 32-bit derived seeds exactly."""
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _lhs_rows(seed: int, width: int) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if width != WIDTH:
        raise ValueError(f"fixed LHS requires width {WIDTH}, got {width}")
    rng = np.random.default_rng(seed)
    positive = np.empty((POSITIVE_ROWS, width), dtype=np.float32)
    low_strata = np.empty((POSITIVE_ROWS, width), dtype=np.int64)
    for j in range(width):
        pair_ids = rng.permutation(POSITIVE_ROWS)
        jitter = np.clip(rng.random(POSITIVE_ROWS), JITTER_LO, JITTER_HI)
        u_low = (pair_ids.astype(np.float64) + jitter) / float(TOTAL_ROWS)
        z_low = ndtri_dependency_free(u_low)
        orientation = rng.integers(0, 2, size=POSITIVE_ROWS, dtype=np.int8)
        positive[:, j] = np.where(orientation == 0, z_low, -z_low).astype(np.float32)
        low_strata[:, j] = pair_ids
    full = np.concatenate((positive, -positive), axis=0).astype(np.float64)
    strata_ok = True
    for j in range(width):
        # The exact audit uses the generated pair IDs and their complementary strata.
        covered = np.concatenate((low_strata[:, j], TOTAL_ROWS - 1 - low_strata[:, j]))
        strata_ok = strata_ok and np.array_equal(np.sort(covered), np.arange(TOTAL_ROWS))
    cov = (full.T @ full) / float(TOTAL_ROWS)
    off = cov - np.diag(np.diag(cov))
    radii = np.linalg.norm(full, axis=1)
    diagnostics = {
        "exact_strata": bool(strata_ok),
        "max_full_coordinate_mean_abs": float(np.max(np.abs(np.mean(full, axis=0)))),
        "radius_mean": float(np.mean(radii)),
        "radius_median": float(np.median(radii)),
        "radius_min": float(np.min(radii)),
        "radius_max": float(np.max(radii)),
        "diag_second_moment_min": float(np.min(np.diag(cov))),
        "diag_second_moment_median": float(np.median(np.diag(cov))),
        "diag_second_moment_max": float(np.max(np.diag(cov))),
        "cov_frobenius_relative_error": float(np.linalg.norm(cov - np.eye(width), ord="fro") / np.linalg.norm(np.eye(width), ord="fro")),
        "cov_max_abs_offdiag": float(np.max(np.abs(off - np.diag(np.diag(off))))),
        "antipode_max_abs": float(np.max(np.abs(full[POSITIVE_ROWS:] + full[:POSITIVE_ROWS]))),
    }
    return positive, diagnostics


def _iid_gaussian_rows(seed: int, width: int) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    rng = np.random.default_rng(seed)
    positive = rng.standard_normal((POSITIVE_ROWS, width)).astype(np.float32)
    full = np.concatenate((positive, -positive), axis=0).astype(np.float64)
    cov = (full.T @ full) / float(TOTAL_ROWS)
    off = cov - np.diag(np.diag(cov))
    radii = np.linalg.norm(full, axis=1)
    diagnostics = {
        "exact_strata": False,
        "max_full_coordinate_mean_abs": float(np.max(np.abs(np.mean(full, axis=0)))),
        "radius_mean": float(np.mean(radii)),
        "radius_median": float(np.median(radii)),
        "radius_min": float(np.min(radii)),
        "radius_max": float(np.max(radii)),
        "diag_second_moment_min": float(np.min(np.diag(cov))),
        "diag_second_moment_median": float(np.median(np.diag(cov))),
        "diag_second_moment_max": float(np.max(np.diag(cov))),
        "cov_frobenius_relative_error": float(np.linalg.norm(cov - np.eye(width), ord="fro") / np.linalg.norm(np.eye(width), ord="fro")),
        "cov_max_abs_offdiag": float(np.max(np.abs(off - np.diag(np.diag(off))))),
        "antipode_max_abs": float(np.max(np.abs(full[POSITIVE_ROWS:] + full[:POSITIVE_ROWS]))),
    }
    return positive, diagnostics


def _hadamard_rows(seed: int, width: int, blocks: int) -> np.ndarray:
    """Return 16 independent positive half-bases, never interleaved pairs."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows: list[np.ndarray] = []
    for _ in range(blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        rows.append(base * flips[None, :])
    return np.concatenate(rows, axis=0)


def _recolor_and_propagate(weights: list[fnp.ndarray], rows: np.ndarray) -> np.ndarray:
    """Apply the fixed current route transforms to one positive input set."""
    positive_rows = fnp.array(rows, dtype=fnp.float32)
    pre_half = _strassen_matmul(positive_rows, weights[0], 3)
    y = fnp.concatenate(
        (fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0
    )

    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3)
        / float(centered.shape[0])
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(weights[0].shape[1])
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]

    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            # Targets may use float64 moments, but the actual centered
            # variance-match application is deliberately fp32.
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
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def _current_route_estimate(mlp, seed: int, rep: int) -> np.ndarray:
    weights = [w.astype(fnp.float32) for w in mlp.weights]
    rows = _hadamard_rows(_derived_seed(seed, CURRENT_STREAM, rep), mlp.width, BLOCKS)
    return _recolor_and_propagate(weights, rows)


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
        lhs_rows, lhs_diag = _lhs_rows(_derived_seed(seed, LHS_STREAM, rep), WIDTH)
        iid_rows, iid_diag = _iid_gaussian_rows(_derived_seed(seed, IID_STREAM, rep), WIDTH)
        estimates = {
            "current": _current_route_estimate(mlp, seed, rep),
            "antithetic_gaussian_lhs": _recolor_and_propagate(weights_f32, lhs_rows),
            "iid_gaussian": _recolor_and_propagate(weights_f32, iid_rows),
        }
        estimated_reps.append({"rep": rep, "estimates": estimates, "lhs_diag": lhs_diag, "iid_diag": iid_diag})

    # Truth is accessed only after every method and replication estimate is fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    reps: list[dict[str, object]] = []
    for estimated in estimated_reps:
        estimates = estimated["estimates"]
        mse = {
            name: float(np.mean((value - truth) ** 2))
            for name, value in estimates.items()
        }
        reps.append(
            {
                "rep": estimated["rep"],
                "mse": mse,
                "lhs_diagnostics": estimated.get("lhs_diag"),
                "iid_diagnostics": estimated.get("iid_diag"),
                "estimates": {name: value.tolist() for name, value in estimates.items()},
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
            "antithetic_rows": 2 * POSITIVE_ROWS,
            "blocks": BLOCKS,
            "reps": REPS,
            "lhs_transform": "per-coordinate permuted low stratum + jitter -> ndtri -> independent orientation + exact antipodes",
            "iid_control": "ordinary unnormalized iid Gaussian positive half + exact antipodes",
            "raw_input_normalization": "none",
            "first_layer": "global exact ReLU mean/covariance recolor",
            "first_successor": "fp32 centered_apply, strength 1.5",
            "propagation": "fp32 L3 Strassen",
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
