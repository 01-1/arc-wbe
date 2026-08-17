#!/usr/bin/env python3
"""Fixed Hadamard-oriented antithetic Gaussian Latin-hypercube gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = Path(__file__).resolve().parent
for path in (ROOT, PAYLOAD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sobol_runtime_feasibility_generator_20260710 import ndtri_dependency_free

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


WIDTH = 256
DEPTH = 32
TOTAL_ROWS = 8192
POSITIVE_ROWS = 4096
BLOCKS = 16
MAGNITUDE_STREAM = 0x4D41_0710
INDEPENDENT_SIGN_STREAM = 0x1D51_0710
HADAMARD_SIGN_STREAM = 0x4841_0710
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "hadamard-lhs-v1"


def _derived_seed(seed: int, stream: int) -> int:
    return int((seed ^ stream) % (1 << 32))


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _hadamard_rows(seed: int, width: int, blocks: int) -> np.ndarray:
    """Corrected current route: independent positive bases, never interleaved."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows = []
    for _ in range(blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        rows.append(base * flips[None, :])
    return np.concatenate(rows, axis=0)


def _shared_magnitudes(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One shared coordinatewise permutation/jitter magnitude table."""
    rng = np.random.default_rng(seed)
    pair_ids = np.stack(
        [rng.permutation(POSITIVE_ROWS) for _ in range(WIDTH)], axis=1
    ).astype(np.int32)
    jitter = rng.random((POSITIVE_ROWS, WIDTH), dtype=np.float64)
    tiny = np.nextafter(0.0, 1.0)
    below_one = np.nextafter(1.0, 0.0)
    jitter = np.maximum(np.minimum(jitter, below_one), tiny)
    u_low = (pair_ids.astype(np.float64) + jitter) / float(TOTAL_ROWS)
    magnitudes = -ndtri_dependency_free(u_low)
    return magnitudes.astype(np.float32), pair_ids


def _independent_signs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(
        rng.integers(0, 2, size=(POSITIVE_ROWS, WIDTH)) == 0, -1.0, 1.0
    ).astype(np.float32)


def _hadamard_signs(seed: int) -> np.ndarray:
    """Exactly 16 independent randomized positive Hadamard half-bases."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hadamard(WIDTH), dtype=np.float32)
    rows = []
    for _ in range(BLOCKS):
        column_flips = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=WIDTH
        )
        rows.append(base * column_flips[None, :])
    return np.concatenate(rows, axis=0)


def _full_antithetic(x_half: np.ndarray) -> np.ndarray:
    return np.concatenate((x_half, -x_half), axis=0)


def _lhs_diagnostics(magnitudes: np.ndarray, pair_ids: np.ndarray, signs: np.ndarray) -> dict[str, object]:
    """Audit exact strata, antipodes, covariance, diagonal, and off-diagonals."""
    x_half = signs * magnitudes
    full = _full_antithetic(x_half)
    strata = np.concatenate((pair_ids, TOTAL_ROWS - 1 - pair_ids), axis=0)
    expected = np.arange(TOTAL_ROWS, dtype=np.int32)[:, None]
    strata_ok = bool(np.all(np.sort(strata, axis=0) == expected))
    antipode_max_abs = float(np.max(np.abs(full[POSITIVE_ROWS:] + full[:POSITIVE_ROWS])))
    coord_mean = np.mean(full, axis=0)
    covariance = (full.T @ full) / float(TOTAL_ROWS)
    diagonal = np.diag(covariance)
    offdiag = covariance - np.diag(diagonal)
    return {
        "strata_exact": strata_ok,
        "strata_unique_per_coordinate": int(
            np.min([len(np.unique(strata[:, j])) for j in range(WIDTH)])
        ),
        "antipode_max_abs": antipode_max_abs,
        "coordinate_mean_max_abs": float(np.max(np.abs(coord_mean))),
        "diagonal_second_moment_mean": float(np.mean(diagonal)),
        "diagonal_second_moment_max_abs_error": float(np.max(np.abs(diagonal - 1.0))),
        "offdiagonal_rms": float(np.sqrt(np.mean(offdiag * offdiag))),
        "offdiagonal_max_abs": float(np.max(np.abs(offdiag))),
    }


def _recolor_and_propagate(weights: list[fnp.ndarray], positive_rows: np.ndarray) -> np.ndarray:
    """Current exact recolor, fp32 centered_apply, and fp32 L3 suffix."""
    rows = fnp.array(positive_rows, dtype=fnp.float32)
    pre_half = _strassen_matmul(rows, weights[0], 3)
    y = fnp.concatenate(
        (fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0
    )
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(
            centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3
        )
        / float(centered.shape[0])
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(WIDTH)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
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

    magnitudes, pair_ids = _shared_magnitudes(_derived_seed(seed, MAGNITUDE_STREAM))
    independent_signs = _independent_signs(
        _derived_seed(seed, INDEPENDENT_SIGN_STREAM)
    )
    hadamard_signs = _hadamard_signs(_derived_seed(seed, HADAMARD_SIGN_STREAM))
    independent_rows = independent_signs * magnitudes
    hadamard_rows = hadamard_signs * magnitudes

    # All estimates and label-free diagnostics are fixed before truth access.
    estimates = {
        "current": _recolor_and_propagate(
            weights_f32,
            _hadamard_rows(_derived_seed(seed, CURRENT_STREAM), WIDTH, BLOCKS),
        ),
        "lhs_independent": _recolor_and_propagate(weights_f32, independent_rows),
        "lhs_hadamard": _recolor_and_propagate(weights_f32, hadamard_rows),
    }
    diagnostics = {
        "lhs_independent": _lhs_diagnostics(
            magnitudes, pair_ids, independent_signs
        ),
        "lhs_hadamard": _lhs_diagnostics(magnitudes, pair_ids, hadamard_signs),
    }

    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
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
            "total_rows": TOTAL_ROWS,
            "positive_rows": POSITIVE_ROWS,
            "blocks": BLOCKS,
            "antipodes": "exact row negation",
            "magnitude_table": "shared coordinatewise permutation of pair IDs with open jitter",
            "input_normalization": "none",
            "raw_input_recolor": "none",
            "first_layer": "global exact ReLU mean/covariance recolor",
            "first_successor": "fp32 centered_apply, strength 1.5",
            "propagation": "fp32 L3 Strassen",
        },
        "mse": {name: float(np.mean((value - truth) ** 2)) for name, value in estimates.items()},
        "diagnostics": diagnostics,
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
