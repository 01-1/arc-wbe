#!/usr/bin/env python3
"""Frozen truth-bank payload for antipodal odd low-rank transport."""

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

import flopscope.numpy as fnp  # noqa: E402
from estimator import (  # noqa: E402
    _DEEP_VARIANCE_MATCH_STRENGTH,
    _MIN_VARIANCE,
    _gaussian_relu_variance,
    _hadamard,
    _strassen_matmul,
    _zero_mean_relu_mean_cov,
)
from local_engine import build_mlp  # noqa: E402


WIDTH = 256
DEPTH = 32
BLOCKS = 16
PAIRS_PER_BLOCK = WIDTH
POSITIVE_ROWS = BLOCKS * PAIRS_PER_BLOCK
TOTAL_ROWS = 2 * POSITIVE_ROWS
REPS = 3
SHARED_LAST_LAYER = 4
ROUTE_STREAMS = (0x0DD1_0710, 0x0DD1_1710, 0x0DD1_2710)
REFRESH_STREAMS = (0x6A40_0710, 0x6A40_1710, 0x6A40_2710, 0x6A40_3710)
REFRESH_BOUNDARIES = (4, 8, 16, 24)
REFRESH_RANKS = (64, 32, 16, 8)
OVERSAMPLE = 8
GRAM_JITTER_REL = 1e-7
PAIR_RECON_TOL = 2e-7
Q_ORTH_TOL = 2e-5
PROJECTION_RANGE_TOL = 5e-5
PROJECTION_IDENTITY_TOL = 5e-4
SECOND_MOMENT_REL_TOL = 2e-4
SCRIPT_VERSION = "odd-lr-transport-v1-20260710"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int, rep: int) -> int:
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _hadamard_positive(seed: int) -> fnp.ndarray:
    """Return 16 independently signed positive Hadamard bases in block order."""
    rng = fnp.random.default_rng(seed)
    base = _hadamard(WIDTH).astype(fnp.float32)
    blocks = []
    for _ in range(BLOCKS):
        signs = (2.0 * rng.integers(0, 2, size=WIDTH) - 1.0).astype(fnp.float32)
        blocks.append(base * signs[None, :])
    return fnp.concatenate(tuple(blocks), axis=0).astype(fnp.float32)


def _scalar(value: fnp.ndarray) -> float:
    return float(np.asarray(value))


def _finite(value: fnp.ndarray) -> bool:
    return bool(np.asarray(fnp.all(fnp.isfinite(value))))


def _relative_fro(a: fnp.ndarray, b: fnp.ndarray) -> fnp.ndarray:
    return fnp.linalg.norm(a - b, ord="fro") / fnp.maximum(
        fnp.linalg.norm(b, ord="fro"), _MIN_VARIANCE
    )


def _initial_recolor(
    weights: list[fnp.ndarray], pre_half: fnp.ndarray
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Apply the corrected current global first-ReLU mean/covariance recolor."""
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    y = fnp.concatenate(
        (fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0
    )
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(
            centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3
        )
        / float(TOTAL_ROWS)
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(WIDTH)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    state0 = _strassen_matmul(
        centered.astype(fnp.float32), recolor.astype(fnp.float32), 3
    ) + target_mean.astype(fnp.float32)[None, :]
    return state0.astype(fnp.float32), target_mean, target_cov


def _current_successor_update(x: fnp.ndarray, pre: fnp.ndarray) -> fnp.ndarray:
    """Apply current strength-1.5 matching with fp32 centering/writeback."""
    pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
    pre_centered = pre - pre_mean[None, :]
    target_var = _gaussian_relu_variance(
        pre_mean,
        fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
    )
    sample_mean = fnp.mean(x, axis=0).astype(fnp.float64)
    centered = x - sample_mean.astype(fnp.float32)[None, :]
    sample_var = fnp.maximum(
        fnp.mean(centered * centered, axis=0).astype(fnp.float64), _MIN_VARIANCE
    )
    scale = (
        1.0
        + _DEEP_VARIANCE_MATCH_STRENGTH
        * (fnp.sqrt(target_var / sample_var) - 1.0)
    ).astype(fnp.float32)
    return (
        centered.astype(fnp.float32) * scale[None, :]
        + sample_mean.astype(fnp.float32)[None, :]
    ).astype(fnp.float32)


def _shared_layer4(weights: list[fnp.ndarray], state0: fnp.ndarray) -> fnp.ndarray:
    x = state0
    for layer_idx in range(1, SHARED_LAST_LAYER + 1):
        pre = _strassen_matmul(x, weights[layer_idx], 3)
        x = fnp.maximum(pre, 0.0).astype(fnp.float32)
        if layer_idx == 1:
            x = _current_successor_update(x, pre)
    return x.astype(fnp.float32)


def _pair_split(x: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray, dict[str, object]]:
    plus = x[:POSITIVE_ROWS]
    minus = x[POSITIVE_ROWS:]
    even = ((plus + minus) * 0.5).astype(fnp.float32)
    odd = ((plus - minus) * 0.5).astype(fnp.float32)
    plus_residual = _relative_fro(even + odd, plus)
    minus_residual = _relative_fro(even - odd, minus)
    diagnostics = {
        "plus_relative_fro": _scalar(plus_residual),
        "minus_relative_fro": _scalar(minus_residual),
        "max_relative_fro": _scalar(fnp.maximum(plus_residual, minus_residual)),
        "finite": _finite(even) and _finite(odd),
    }
    return even, odd, diagnostics


def _refresh_subspace(
    odd: fnp.ndarray, rank: int, rng_seed: int, boundary: int
) -> tuple[fnp.ndarray, fnp.ndarray, dict[str, object]]:
    """One-pass randomized left range with the frozen Cholesky/eigh recipe."""
    sketch_width = rank + OVERSAMPLE
    rng = fnp.random.default_rng(rng_seed)
    omega = (
        2.0 * rng.integers(0, 2, size=(WIDTH, sketch_width)) - 1.0
    ).astype(fnp.float32) / fnp.sqrt(fnp.array(float(sketch_width), dtype=fnp.float32))
    y = _strassen_matmul(odd, omega, 3).astype(fnp.float32)

    # Only these small Gram/factorization operations are promoted to fp64.
    y64 = y.astype(fnp.float64)
    gram = y64.T @ y64
    gram = (gram + gram.T) * 0.5
    gram_eigenvalues, _ = fnp.linalg.eigh(gram)
    gram_scale = fnp.maximum(fnp.mean(fnp.diag(gram)), _MIN_VARIANCE)
    jitter = gram_scale * GRAM_JITTER_REL
    chol = fnp.linalg.cholesky(gram + jitter * fnp.eye(sketch_width))
    q0 = fnp.linalg.solve(chol, y64.T).T.astype(fnp.float32)

    b0 = (q0.T @ odd).astype(fnp.float32)
    range_gram = (b0 @ b0.T).astype(fnp.float64)
    range_gram = (range_gram + range_gram.T) * 0.5
    eigenvalues, eigenvectors = fnp.linalg.eigh(range_gram)
    # eigh is ascending; reverse the retained top-r columns deterministically.
    u_top = eigenvectors[:, -rank:][:, ::-1].astype(fnp.float32)
    q = (q0 @ u_top).astype(fnp.float32)

    coefficients = (q.T @ odd).astype(fnp.float32)
    projected = (q @ coefficients).astype(fnp.float32)
    residual = _relative_fro(projected, odd)
    odd_energy = fnp.maximum(fnp.sum(odd * odd), _MIN_VARIANCE)
    captured_energy = fnp.sum(projected * projected) / odd_energy
    q_gram_f32 = (q.T @ q).astype(fnp.float32)
    q_gram = q_gram_f32.astype(fnp.float64)
    identity = fnp.eye(rank)
    orthogonality = fnp.linalg.norm(q_gram - identity, ord="fro") / fnp.linalg.norm(
        identity, ord="fro"
    )
    identity_error = fnp.abs(captured_energy + residual * residual - 1.0)
    diagnostics = {
        "boundary": boundary,
        "rank": rank,
        "oversample": OVERSAMPLE,
        "rng_seed": rng_seed,
        "projection_relative_fro": _scalar(residual),
        "captured_energy": _scalar(captured_energy),
        "projection_identity_error": _scalar(identity_error),
        "q_orthogonality_relative_fro": _scalar(orthogonality),
        "gram_min_eigenvalue": _scalar(gram_eigenvalues[0]),
        "gram_jitter": _scalar(jitter),
        "gram_min_plus_jitter": _scalar(gram_eigenvalues[0] + jitter),
        "range_retained_min_eigenvalue": _scalar(eigenvalues[-rank]),
        "range_max_eigenvalue": _scalar(eigenvalues[-1]),
        "finite": all(
            (
                _finite(omega),
                _finite(y),
                _finite(q),
                _finite(coefficients),
                _finite(gram_eigenvalues),
                _finite(eigenvalues),
            )
        ),
    }
    return q, q_gram_f32, diagnostics


def _candidate_transport(
    weights: list[fnp.ndarray], layer4: fnp.ndarray, seed: int, rep: int
) -> tuple[fnp.ndarray, fnp.ndarray, dict[str, object]]:
    even, odd, pair_diagnostics = _pair_split(layer4)
    refreshes = []
    layers = []
    q = None
    refresh_lookup = dict(zip(REFRESH_BOUNDARIES, REFRESH_RANKS))
    stream_lookup = dict(zip(REFRESH_BOUNDARIES, REFRESH_STREAMS))

    q, q_gram, refresh_diag = _refresh_subspace(
        odd,
        refresh_lookup[4],
        _derived_seed(seed, stream_lookup[4], rep),
        4,
    )
    refreshes.append(refresh_diag)
    for layer_idx in range(5, DEPTH):
        weight = weights[layer_idx]
        coefficients = (q.T @ odd).astype(fnp.float32)
        target_sumsq = fnp.sum(odd * odd, axis=0).astype(fnp.float32)
        projected_sumsq = fnp.sum(
            coefficients * (q_gram @ coefficients), axis=0
        ).astype(fnp.float32)
        coordinate_scale = fnp.sqrt(
            fnp.maximum(target_sumsq, _MIN_VARIANCE)
            / fnp.maximum(projected_sumsq, _MIN_VARIANCE)
        ).astype(fnp.float32)
        scaled_coefficients = (
            coefficients * coordinate_scale[None, :]
        ).astype(fnp.float32)
        restored_sumsq = fnp.sum(
            scaled_coefficients * (q_gram @ scaled_coefficients), axis=0
        ).astype(fnp.float32)
        coordinate_relative_error = fnp.abs(
            restored_sumsq - target_sumsq
        ) / fnp.maximum(target_sumsq, _MIN_VARIANCE)
        sorted_scale = fnp.sort(coordinate_scale)
        a = _strassen_matmul(even, weight, 3).astype(fnp.float32)
        cw = _strassen_matmul(scaled_coefficients, weight, 3).astype(fnp.float32)
        b = (q @ cw).astype(fnp.float32)
        plus = fnp.maximum(a + b, 0.0).astype(fnp.float32)
        minus = fnp.maximum(a - b, 0.0).astype(fnp.float32)
        even = ((plus + minus) * 0.5).astype(fnp.float32)
        odd = ((plus - minus) * 0.5).astype(fnp.float32)
        layers.append(
            {
                "layer": layer_idx,
                "rank": int(q.shape[1]),
                "post_scale_second_moment_relative_error_max": _scalar(
                    fnp.max(coordinate_relative_error)
                ),
                "post_scale_second_moment_relative_error_mean": _scalar(
                    fnp.mean(coordinate_relative_error)
                ),
                "scale_min": _scalar(fnp.min(coordinate_scale)),
                "scale_median": _scalar(
                    0.5 * (sorted_scale[WIDTH // 2 - 1] + sorted_scale[WIDTH // 2])
                ),
                "scale_max": _scalar(fnp.max(coordinate_scale)),
                "finite": _finite(coordinate_scale)
                and _finite(coordinate_relative_error),
            }
        )
        if layer_idx in (8, 16, 24):
            q, q_gram, refresh_diag = _refresh_subspace(
                odd,
                refresh_lookup[layer_idx],
                _derived_seed(seed, stream_lookup[layer_idx], rep),
                layer_idx,
            )
            refreshes.append(refresh_diag)

    diagnostics = {
        "pair_reconstruction": pair_diagnostics,
        "refreshes": refreshes,
        "layers": layers,
        "finite": _finite(even)
        and _finite(odd)
        and all(d["finite"] for d in refreshes)
        and all(d["finite"] for d in layers),
    }
    return fnp.mean(even, axis=0).astype(fnp.float64), even, diagnostics


def _current_final(
    weights: list[fnp.ndarray], layer4: fnp.ndarray
) -> tuple[fnp.ndarray, fnp.ndarray]:
    x = layer4
    for layer_idx in range(5, DEPTH):
        x = fnp.maximum(_strassen_matmul(x, weights[layer_idx], 3), 0.0).astype(
            fnp.float32
        )
    exact_even = (
        (x[:POSITIVE_ROWS] + x[POSITIVE_ROWS:]) * 0.5
    ).astype(fnp.float32)
    return fnp.mean(exact_even, axis=0).astype(fnp.float64), exact_even


def _block_correction_diagnostics(
    exact_even: fnp.ndarray, candidate_even: fnp.ndarray
) -> dict[str, object]:
    exact_blocks = fnp.mean(
        fnp.reshape(exact_even, (BLOCKS, PAIRS_PER_BLOCK, WIDTH)), axis=1
    ).astype(fnp.float32)
    candidate_blocks = fnp.mean(
        fnp.reshape(candidate_even, (BLOCKS, PAIRS_PER_BLOCK, WIDTH)), axis=1
    ).astype(fnp.float32)
    correction = exact_blocks - candidate_blocks
    # Center over the 16 blocks separately for each output coordinate, then pool.
    exact_centered = exact_blocks - fnp.mean(exact_blocks, axis=0)[None, :]
    candidate_centered = candidate_blocks - fnp.mean(
        candidate_blocks, axis=0
    )[None, :]
    correction_centered = correction - fnp.mean(correction, axis=0)[None, :]
    exact_var = fnp.mean(exact_centered * exact_centered)
    candidate_var = fnp.mean(candidate_centered * candidate_centered)
    correction_var = fnp.mean(correction_centered * correction_centered)
    covariance = fnp.mean(exact_centered * candidate_centered)
    denominator = fnp.sqrt(
        fnp.maximum(exact_var * candidate_var, _MIN_VARIANCE * _MIN_VARIANCE)
    )
    diagnostics = {
        "correction_over_exact_variance": _scalar(
            correction_var / fnp.maximum(exact_var, _MIN_VARIANCE)
        ),
        "exact_candidate_correlation": _scalar(covariance / denominator),
        "candidate_over_exact_variance": _scalar(
            candidate_var / fnp.maximum(exact_var, _MIN_VARIANCE)
        ),
        "exact_block_variance": _scalar(exact_var),
        "candidate_block_variance": _scalar(candidate_var),
        "correction_block_variance": _scalar(correction_var),
        "finite": _finite(exact_blocks)
        and _finite(candidate_blocks)
        and _finite(correction),
        "grouping": "16 original Hadamard blocks x 256 aligned pairs; centered per output coordinate over blocks, then pooled",
        "diagnostic_only": True,
    }
    return diagnostics


def run_one(shard_index: int, bank_path: Path) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights_np = [np.asarray(weight, dtype=np.float32) for weight in mlp.weights]
    actual = _weights_sha256(weights_np)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")
    weights = [fnp.array(weight, dtype=fnp.float32) for weight in weights_np]

    fixed_reps = []
    for rep, route_stream in enumerate(ROUTE_STREAMS):
        positive = _hadamard_positive(_derived_seed(seed, route_stream, rep))
        pre_half = _strassen_matmul(positive, weights[0], 3)
        state0, target_mean, target_cov = _initial_recolor(weights, pre_half)
        layer4 = _shared_layer4(weights, state0)
        current_vector, exact_even = _current_final(weights, layer4)
        candidate_vector, candidate_even, transport_diagnostics = _candidate_transport(
            weights, layer4, seed, rep
        )
        block_diagnostics = _block_correction_diagnostics(exact_even, candidate_even)
        fixed_reps.append(
            {
                "rep": rep,
                "current_vector": current_vector,
                "candidate_vector": candidate_vector,
                "diagnostics": {
                    "state0_finite": _finite(state0),
                    "layer4_finite": _finite(layer4),
                    "target_mean_finite": _finite(target_mean),
                    "target_cov_finite": _finite(target_cov),
                    "transport": transport_diagnostics,
                    "block_correction": block_diagnostics,
                },
            }
        )

    # Truth is intentionally accessed only after all vectors and diagnostics are fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    reps = []
    for item in fixed_reps:
        current = np.asarray(item["current_vector"], dtype=np.float64)
        candidate = np.asarray(item["candidate_vector"], dtype=np.float64)
        reps.append(
            {
                "rep": item["rep"],
                "mse": {
                    "current": float(np.mean((current - truth) ** 2)),
                    "candidate": float(np.mean((candidate - truth) ** 2)),
                },
                "current_vector": current.tolist(),
                "candidate_vector": candidate.tolist(),
                "diagnostics": item["diagnostics"],
            }
        )

    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": actual == expected,
        "weights_sha256": actual,
        "expected_weights_sha256": expected,
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "blocks": BLOCKS,
            "pairs_per_block": PAIRS_PER_BLOCK,
            "positive_rows": POSITIVE_ROWS,
            "total_rows": TOTAL_ROWS,
            "reps": REPS,
            "route_streams": list(ROUTE_STREAMS),
            "refresh_streams": list(REFRESH_STREAMS),
            "schedule": [
                {"refresh_after_layer": 4, "rank": 64, "used_for_layers": [5, 8]},
                {"refresh_after_layer": 8, "rank": 32, "used_for_layers": [9, 16]},
                {"refresh_after_layer": 16, "rank": 16, "used_for_layers": [17, 24]},
                {"refresh_after_layer": 24, "rank": 8, "used_for_layers": [25, 31]},
            ],
            "oversample": OVERSAMPLE,
            "state_indexing": "state0=post-recolor first ReLU; W1..W4 shared exact; candidate transport W5..W31; current exact W5..W31",
            "pairing": "positive_rows[p] paired with negative_rows[p] after global recolor",
            "prediction_scope": "final W31 pair-even mean vector only",
            "first_successor": "fp64 moments; fp32 centered strength-1.5 scale application/writeback",
            "propagation": "fp32 L3 Strassen",
            "subspace_dtypes": "fp32 Omega/Y/Q0/B0/Q/Qgram/C/C_scaled/A/B/E/O; fp64 small Gram, Cholesky, solve, and eigh",
            "odd_energy_restore": "each layer uses Qgram=Q.T@Q and coordinate scale sqrt(max(sum(O^2),tiny)/max(sum(C*(Qgram@C)),tiny)); restored energy is checked by the same identity; no per-layer Q@C diagnostic apply and no clipping",
            "recolor_dtypes": "corrected current route: fp64 moments/Cholesky/recolor solve; fp32 centered recolor application/writeback",
            "tolerances": {
                "pair_reconstruction_relative_fro_max": PAIR_RECON_TOL,
                "q_orthogonality_relative_fro_max": Q_ORTH_TOL,
                "projection_range_tolerance": PROJECTION_RANGE_TOL,
                "projection_identity_error_max": PROJECTION_IDENTITY_TOL,
                "post_scale_second_moment_relative_error_max": SECOND_MOMENT_REL_TOL,
                "gram_jitter_relative": GRAM_JITTER_REL,
            },
            "block_correction": "diagnostic-only statistics from 16x256 final pair-even block means centered per output coordinate over blocks, then pooled",
        },
        "truth_final": truth.tolist(),
        "reps": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("WHEST_SHARD_INDEX", "0")),
    )
    parser.add_argument(
        "--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank)
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
