#!/usr/bin/env python3
"""Coordinator-fixed Sobol sphere versus pilot-Jacobian LT gate."""

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
PAYLOAD_DIR = Path(__file__).resolve().parent
if str(PAYLOAD_DIR) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_DIR))

from sobol_runtime_feasibility_generator_20260710 import sobol_normal_rows

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
POSITIVE_ROWS = 4096
HADAMARD_BLOCKS = 16
PILOTS = 8
PROBES = 8
SOBOL_STREAM = 0x50B01_0710
CURRENT_STREAM = 0xC0A1_0710
PILOT_STREAM = 0x4A43_0710
SCRIPT_VERSION = "sobol-jacobian-lt-v2"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int, rep: int) -> int:
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _sobol_sphere_rows(seed: int, width: int) -> np.ndarray:
    if width != WIDTH:
        raise ValueError(f"reviewed Sobol generator is fixed at width {WIDTH}; got {width}")
    return sobol_normal_rows(seed)


def _hadamard_rows(seed: int, width: int, blocks: int) -> np.ndarray:
    """Exactly independent positive half-bases; no interleaving helper."""
    rng = np.random.default_rng(seed)
    base = np.asarray(_hadamard(width), dtype=np.float32)
    rows = []
    for _ in range(blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        rows.append(base * flips[None, :])
    return np.concatenate(rows, axis=0)


def _pilot_q_and_concentration(
    weights: list[fnp.ndarray], seed: int
) -> tuple[fnp.ndarray, list[float], list[float]]:
    """Build complete QR Q from eight label-free input-gradient probes."""
    rng = fnp.random.default_rng(seed)
    pilots_raw = rng.standard_normal((PILOTS, WIDTH)).astype(fnp.float32)
    pilot_norm = fnp.sqrt(fnp.sum(pilots_raw * pilots_raw, axis=1))
    pilots = pilots_raw / pilot_norm[:, None]
    probes = (2.0 * rng.integers(0, 2, size=(PROBES, WIDTH)) - 1.0).astype(fnp.float32)

    gradients = []
    for pilot_idx in range(PILOTS):
        x = pilots[pilot_idx : pilot_idx + 1]
        gates = []
        for weight in weights:
            pre = fnp.matmul(x, weight)
            gates.append(pre > 0.0)
            x = fnp.maximum(pre, 0.0)

        # Scalar probe is probe dot final activation. Backpropagate the
        # resulting row-vector gradient through every saved ReLU gate.
        grad = probes[pilot_idx : pilot_idx + 1]
        for layer_idx in range(len(weights) - 1, -1, -1):
            grad = grad * fnp.where(gates[layer_idx], 1.0, 0.0)
            grad = fnp.matmul(grad, weights[layer_idx].T)
        gradients.append(grad[0])

    gradient_matrix = fnp.stack(tuple(gradients), axis=1)
    q, r = fnp.linalg.qr(gradient_matrix, mode="complete")
    diagonal = fnp.diag(r[:PILOTS, :PILOTS])
    signs = fnp.where(diagonal >= 0.0, 1.0, -1.0)
    q = q * fnp.concatenate((signs, fnp.ones(WIDTH - PILOTS)))[None, :]

    # This diagnostic is deliberately downstream of the fixed gradient/Q
    # construction and cannot influence the transform or estimates.
    singular = np.linalg.svd(np.asarray(gradient_matrix, dtype=np.float64), compute_uv=False)
    energy = np.maximum(np.sum(singular * singular), 1e-300)
    concentration = [float(np.sum(singular[:k] ** 2) / energy) for k in (1, 2, 4, 8)]
    return q, [float(value) for value in singular], concentration


def _recolor_and_propagate(weights: list[fnp.ndarray], positive_rows: np.ndarray) -> np.ndarray:
    """Current global recolor + fp32 centered_apply + fp32 L3 propagation."""
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


def _current_estimate(mlp, seed: int, rep: int, weights: list[fnp.ndarray]) -> np.ndarray:
    rows = _hadamard_rows(
        _derived_seed(seed, CURRENT_STREAM, rep), mlp.width, HADAMARD_BLOCKS
    )
    return _recolor_and_propagate(weights, rows)


def run_one(shard_index: int, bank_path: Path, reps: int) -> dict[str, object]:
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
    for rep in range(reps):
        # Both Sobol routes receive this exact same pre-transform array.
        sobol_rows = _sobol_sphere_rows(
            _derived_seed(seed, SOBOL_STREAM, rep), WIDTH
        )
        pilot_q, singular_values, concentration = _pilot_q_and_concentration(
            weights_f32, _derived_seed(seed, PILOT_STREAM, rep)
        )
        sobol_rows_f32 = fnp.array(sobol_rows, dtype=fnp.float32)
        lt_rows = fnp.matmul(sobol_rows_f32, pilot_q.T)
        estimates = {
            "current": _current_estimate(mlp, seed, rep, weights_f32),
            "sobol_unrotated": _recolor_and_propagate(weights_f32, sobol_rows),
            "sobol_lt": _recolor_and_propagate(weights_f32, np.asarray(lt_rows)),
        }
        estimated_reps.append(
            {
                "rep": rep,
                "estimates": estimates,
                "pilot_singular_values": singular_values,
                "pilot_singular_concentration_top_1_2_4_8": concentration,
            }
        )

    # Truth is accessed only after every route estimate in every requested
    # replication has been fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    scored_reps = []
    for estimated in estimated_reps:
        estimates = estimated["estimates"]
        scored_reps.append(
            {
                "rep": estimated["rep"],
                "mse": {
                    name: float(np.mean((value - truth) ** 2))
                    for name, value in estimates.items()
                },
                "estimates": {name: value.tolist() for name, value in estimates.items()},
                "pilot_singular_values": estimated["pilot_singular_values"],
                "pilot_singular_concentration_top_1_2_4_8": estimated[
                    "pilot_singular_concentration_top_1_2_4_8"
                ],
            }
        )
    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "stage_reps": reps,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": True,
        "weights_sha256": actual,
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "positive_rows": POSITIVE_ROWS,
            "antithetic_rows": 2 * POSITIVE_ROWS,
            "hadamard_blocks": HADAMARD_BLOCKS,
            "pilots": PILOTS,
            "probes": PROBES,
            "first_layer": "global exact ReLU mean/covariance recolor",
            "first_successor": "fp32 centered_apply, strength 1.5",
            "propagation": "fp32 L3 Strassen",
            "sobol_transform": "same 4096 ndtri sphere rows; LT x=z@Q.T",
            "radial_factor": "none",
            "q_orientation": "complete QR columns 0:8 are pilot-gradient span",
        },
        "truth_final": truth.tolist(),
        "reps": scored_reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--reps", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank, args.reps)
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
