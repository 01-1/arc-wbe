#!/usr/bin/env python3
"""Truth-bank payload for paired Gaussian LHS/Sobol ZCA folding."""

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
    _DIRECTION_NUMBERS,
)


WIDTH = 256
DEPTH = 32
POSITIVE_ROWS = 4096
TOTAL_ROWS = 8192
BLOCKS = 16
REPS = 1
EIGEN_FLOOR = 1e-8
UNIFORM_LO = 2.0 ** -53
UNIFORM_HI = 1.0 - UNIFORM_LO
LHS_STREAM = 0x4C4853_0710
SOBOL_STREAM = 0x50B01_0710
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "folded-zca-qmc-v1"
BITS = 30
SOBOL_ROWS = 1 << 12
SOBOL_DIRECTIONS = fnp.array(_DIRECTION_NUMBERS, dtype=fnp.uint32)
BIT_POSITIONS = fnp.arange(BITS, dtype=fnp.uint32)


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int, rep: int) -> int:
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _ndtri_fnp(p: fnp.ndarray) -> fnp.ndarray:
    """Vectorized dependency-free Acklam inverse normal in flopscope."""
    p = fnp.maximum(fnp.minimum(p, UNIFORM_HI), UNIFORM_LO).astype(fnp.float64)
    plow = 0.02425
    low = p < plow
    high = p > 1.0 - plow
    q_low = fnp.sqrt(-2.0 * fnp.log(p))
    num_low = (((((-7.784894002430293e-03 * q_low - 3.223964580411365e-01) * q_low - 2.400758277161838) * q_low - 2.549732539343734) * q_low + 4.374664141464968) * q_low + 2.938163982698783)
    den_low = ((((7.784695709041462e-03 * q_low + 3.224671290700398e-01) * q_low + 2.445134137142996) * q_low + 3.754408661907416) * q_low + 1.0)
    q_high = fnp.sqrt(-2.0 * fnp.log(1.0 - p))
    num_high = (((((-7.784894002430293e-03 * q_high - 3.223964580411365e-01) * q_high - 2.400758277161838) * q_high - 2.549732539343734) * q_high + 4.374664141464968) * q_high + 2.938163982698783)
    den_high = ((((7.784695709041462e-03 * q_high + 3.224671290700398e-01) * q_high + 2.445134137142996) * q_high + 3.754408661907416) * q_high + 1.0)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((( -39.69683028665376 * r_mid + 220.9460984245205) * r_mid - 275.9285104469687) * r_mid + 138.3577518672690) * r_mid - 30.66479806614716) * r_mid + 2.506628277459239) * q_mid)
    den_mid = ((((( -54.47609879822406 * r_mid + 161.5858368580409) * r_mid - 155.6989798598866) * r_mid + 66.80131188771972) * r_mid - 13.28068155288572) * r_mid + 1.0)
    return fnp.where(low, num_low / den_low, fnp.where(high, -num_high / den_high, num_mid / den_mid))


def _lhs_positive(seed: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed)
    columns = []
    for j in range(WIDTH):
        pair_ids = rng.permutation(POSITIVE_ROWS)
        jitter = fnp.maximum(fnp.minimum(rng.random(POSITIVE_ROWS), UNIFORM_HI), UNIFORM_LO)
        u_low = (pair_ids.astype(fnp.float64) + jitter) / float(TOTAL_ROWS)
        z_low = _ndtri_fnp(u_low)
        orientation = rng.integers(0, 2, size=POSITIVE_ROWS, dtype=fnp.int8)
        columns.append(fnp.where(orientation == 0, z_low, -z_low).astype(fnp.float32))
    return fnp.stack(tuple(columns), axis=1).astype(fnp.float32)


def _sobol_positive(seed: int) -> fnp.ndarray:
    """SciPy-compatible LMS+digital-shift Sobol construction in flopscope."""
    rng = fnp.random.default_rng(seed)
    random_bits = rng.integers(0, 2, size=(WIDTH, BITS), dtype=fnp.uint32)
    bit_weights = fnp.left_shift(fnp.ones(BITS, dtype=fnp.uint32), BIT_POSITIONS)
    shift = fnp.sum(random_bits * bit_weights[None, :], axis=1).astype(fnp.uint32)
    ltm = fnp.tril(rng.integers(0, 2, size=(WIDTH, BITS, BITS), dtype=fnp.uint32))
    ltm = fnp.maximum(ltm, fnp.eye(BITS, dtype=fnp.uint32)[None, :, :])
    direction_bits = fnp.bitwise_and(
        fnp.right_shift(SOBOL_DIRECTIONS[:, :, None], BIT_POSITIONS[None, None, :]), 1
    )
    transformed_bits = fnp.sum(
        ltm[:, None, :, ::-1] * direction_bits[:, :, None, :], axis=3
    ) % 2
    powers = fnp.left_shift(
        fnp.ones(BITS, dtype=fnp.uint32), (BITS - 1) - BIT_POSITIONS
    )
    directions = fnp.sum(transformed_bits * powers[None, None, :], axis=2).astype(fnp.uint32)
    indices = fnp.arange(SOBOL_ROWS, dtype=fnp.uint32)
    gray = fnp.bitwise_xor(indices, fnp.right_shift(indices, 1))
    quasi = fnp.broadcast_to(shift[None, :], (SOBOL_ROWS, WIDTH)).astype(fnp.uint32)
    for bit in range(BITS):
        active = fnp.bitwise_and(fnp.right_shift(gray, bit), 1).astype(fnp.bool_)
        quasi = fnp.bitwise_xor(
            quasi,
            fnp.where(active[:, None], directions[None, :, bit], fnp.array(0, dtype=fnp.uint32)),
        )
    uniforms = quasi.astype(fnp.float64) * (1.0 / float(1 << BITS))
    uniforms = fnp.maximum(fnp.minimum(uniforms, UNIFORM_HI), UNIFORM_LO)
    return _ndtri_fnp(uniforms).astype(fnp.float32)


def _hadamard_positive(seed: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed)
    base = _hadamard(WIDTH).astype(fnp.float32)
    pieces = []
    for _ in range(BLOCKS):
        flips = (2.0 * rng.integers(0, 2, size=WIDTH) - 1.0).astype(fnp.float32)
        pieces.append(base * flips[None, :])
    return fnp.concatenate(tuple(pieces), axis=0).astype(fnp.float32)


def _scalar(value: fnp.ndarray) -> float:
    return float(np.asarray(value))


def _vector_stats(values: fnp.ndarray) -> dict[str, float]:
    x = values.astype(fnp.float64)
    ordered = fnp.sort(x)
    mid = (ordered[x.shape[0] // 2 - 1] + ordered[x.shape[0] // 2]) * 0.5
    return {
        "min": _scalar(fnp.min(x)),
        "median": _scalar(mid),
        "max": _scalar(fnp.max(x)),
    }


def _rel_fro(a: fnp.ndarray, b: fnp.ndarray) -> fnp.ndarray:
    return fnp.divide(
        fnp.linalg.norm(a - b, ord="fro"),
        fnp.maximum(fnp.linalg.norm(b, ord="fro"), _MIN_VARIANCE),
    )


def _offdiag(cov: fnp.ndarray) -> fnp.ndarray:
    return cov - fnp.diag(fnp.diag(cov))


def _zca(U: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray, dict[str, object]]:
    # S and all whitening algebra remain metered flopscope operations.
    S = (_strassen_matmul(U.T, U, 3) / float(POSITIVE_ROWS)).astype(fnp.float64)
    S = (S + S.T) * 0.5
    eigenvalues, eigenvectors = fnp.linalg.eigh(S)
    floored = fnp.maximum(eigenvalues, EIGEN_FLOOR)
    inv_sqrt = 1.0 / fnp.sqrt(floored)
    A = eigenvectors @ (inv_sqrt[:, None] * eigenvectors.T)
    A = ((A + A.T) * 0.5).astype(fnp.float32)
    # Fold ZCA into W0; U@A is never formed.
    post = A.T.astype(fnp.float64) @ S @ A.astype(fnp.float64)
    identity = fnp.eye(WIDTH).astype(fnp.float64)
    pre_off = _offdiag(S)
    off = _offdiag(post)
    pre_diag_error = fnp.diag(S) - 1.0
    radius2 = fnp.sum(U * U, axis=1).astype(fnp.float64)
    diag_error = fnp.diag(post) - 1.0
    diagnostics = {
        "pre_cov_rel_fro": _scalar(_rel_fro(S, identity)),
        "pre_diag_max_abs_error": _scalar(fnp.max(fnp.abs(pre_diag_error))),
        "pre_offdiag_rms": _scalar(fnp.sqrt(fnp.sum(pre_off * pre_off) / float(WIDTH * (WIDTH - 1)))),
        "pre_offdiag_max_abs": _scalar(fnp.max(fnp.abs(pre_off))),
        "post_cov_rel_fro": _scalar(_rel_fro(post, identity)),
        "post_diag_max_abs_error": _scalar(fnp.max(fnp.abs(diag_error))),
        "post_offdiag_rms": _scalar(fnp.sqrt(fnp.sum(off * off) / float(WIDTH * (WIDTH - 1)))),
        "post_offdiag_max_abs": _scalar(fnp.max(fnp.abs(off))),
        "eigenvalues": _vector_stats(floored),
        "condition": _scalar(fnp.max(floored) / fnp.maximum(fnp.min(floored), EIGEN_FLOOR)),
        "pre_radius_mean": _scalar(fnp.mean(fnp.sqrt(radius2))),
        "pre_radius_std": _scalar(fnp.sqrt(fnp.mean((fnp.sqrt(radius2) - fnp.mean(fnp.sqrt(radius2))) ** 2))),
        "post_radius_rms": _scalar(fnp.sqrt(fnp.sum(fnp.diag(post)))),
    }
    return A, S, diagnostics


def _recolor_and_propagate(
    weights: list[fnp.ndarray],
    pre_half: fnp.ndarray,
    target_mean: fnp.ndarray,
    target_cov: fnp.ndarray,
) -> np.ndarray:
    y = fnp.concatenate((fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0)
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3)
        / float(TOTAL_ROWS)
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
                1.0 + _DEEP_VARIANCE_MATCH_STRENGTH
                * (fnp.sqrt(target_var / sample_var) - 1.0)
            ).astype(fnp.float32)
            x = centered_apply * scale[None, :] + sample_mean_1.astype(fnp.float32)[None, :]
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def _pair_diag(
    U: fnp.ndarray,
    A: fnp.ndarray,
    S: fnp.ndarray,
    pre_base: fnp.ndarray,
    pre_zca: fnp.ndarray,
    target_pre_cov: fnp.ndarray,
) -> dict[str, object]:
    base_cov = (_strassen_matmul(pre_base.T, pre_base, 3) / float(POSITIVE_ROWS)).astype(fnp.float64)
    zca_cov = (_strassen_matmul(pre_zca.T, pre_zca, 3) / float(POSITIVE_ROWS)).astype(fnp.float64)
    post = A.T.astype(fnp.float64) @ S @ A.astype(fnp.float64)
    return {
        "base_antipode_max_abs": _scalar(fnp.max(fnp.abs(fnp.concatenate((U, -U), axis=0)[POSITIVE_ROWS:] + fnp.concatenate((U, -U), axis=0)[:POSITIVE_ROWS]))),
        "zca_antipode_max_abs": _scalar(fnp.max(fnp.abs(fnp.concatenate((pre_zca, -pre_zca), axis=0)[POSITIVE_ROWS:] + fnp.concatenate((pre_zca, -pre_zca), axis=0)[:POSITIVE_ROWS]))),
        "preactivation_cov_rel_fro_base": _scalar(_rel_fro(base_cov, target_pre_cov)),
        "preactivation_cov_rel_fro_zca": _scalar(_rel_fro(zca_cov, target_pre_cov)),
        "zca_post_cov_rel_fro": _scalar(_rel_fro(post, fnp.eye(WIDTH).astype(fnp.float64))),
    }


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
    target_pre_cov = (weights_f32[0].T @ weights_f32[0]).astype(fnp.float64)
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights_f32[0].T @ weights_f32[0])
    estimated: list[dict[str, object]] = []
    for rep in range(REPS):
        lhs = fnp.array(_lhs_positive(_derived_seed(seed, LHS_STREAM, rep)), dtype=fnp.float32)
        sobol = fnp.array(_sobol_positive(_derived_seed(seed, SOBOL_STREAM, rep)), dtype=fnp.float32)
        current = _hadamard_positive(_derived_seed(seed, CURRENT_STREAM, rep))
        U_routes = {"lhs": lhs, "sobol": sobol}
        estimates = {"current": _recolor_and_propagate(weights_f32, _strassen_matmul(current, weights_f32[0], 3), target_mean, target_cov)}
        diagnostics: dict[str, object] = {
            "current": {"antipode_max_abs": _scalar(fnp.max(fnp.abs(fnp.concatenate((current, -current), axis=0)[POSITIVE_ROWS:] + fnp.concatenate((current, -current), axis=0)[:POSITIVE_ROWS])))}
        }
        for family, U in U_routes.items():
            pre_base = _strassen_matmul(U, weights_f32[0], 3)
            A, S, zca_diag = _zca(U)
            W0_eff = A @ weights_f32[0]
            pre_zca = _strassen_matmul(U, W0_eff, 3)
            pair = _pair_diag(U, A, S, pre_base, pre_zca, target_pre_cov)
            diagnostics[family] = {"zca": {**zca_diag, **pair}}
            estimates[f"{family}_base"] = _recolor_and_propagate(weights_f32, pre_base, target_mean, target_cov)
            estimates[f"{family}_zca"] = _recolor_and_propagate(weights_f32, pre_zca, target_mean, target_cov)
        estimated.append({"rep": rep, "estimates": estimates, "diagnostics": diagnostics})

    # Truth is read only after every route vector and diagnostic is fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    reps = []
    for item in estimated:
        reps.append({
            "rep": item["rep"],
            "mse": {name: float(np.mean((value - truth) ** 2)) for name, value in item["estimates"].items()},
            "estimates": {name: value.tolist() for name, value in item["estimates"].items()},
            "diagnostics": item["diagnostics"],
        })
    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": True,
        "weights_sha256": actual,
        "config": {
            "width": WIDTH, "depth": DEPTH, "positive_rows": POSITIVE_ROWS,
            "total_rows": TOTAL_ROWS, "blocks": BLOCKS, "reps": REPS,
            "routes": ["current", "lhs_base", "lhs_zca", "sobol_base", "sobol_zca"],
            "lhs_generation": "fnp Generator permutation/random/integer calls, fnp Acklam inverse-normal, fnp column stack; frozen LHS streams and strata",
            "sobol_generation": "fnp Generator LMS lower-triangular scramble/digital shift, fnp GF(2) directions/Gray rows, fnp Acklam inverse-normal",
            "generator_parity": "fnp LHS/Sobol rows parity-checked against reviewed dependency-free NumPy generators",
            "zca": "flopscope float64 eigh, eigen floor 1e-8, symmetric A=S^(-1/2), folded W0_eff=A@W0; U@A not formed",
            "first_layer": "exact global ReLU mean/covariance recolor with original W0 target",
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
