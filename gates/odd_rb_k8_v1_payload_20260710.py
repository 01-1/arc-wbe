#!/usr/bin/env python3
"""Truth-bank payload for antipodal odd-state Rao--Blackwell K8 closure."""

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

import flopscope as flops
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
TOTAL_ROWS = 8192
BLOCKS = 16
REPS = 3
PREFIX_LAYERS = 8
SUFFIX_START = 9
STREAMS = (0x0DD8_0710, 0x0DD8_1710, 0x0DD8_2710)
SCRIPT_VERSION = "odd-rb-k8-v1"
SERIAL_TOL = 1e-12


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, stream: int, rep: int) -> int:
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _hadamard_positive(seed: int) -> fnp.ndarray:
    """Exactly 16 independent positive randomized Hadamard bases."""
    rng = fnp.random.default_rng(seed)
    base = _hadamard(WIDTH).astype(fnp.float32)
    pieces = []
    for _ in range(BLOCKS):
        flips = (2.0 * rng.integers(0, 2, size=WIDTH) - 1.0).astype(fnp.float32)
        pieces.append(base * flips[None, :])
    return fnp.concatenate(tuple(pieces), axis=0).astype(fnp.float32)


def _scalar(value: fnp.ndarray) -> float:
    return float(np.asarray(value))


def _finite(value: fnp.ndarray) -> bool:
    return bool(np.asarray(fnp.all(fnp.isfinite(value))))


def _rel_fro(a: fnp.ndarray, b: fnp.ndarray) -> fnp.ndarray:
    return fnp.divide(
        fnp.linalg.norm(a - b, ord="fro"),
        fnp.maximum(fnp.linalg.norm(b, ord="fro"), _MIN_VARIANCE),
    )


def _initial_recolor(
    weights: list[fnp.ndarray], pre_half: fnp.ndarray
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Return state-0 full pair rows and original exact Gaussian target moments."""
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
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
    state0 = _strassen_matmul(
        centered.astype(fnp.float32), recolor.astype(fnp.float32), 3
    ) + target_mean.astype(fnp.float32)[None, :]
    return state0.astype(fnp.float32), target_mean, target_cov


def _current_successor_update(
    x: fnp.ndarray, pre: fnp.ndarray
) -> fnp.ndarray:
    pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
    pre_centered = pre - pre_mean[None, :]
    target_var = _gaussian_relu_variance(
        pre_mean,
        fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
    )
    sample_mean = fnp.mean(x, axis=0).astype(fnp.float64)
    centered = x - sample_mean[None, :]
    sample_var = fnp.maximum(
        fnp.mean(centered * centered, axis=0).astype(fnp.float64), _MIN_VARIANCE
    )
    centered_apply = x - sample_mean.astype(fnp.float32)[None, :]
    scale = (
        1.0 + _DEEP_VARIANCE_MATCH_STRENGTH
        * (fnp.sqrt(target_var / sample_var) - 1.0)
    ).astype(fnp.float32)
    return centered_apply * scale[None, :] + sample_mean.astype(fnp.float32)[None, :]


def _prefix_state(
    weights: list[fnp.ndarray], state0: fnp.ndarray
) -> fnp.ndarray:
    """State 0 is post-recolor; W1..W8 inclusive produce the K8 branch."""
    x = state0
    for layer_idx in range(1, PREFIX_LAYERS + 1):
        pre = _strassen_matmul(x, weights[layer_idx], 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            x = _current_successor_update(x, pre).astype(fnp.float32)
    return x.astype(fnp.float32)


def _factorize_odd(v: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    q = v * v
    g = fnp.mean(q)
    r = fnp.mean(q, axis=1) / fnp.maximum(g, _MIN_VARIANCE)
    c = fnp.mean(q, axis=0)
    approx = r[:, None] * c[None, :]
    residual = _rel_fro(approx, q)
    return g, r, c, residual


def _odd_closure(
    weights: list[fnp.ndarray], branch: fnp.ndarray
) -> tuple[fnp.ndarray, dict[str, object]]:
    h_plus = branch[:POSITIVE_ROWS]
    h_minus = branch[POSITIVE_ROWS:]
    even = (h_plus + h_minus) * 0.5
    odd = (h_plus - h_minus) * 0.5
    g, r, c, initial_residual = _factorize_odd(odd)
    initial_g = g
    initial_r_mean = fnp.mean(r)
    initial_c_mean = fnp.mean(c)
    means = []
    layer_diags = []
    e = even.astype(fnp.float32)
    for layer_idx in range(SUFFIX_START, DEPTH):
        weight = weights[layer_idx]
        a = _strassen_matmul(e, weight, 3)
        base = fnp.sum(c[:, None] * (weight * weight), axis=0)
        s2 = fnp.maximum(r[:, None] * base[None, :], _MIN_VARIANCE)
        s = fnp.sqrt(s2)
        alpha = a / s
        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        e_new = s * phi + a * Phi
        second = (a * a + s2) * Phi + a * s * phi
        odd_second_positive = (a * a - s2) * (2.0 * Phi - 1.0) + 2.0 * a * s * phi
        odd_second = fnp.where(a <= 0.0, 0.0, odd_second_positive)
        vo_raw = 0.5 * (second - odd_second)
        vo = fnp.maximum(vo_raw, 0.0)
        g, r, c, residual = _factorize_odd(fnp.sqrt(vo))
        e_new_fp32 = e_new.astype(fnp.float32)
        means.append(fnp.mean(e_new_fp32, axis=0).astype(fnp.float64))
        layer_diags.append({
            "layer": layer_idx,
            "factor_residual": _scalar(residual),
            "g": _scalar(g),
            "s2_min": _scalar(fnp.min(s2)),
            "vo_raw_min": _scalar(fnp.min(vo_raw)),
            "vo_min": _scalar(fnp.min(vo)),
            "finite": _finite(e_new) and _finite(s2) and _finite(vo),
        })
        e = e_new_fp32
    diagnostics = {
        "initial_factor_residual": _scalar(initial_residual),
        "initial_g": _scalar(initial_g),
        "initial_r_mean": _scalar(initial_r_mean),
        "initial_c_mean": _scalar(initial_c_mean),
        "layers": layer_diags,
        "finite": _finite(branch) and all(bool(d["finite"]) for d in layer_diags),
        "s2_positive": all(float(d["s2_min"]) > 0.0 for d in layer_diags),
        "vo_nonnegative": all(float(d["vo_raw_min"]) >= -SERIAL_TOL for d in layer_diags),
    }
    return fnp.stack(means, axis=0).astype(fnp.float64), diagnostics


def _current_suffix(
    weights: list[fnp.ndarray], branch: fnp.ndarray
) -> fnp.ndarray:
    x = branch.astype(fnp.float32)
    means = []
    for layer_idx in range(SUFFIX_START, DEPTH):
        x = fnp.maximum(_strassen_matmul(x, weights[layer_idx], 3), 0.0)
        means.append(fnp.mean(x, axis=0).astype(fnp.float64))
    return fnp.stack(means, axis=0)


def run_one(shard_index: int, bank_path: Path) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights_np = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights_np)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")
    weights = [fnp.array(w, dtype=fnp.float32) for w in weights_np]
    estimated = []
    for rep, stream in enumerate(STREAMS):
        positive = _hadamard_positive(_derived_seed(seed, stream, rep))
        pre_half = _strassen_matmul(positive, weights[0], 3)
        state0, target_mean, target_cov = _initial_recolor(weights, pre_half)
        branch = _prefix_state(weights, state0)
        current_suffix = _current_suffix(weights, branch)
        candidate_suffix, closure_diag = _odd_closure(weights, branch)
        current_vector = current_suffix[-1]
        candidate_vector = candidate_suffix[-1]
        estimated.append({
            "rep": rep,
            "current_vector": current_vector,
            "candidate_vector": candidate_vector,
            "diagnostics": {
                "branch_finite": _finite(branch),
                "closure": closure_diag,
                "target_mean_finite": _finite(target_mean),
                "target_cov_finite": _finite(target_cov),
            },
        })

    # Truth is intentionally read only after all three route vectors and diagnostics are fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    reps = []
    for item in estimated:
        current = item["current_vector"]
        candidate = item["candidate_vector"]
        reps.append({
            "rep": item["rep"],
            "mse": {
                "current": float(np.mean((np.asarray(current) - truth) ** 2)),
                "candidate": float(np.mean((np.asarray(candidate) - truth) ** 2)),
            },
            "current_vector": np.asarray(current).tolist(),
            "candidate_vector": np.asarray(candidate).tolist(),
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
            "width": WIDTH,
            "depth": DEPTH,
            "positive_rows": POSITIVE_ROWS,
            "total_rows": TOTAL_ROWS,
            "blocks": BLOCKS,
            "reps": REPS,
            "streams": list(STREAMS),
            "state_indexing": "state0=post-recolor first ReLU; W1..W8 prefix; branch after W8; suffix W9..W31",
            "prediction_scope": "final W31 mean only; all 23 W9..W31 closure steps computed and diagnosed",
            "pairing": "positive_rows[p] paired with negative_rows[p]",
            "closure": "scalar Gaussian odd-state Rao-Blackwell with rank-one row-coordinate variance carrier",
            "first_successor": "fp64 statistics with fp32 centered apply/scale/writeback, strength 1.5",
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
