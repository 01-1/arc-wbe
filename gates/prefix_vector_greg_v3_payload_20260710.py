#!/usr/bin/env python3
"""Prefix vector-GREG v3 research payload (truth delayed)."""

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

import flopscope as flops  # noqa: E402
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
POOL_BLOCKS = 32
PILOT_BLOCKS = 2
MAIN_BLOCKS = 30
PAIRS_PER_BLOCK = WIDTH
N_PAIRS = POOL_BLOCKS * PAIRS_PER_BLOCK
N_PILOT = PILOT_BLOCKS * PAIRS_PER_BLOCK
N_MAIN = MAIN_BLOCKS * PAIRS_PER_BLOCK
KS = (4, 6, 8)
TERMINAL_BLOCKS = {4: 11, 6: 9, 8: 8}
N_SELECTED = {k: (b - PILOT_BLOCKS) * PAIRS_PER_BLOCK for k, b in TERMINAL_BLOCKS.items()}
CURRENT_BLOCKS = 16
CANDIDATE_STREAM = 0xA551_0710
CURRENT_STREAM = 0xC0A1_0710
SCRIPT_VERSION = "prefix-vector-greg-v3"


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _seed(seed: int, stream: int, rep: int = 0) -> int:
    return int((seed ^ stream ^ (rep * 0x9E3779B9)) % (1 << 32))


def _independent_positive_bases(seed: int, blocks: int, stream: int) -> np.ndarray:
    rng = np.random.default_rng(_seed(seed, stream))
    base = np.asarray(_hadamard(WIDTH), dtype=np.float32)
    rows = []
    for _ in range(blocks):
        flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=WIDTH)
        rows.append(base * flips[None, :])
    return np.concatenate(rows, axis=0)


def _first_layer_recolor(weights: list[fnp.ndarray], positive: np.ndarray) -> fnp.ndarray:
    x_half = fnp.array(positive, dtype=fnp.float32)
    pre_half = _strassen_matmul(x_half, weights[0], 3)
    y = fnp.concatenate((fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0)
    target_mean, target_cov = _zero_mean_relu_mean_cov(weights[0].T @ weights[0])
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.astype(fnp.float32).T, centered.astype(fnp.float32), 3)
        / float(centered.shape[0])
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(WIDTH)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    centered = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    return centered + target_mean.astype(fnp.float32)[None, :]


def _first_successor_match(x: fnp.ndarray, pre: fnp.ndarray) -> fnp.ndarray:
    pre_mean = fnp.mean(pre, axis=0).astype(fnp.float32)
    pre_centered = pre - pre_mean[None, :]
    target_var = _gaussian_relu_variance(
        pre_mean.astype(fnp.float64),
        fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
    )
    sample_mean = fnp.mean(x, axis=0).astype(fnp.float32)
    centered_apply = x - sample_mean[None, :]
    sample_var = fnp.maximum(
        fnp.mean(centered_apply * centered_apply, axis=0).astype(fnp.float64),
        _MIN_VARIANCE,
    )
    scale = (
        1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (fnp.sqrt(target_var / sample_var) - 1.0)
    ).astype(fnp.float32)
    return centered_apply * scale[None, :] + sample_mean[None, :]


def _pair_fold(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0] // 2
    return 0.5 * (raw[:n] + raw[n:])


def _prefix_pool(mlp, seed: int) -> tuple[list[fnp.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    weights = [fnp.array(w, dtype=fnp.float32) for w in mlp.weights]
    positive = _independent_positive_bases(seed, POOL_BLOCKS, CANDIDATE_STREAM)
    x = _first_layer_recolor(weights, positive)
    raw_by_k: dict[int, np.ndarray] = {}
    folded_by_k: dict[int, np.ndarray] = {}
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            x = _first_successor_match(x, pre)
        k = layer_idx + 1
        if k in KS:
            raw = np.asarray(x, dtype=np.float64)
            raw_by_k[k] = raw
            folded_by_k[k] = _pair_fold(raw)
        if layer_idx == max(KS) - 1:
            break
    return weights, raw_by_k, folded_by_k


def _suffix_final(raw: np.ndarray, weights: list[fnp.ndarray], k: int) -> np.ndarray:
    return _pair_fold(_suffix_individual(raw, weights, k))


def _suffix_individual(raw: np.ndarray, weights: list[fnp.ndarray], k: int) -> np.ndarray:
    x = fnp.array(raw, dtype=fnp.float32)
    for weight in weights[k:]:
        x = fnp.maximum(_strassen_matmul(x, weight, 3), 0.0)
    return np.asarray(x, dtype=np.float64)


def _full_route_mean(mlp, seed: int, blocks: int) -> np.ndarray:
    weights = [fnp.array(w, dtype=fnp.float32) for w in mlp.weights]
    positive = _independent_positive_bases(seed, blocks, CURRENT_STREAM)
    x = _first_layer_recolor(weights, positive)
    for layer_idx, weight in enumerate(weights[1:], start=1):
        pre = _strassen_matmul(x, weight, 3)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            x = _first_successor_match(x, pre)
    return np.asarray(fnp.mean(x, axis=0), dtype=np.float64)


def _fit_vector_predict(pilot_x: np.ndarray, pilot_y: np.ndarray, main_x: np.ndarray) -> tuple[np.ndarray, float, float]:
    means = np.mean(pilot_x, axis=0)
    scales = np.maximum(np.std(pilot_x, axis=0), np.sqrt(_MIN_VARIANCE))
    pilot_z = (pilot_x - means[None, :]) / scales[None, :]
    main_z = (main_x - means[None, :]) / scales[None, :]
    centered_x = pilot_z - np.mean(pilot_z, axis=0)[None, :]
    centered_y = pilot_y - np.mean(pilot_y, axis=0)[None, :]
    gram = centered_x.T @ centered_x
    ridge = 0.1 * float(np.trace(gram)) / WIDTH
    beta = np.linalg.solve(gram + ridge * np.eye(WIDTH), centered_x.T @ centered_y)
    pred = np.mean(pilot_y, axis=0)[None, :] + (main_z - np.mean(pilot_z, axis=0)[None, :]) @ beta
    return pred, ridge, float(np.linalg.cond(gram + ridge * np.eye(WIDTH)))


def _select_stratified(pred: np.ndarray, seed: int, k: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    order = np.argsort(pred, kind="stable")
    n_selected = N_SELECTED[k]
    strata = np.array_split(order, n_selected)
    rng = np.random.default_rng(_seed(seed, CANDIDATE_STREAM ^ (k * 0x13579BDF)))
    selected = np.array([int(s[rng.integers(len(s))]) for s in strata], dtype=np.int64)
    sizes = np.array([len(s) for s in strata], dtype=np.float64)
    return selected, sizes / float(N_MAIN), [len(s) for s in strata]


def _candidate_for_k(
    seed: int,
    k: int,
    weights: list[fnp.ndarray],
    raw: np.ndarray,
) -> dict[str, object]:
    # Individual pilot/main rows are paired only after the multi-output fit.
    pilot_raw = np.concatenate((raw[:N_PILOT], raw[N_PAIRS : N_PAIRS + N_PILOT]), axis=0)
    main_raw = np.concatenate((raw[N_PILOT:N_PAIRS], raw[N_PAIRS + N_PILOT:]), axis=0)
    pilot_x = pilot_raw
    main_x = main_raw
    pilot_y = _suffix_individual(pilot_raw, weights, k)
    main_pred_individual, ridge, condition = _fit_vector_predict(pilot_x, pilot_y, main_x)
    main_pred_pair = _pair_fold(main_pred_individual)
    pilot_final = _pair_fold(pilot_y)
    pilot_mean = np.mean(pilot_final, axis=0)
    centered_pilot_final = pilot_final - pilot_mean[None, :]
    _, _, vh = np.linalg.svd(centered_pilot_final, full_matrices=False)
    pc = vh[0]
    if pc[np.argmax(np.abs(pc))] < 0.0:
        pc = -pc
    pilot_pc_share = float(np.var(centered_pilot_final @ pc) / max(np.sum(np.var(centered_pilot_final, axis=0)), _MIN_VARIANCE))
    scores = (main_pred_pair - np.mean(main_pred_pair, axis=0)[None, :]) @ pc
    selected_local, inclusion_weights, stratum_sizes = _select_stratified(scores, seed, k)
    selected_global = N_PILOT + selected_local
    # Indices and weights are fully materialized before selected suffix reads.
    selected_rows = np.concatenate((raw[selected_global], raw[selected_global + N_PAIRS]), axis=0)
    selected_final = _suffix_final(selected_rows, weights, k)
    pbar_main = np.mean(main_pred_pair, axis=0)
    selected_predictor = main_pred_pair[selected_local]
    selected_residual = selected_final - selected_predictor
    main_residual = np.sum(selected_residual * inclusion_weights[:, None], axis=0)
    pilot_weight = 2.0 / 32.0
    candidate = pilot_weight * pilot_mean + (1.0 - pilot_weight) * (pbar_main + main_residual)
    raw_main = np.sum(selected_final * inclusion_weights[:, None], axis=0)
    raw_rank = pilot_weight * pilot_mean + (1.0 - pilot_weight) * raw_main

    rng_control = np.random.default_rng(_seed(seed, CANDIDATE_STREAM ^ (k * 0x2468ACE1)))
    control_local = rng_control.choice(N_MAIN, size=N_SELECTED[k], replace=False)
    control_global = N_PILOT + control_local
    control_rows = np.concatenate((raw[control_global], raw[control_global + N_PAIRS]), axis=0)
    control_final = _suffix_final(control_rows, weights, k)
    control_predictor = main_pred_pair[control_local]
    control_pbar = pbar_main
    control_residual = np.mean(control_final - control_predictor, axis=0)
    control = pilot_weight * pilot_mean + (1.0 - pilot_weight) * (control_pbar + control_residual)
    selected_corr = float(np.corrcoef(scores[selected_local], selected_final @ pc)[0, 1]) if len(selected_local) > 1 else 0.0
    regression_flops = {
        "pilot_xtx": float(2 * (2 * N_PILOT) * WIDTH * WIDTH),
        "pilot_xty": float(2 * (2 * N_PILOT) * WIDTH * WIDTH),
        "rhs_factorization_and_triangular_solves": float(8.0 / 3.0 * WIDTH**3),
        "main_predictor_matmul": float(2 * (2 * N_MAIN) * WIDTH * WIDTH),
        "pilot_final_svd_top_pc": float(2 * N_PILOT * WIDTH * WIDTH),
        "standardize_pca_sort_reduce": float((2 * N_MAIN) * WIDTH * 8 + N_MAIN * np.log2(max(N_MAIN, 2)) + N_MAIN * WIDTH),
    }
    regression_flops["total"] = float(sum(regression_flops.values()))
    dense_work = POOL_BLOCKS * k + TERMINAL_BLOCKS[k] * (32 - k)
    projected = float(2.535e10 / (CURRENT_BLOCKS * 32) * dense_work + regression_flops["total"])
    return {
        "k": k,
        "candidate": candidate,
        "raw_rank": raw_rank,
        "control": control,
        "selected_count": int(N_SELECTED[k]),
        "pilot_weight": float(pilot_weight),
        "ridge_lambda": ridge,
        "ridge_condition": condition,
        "selected_predictor_corr": selected_corr if np.isfinite(selected_corr) else 0.0,
        "main_predictor_corr": None,
        "_pred_main": main_pred_pair,
        "_pc": pc,
        "_regression_flops": regression_flops,
        "pilot_pc_share": pilot_pc_share,
        "stratum_size_min": min(stratum_sizes),
        "stratum_size_max": max(stratum_sizes),
        "compute_dense_ratio": float(dense_work / (CURRENT_BLOCKS * 32)),
        "projected_raw_flops": projected,
    }


def run_one(shard_index: int, bank_path: Path, reps: int) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights_np = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights_np)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")

    fixed_reps = []
    for rep in range(reps):
        route_seed = seed ^ (rep * 0x9E3779B9)
        weights, raw_by_k, folded_by_k = _prefix_pool(mlp, route_seed)
        candidates = {
            k: _candidate_for_k(route_seed, k, weights, raw_by_k[k])
            for k in KS
        }
        # Delayed counterfactual boundary: only after every candidate index,
        # predictor mean, stratum, and inclusion weight is fixed do we read
        # unselected main suffix outputs for the full-main correlation/ceiling.
        for k, result in candidates.items():
            main_rows = np.concatenate((raw_by_k[k][N_PILOT:N_PAIRS], raw_by_k[k][N_PAIRS + N_PILOT:]), axis=0)
            main_final = _suffix_final(main_rows, weights, k)
            prediction = result["_pred_main"]
            residual = main_final - prediction
            centered_actual = main_final - np.mean(main_final, axis=0)[None, :]
            result["full_main_r2"] = float(1.0 - np.sum(residual * residual) / max(np.sum(centered_actual * centered_actual), _MIN_VARIANCE))
            result["full_main_residual_fraction"] = float(1.0 - result["full_main_r2"])
            result["main_predictor_corr"] = float(np.corrcoef((prediction - np.mean(prediction, axis=0)[None, :]) @ result["_pc"], (main_final - np.mean(main_final, axis=0)[None, :]) @ result["_pc"])[0, 1]) if len(main_final) > 1 else 0.0
        current = _full_route_mean(mlp, route_seed, CURRENT_BLOCKS)
        full_rows = np.concatenate((raw_by_k[8][:N_PAIRS], raw_by_k[8][N_PAIRS:]), axis=0)
        full_pool_mean = np.mean(_suffix_final(full_rows, weights, 8), axis=0)
        fixed_reps.append({"rep": rep, "candidates": candidates, "current": current, "full_pool": full_pool_mean})

    # Truth access is deliberately after all candidate/control/current vectors are fixed.
    truth = np.asarray(bank["truths"][shard_index, -1], dtype=np.float64)
    bias_proxy = {}
    if reps == 3:
        for k in KS:
            mean_candidate = np.mean(
                np.stack([np.asarray(f["candidates"][k]["candidate"], dtype=np.float64) for f in fixed_reps], axis=0),
                axis=0,
            )
            bias_proxy[k] = float(np.mean((mean_candidate - truth) ** 2))
    rep_results = []
    for fixed in fixed_reps:
        k_results = {}
        for k, result in fixed["candidates"].items():
            candidate = np.asarray(result["candidate"], dtype=np.float64)
            control = np.asarray(result["control"], dtype=np.float64)
            raw_rank = np.asarray(result["raw_rank"], dtype=np.float64)
            current = np.asarray(fixed["current"], dtype=np.float64)
            k_results[str(k)] = {
                key: value for key, value in result.items() if key not in {"candidate", "control", "raw_rank", "_pred_main", "_pc", "_regression_flops"}
            }
            k_results[str(k)].update(
                {
                    "current_mse": float(np.mean((current - truth) ** 2)),
                    "candidate_mse": float(np.mean((candidate - truth) ** 2)),
                    "control_mse": float(np.mean((control - truth) ** 2)),
                    "raw_rank_mse": float(np.mean((raw_rank - truth) ** 2)),
                    "full_pool_ceiling_mse": float(np.mean((np.asarray(fixed["full_pool"]) - truth) ** 2)),
                    "regression_flops": result["_regression_flops"],
                    "three_rep_bias_proxy": bias_proxy.get(k),
                }
            )
        rep_results.append({"rep": fixed["rep"], "k_results": k_results})
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
            "pool_blocks": POOL_BLOCKS,
            "pilot_blocks": PILOT_BLOCKS,
            "main_blocks": MAIN_BLOCKS,
            "pairs": N_PAIRS,
            "rows": 2 * N_PAIRS,
            "ks": KS,
            "terminal_blocks": TERMINAL_BLOCKS,
            "selected_pairs": N_SELECTED,
            "reps": reps,
            "hadamard_bases": "32 independent full positive bases",
            "antipodes": "exact",
            "first_layer": "exact global ReLU mean/covariance recolor",
            "first_successor": "fp32 centered_apply strength 1.5",
            "propagation": "fp32 Strassen level 3",
            "ridge": "centered pilot-only, lambda=0.1*trace(Xt.T@Xt)/256",
            "strata": "stable predicted-score contiguous near-equal, one per stratum",
        },
        "rep_results": rep_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--reps", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank, args.reps)
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
