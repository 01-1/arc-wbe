#!/usr/bin/env python3
"""Fly payload for the packed gate-clustered two-sided pruning gate.

This is a research-only structural measurement.  It rebuilds one fresh truth-
bank MLP, reproduces the live 16-block estimator activation route, and measures
exact input-union sparsity plus label-free output-dead certificates.  The truth
means in the bank are intentionally never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flopscope.numpy as fnp
from local_engine import build_mlp
from estimator import (
    _DEEP_VARIANCE_MATCH_STRENGTH,
    _MIN_VARIANCE,
    _gaussian_relu_variance,
    _hadamard_sign_half_blocks,
    _strassen_matmul,
    _zero_mean_relu_mean_cov,
)


SCRIPT_VERSION = "packed-gate-cluster-pruning-v2-compact-results"
WIDTH = 256
DEPTH = 32
BLOCKS = 16
N_SAMPLES = BLOCKS * 2 * WIDTH
GROUP_SIZES = (32, 64, 128)
STRATEGIES = ("contiguous", "activation_norm", "pc1", "gatekey32", "support_lex")
PAD_QUANTA = (16, 32, 64)
RANK = 2
POWER_ITERS = 2
FP32_U = 2.0 ** -24
STRASSEN_SLACK_FACTOR = 64.0
BASELINE_RAW_B16 = 25_353_276_460.0
FULL_L3_MATMUL_B16 = 748_176_384.0
REPLACED_LAYERS = 30  # weight layers 2..31 inclusive
PROJECT_BLOCKS = 28
PROJECT_SCALE = PROJECT_BLOCKS / BLOCKS
NONREPLACED_RAW_B16 = BASELINE_RAW_B16 - REPLACED_LAYERS * FULL_L3_MATMUL_B16


def weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _stable_argsort(values: np.ndarray) -> np.ndarray:
    try:
        return np.argsort(values, kind="stable")
    except ValueError:
        return np.argsort(values, kind="mergesort")


def _power_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a cheap deterministic rank-2 activation basis and PC1 score.

    Any orthonormal Q makes the residual certificate rigorous.  Two power
    iterations improve tightness while keeping the projected candidate cost
    explicit and far below a charged 256x256 eigensolve per layer.
    """
    mean = np.mean(x, axis=0, dtype=np.float64)
    centered = x - mean[None, :]
    var = np.mean(centered * centered, axis=0)
    base0 = var + 1e-30
    base1 = base0 * np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float64)
    q, _ = np.linalg.qr(np.column_stack((base0, base1)))
    q = q[:, :RANK]
    for _ in range(POWER_ITERS):
        y = centered @ q
        z = centered.T @ y
        q, _ = np.linalg.qr(z)
        q = q[:, :RANK]
    score = centered @ q[:, 0]
    n, width = x.shape
    one_iteration = n * RANK * (2 * width - 1) + width * RANK * (2 * n - 1)
    basis_flops = POWER_ITERS * one_iteration
    # Initialization variance, QR, and final score are included conservatively.
    basis_flops += n * (2 * width - 1) + 8 * width * RANK * RANK
    return q, score, int(basis_flops)


def _packed_uint_key(bits: np.ndarray) -> np.ndarray:
    width = bits.shape[1]
    powers = np.asarray([np.uint64(1) << np.uint64(width - 1 - idx) for idx in range(width)])
    return bits.astype(np.uint64) @ powers


def _sort_orders(
    x: np.ndarray,
    support: np.ndarray,
    pc1_score: np.ndarray,
) -> dict[str, np.ndarray]:
    n = x.shape[0]
    gate_rate = np.mean(support, axis=0)
    variability = gate_rate * (1.0 - gate_rate)
    variable_order = np.argsort(-variability, kind="stable")

    top = variable_order[:32]
    gate_key = _packed_uint_key(support[:, top])

    chunks = []
    for start in range(0, x.shape[1], 64):
        cols = variable_order[start : start + 64]
        chunks.append(_packed_uint_key(support[:, cols]))
    # np.lexsort uses its final key as the primary key; chunk 0 contains the
    # most variable support coordinates and is intentionally primary.
    support_order = np.lexsort(tuple(chunks[::-1]))

    return {
        "contiguous": np.arange(n, dtype=np.int64),
        "activation_norm": _stable_argsort(np.sum(x * x, axis=1)),
        "pc1": _stable_argsort(pc1_score),
        "gatekey32": _stable_argsort(gate_key),
        "support_lex": support_order,
    }


def _sort_flops(strategy: str, n: int, width: int, basis_flops: int, combined: bool) -> int:
    cost = basis_flops if combined or strategy == "pc1" else 0
    if strategy == "activation_norm":
        cost += n * (2 * width - 1)
    return int(cost)


def _box_screen_flops(live_counts: np.ndarray, width: int) -> int:
    # Two interval endpoint products plus a third absolute-envelope product,
    # with conservative arithmetic for bound formation and slack.
    k = live_counts.astype(np.int64)
    dot = np.where(k > 0, 2 * k - 1, 0)
    return int(np.sum(3 * width * dot + 6 * width))


def _low_rank_screen_flops(n: int, width: int, groups: int, basis_flops: int) -> int:
    rank = RANK
    cost = basis_flops
    cost += n * width + groups * width  # means and division
    cost += n * rank * (2 * width - 1)  # Y=(X-mu)Q
    cost += n * width * (2 * rank - 1)  # YQ^T
    cost += 4 * n * width               # residual/subtract/square/reduce/sqrt
    cost += rank * width * (2 * width - 1)  # Q^T W
    cost += width * width * (2 * rank - 1)  # QQ^T W / residual weights
    cost += n * width * (2 * rank - 1)      # Y(Q^T W)
    cost += groups * width * (2 * width - 1)  # mu^T W
    cost += 2 * n * width
    return int(cost)


def _packing_metrics(
    live_counts: np.ndarray,
    certified_counts: np.ndarray,
    group_size: int,
    quantum: int,
) -> dict[str, Any]:
    output_counts = WIDTH - certified_counts.astype(np.int64)
    live_counts = live_counts.astype(np.int64)
    kpad = ((live_counts + quantum - 1) // quantum) * quantum
    opad = ((output_counts + quantum - 1) // quantum) * quantum
    active = (live_counts > 0) & (output_counts > 0)
    kpad = np.where(active, kpad, 0)
    opad = np.where(active, opad, 0)

    matmul_flops = int(np.sum(np.where(active, group_size * opad * (2 * kpad - 1), 0)))
    bucket_counter = Counter(
        (int(k), int(o))
        for k, o, keep in zip(kpad.tolist(), opad.tolist(), active.tolist())
        if keep
    )
    groups_per_bucket = list(bucket_counter.values())
    peak_bytes = 0
    total_bytes = 0
    for (k, o), count in bucket_counter.items():
        elements = count * (group_size * k + k * o + group_size * o)
        total_bytes += 4 * elements
        peak_bytes = max(peak_bytes, 4 * elements)

    activation_elements = int(np.sum(group_size * (live_counts + output_counts)))
    weight_elements = int(np.sum(live_counts * output_counts))
    dense_activation_elements = int(live_counts.shape[0] * group_size * WIDTH)
    return {
        "matmul_flops": matmul_flops,
        "bucket_count": len(bucket_counter),
        "groups_per_bucket_median": float(np.median(groups_per_bucket)) if groups_per_bucket else 0.0,
        "groups_per_bucket_min": int(min(groups_per_bucket)) if groups_per_bucket else 0,
        "peak_bucket_bytes": int(peak_bytes),
        "total_packed_bytes": int(total_bytes),
        "activation_traffic_ratio": activation_elements / max(dense_activation_elements, 1),
        "weight_traffic_ratio": weight_elements / max(dense_activation_elements, 1),
        "total_traffic_ratio": (activation_elements + weight_elements) / max(dense_activation_elements, 1),
    }


def _measure_groups(
    x_sorted: np.ndarray,
    pre_sorted: np.ndarray,
    x_perp_sorted: np.ndarray,
    projected_w_sorted: np.ndarray,
    w: np.ndarray,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    w_abs: np.ndarray,
    w_residual_norm: np.ndarray,
    group_size: int,
    strategy: str,
    basis_flops: int,
) -> dict[str, Any]:
    n, width = x_sorted.shape
    groups = n // group_size
    xs = x_sorted.reshape(groups, group_size, width)
    ps = pre_sorted.reshape(groups, group_size, width)
    x_perp_groups = x_perp_sorted.reshape(groups, group_size, width)
    projected_w_groups = projected_w_sorted.reshape(groups, group_size, width)

    live_mask = np.any(xs != 0.0, axis=1)
    live_counts = np.sum(live_mask, axis=1).astype(np.int64)
    lower = np.min(xs, axis=1)
    upper = np.max(xs, axis=1)
    box_upper = upper @ w_pos + lower @ w_neg
    max_abs = np.maximum(np.abs(lower), np.abs(upper))
    abs_envelope = max_abs @ w_abs
    gamma = np.where(
        live_counts > 0,
        (live_counts * FP32_U) / np.maximum(1.0 - live_counts * FP32_U, 1e-30),
        0.0,
    )
    slack = STRASSEN_SLACK_FACTOR * gamma[:, None] * abs_envelope
    slack += 16.0 * np.finfo(np.float64).eps * (abs_envelope + np.abs(box_upper) + 1.0)
    slack = np.nextafter(slack, np.inf)
    cert_box = box_upper + slack <= 0.0

    mu_perp = np.mean(x_perp_groups, axis=1, dtype=np.float64)
    residual = x_perp_groups - mu_perp[:, None, :]
    residual_radius = np.linalg.norm(residual, axis=2)
    residual_radius = np.nextafter(residual_radius * (1.0 + 64.0 * np.finfo(np.float64).eps), np.inf)
    residual_upper = residual_radius[:, :, None] * w_residual_norm[None, None, :]
    low_rank_upper = mu_perp @ w + np.max(projected_w_groups + residual_upper, axis=1)
    cert_low_rank = low_rank_upper + slack <= 0.0
    cert_combined = cert_box | cert_low_rank

    true_dead = np.max(ps, axis=1) <= 0.0
    box_violations = cert_box & ~true_dead
    low_rank_violations = cert_low_rank & ~true_dead
    combined_violations = cert_combined & ~true_dead
    true_count = int(np.sum(true_dead))

    result: dict[str, Any] = {
        "live_input_mean": float(np.mean(live_counts / width)),
        "live_input_q90": float(np.quantile(live_counts / width, 0.90)),
        "true_output_dead_mean": float(np.mean(np.sum(true_dead, axis=1) / width)),
        "live_counts": live_counts,
    }

    for name, cert, violations in (
        ("box", cert_box, box_violations),
        ("combined", cert_combined, combined_violations),
    ):
        cert_counts = np.sum(cert, axis=1).astype(np.int64)
        product = (live_counts / width) * (1.0 - cert_counts / width)
        result[name] = {
            "certified_output_dead_mean": float(np.mean(cert_counts / width)),
            "certificate_recall": float(np.sum(cert & true_dead) / max(true_count, 1)),
            "violations": int(np.sum(violations)),
            "product_mean": float(np.mean(product)),
            "product_q90": float(np.quantile(product, 0.90)),
            "certified_counts": cert_counts,
            "packing": {},
        }
        for quantum in PAD_QUANTA:
            result[name]["packing"][str(quantum)] = _packing_metrics(
                live_counts,
                cert_counts,
                group_size,
                quantum,
            )

    result["low_rank_only_violations"] = int(np.sum(low_rank_violations))
    result["box_screen_flops"] = _box_screen_flops(live_counts, width)
    result["low_rank_screen_flops"] = _low_rank_screen_flops(n, width, groups, basis_flops)
    result["sort_flops_box"] = _sort_flops(strategy, n, width, basis_flops, combined=False)
    result["sort_flops_combined"] = _sort_flops(strategy, n, width, basis_flops, combined=True)
    return result


def _empty_combo(strategy: str, group_size: int) -> dict[str, Any]:
    combo: dict[str, Any] = {
        "strategy": strategy,
        "group_size": group_size,
        "layers": [],
        "live_input_mean": [],
        "true_output_dead_mean": [],
        "box": {
            "certified_output_dead_mean": [],
            "certificate_recall": [],
            "violations": [],
            "product_mean": [],
            "product_q90": [],
            "packing": {str(q): [] for q in PAD_QUANTA},
        },
        "combined": {
            "certified_output_dead_mean": [],
            "certificate_recall": [],
            "violations": [],
            "product_mean": [],
            "product_q90": [],
            "packing": {str(q): [] for q in PAD_QUANTA},
        },
        "low_rank_only_violations": [],
        "box_screen_flops": [],
        "low_rank_screen_flops": [],
        "sort_flops_box": [],
        "sort_flops_combined": [],
    }
    return combo


def _append_measurement(combo: dict[str, Any], layer_idx: int, measured: dict[str, Any]) -> None:
    combo["layers"].append(layer_idx)
    combo["live_input_mean"].append(measured["live_input_mean"])
    combo["true_output_dead_mean"].append(measured["true_output_dead_mean"])
    combo["low_rank_only_violations"].append(measured["low_rank_only_violations"])
    combo["box_screen_flops"].append(measured["box_screen_flops"])
    combo["low_rank_screen_flops"].append(measured["low_rank_screen_flops"])
    combo["sort_flops_box"].append(measured["sort_flops_box"])
    combo["sort_flops_combined"].append(measured["sort_flops_combined"])
    for cert_name in ("box", "combined"):
        target = combo[cert_name]
        source = measured[cert_name]
        for field in (
            "certified_output_dead_mean",
            "certificate_recall",
            "violations",
            "product_mean",
            "product_q90",
        ):
            target[field].append(source[field])
        for quantum in PAD_QUANTA:
            target["packing"][str(quantum)].append(source["packing"][str(quantum)])


def _finalize_combo(combo: dict[str, Any]) -> None:
    layers = np.asarray(combo["layers"], dtype=np.int64)
    decision = layers >= 3
    combo["decision_layers"] = [3, 31]
    combo["decision_live_input_mean"] = float(np.mean(np.asarray(combo["live_input_mean"])[decision]))
    combo["decision_true_output_dead_mean"] = float(
        np.mean(np.asarray(combo["true_output_dead_mean"])[decision])
    )

    for cert_name in ("box", "combined"):
        cert = combo[cert_name]
        cert["decision_product_mean"] = float(np.mean(np.asarray(cert["product_mean"])[decision]))
        cert["decision_certificate_recall"] = float(
            np.mean(np.asarray(cert["certificate_recall"])[decision])
        )
        cert["total_violations"] = int(np.sum(cert["violations"]))
        packed_projection = {}
        for quantum in PAD_QUANTA:
            qkey = str(quantum)
            rows = cert["packing"][qkey]
            matmul_total = sum(int(row["matmul_flops"]) for row in rows)
            box_total = sum(int(value) for value in combo["box_screen_flops"])
            if cert_name == "box":
                screen_total = box_total
                sort_total = sum(int(value) for value in combo["sort_flops_box"])
            else:
                screen_total = box_total + sum(int(value) for value in combo["low_rank_screen_flops"])
                sort_total = sum(int(value) for value in combo["sort_flops_combined"])
            packed_total_b16 = matmul_total + screen_total + sort_total
            projected_raw = (NONREPLACED_RAW_B16 + packed_total_b16) * PROJECT_SCALE
            bucket_counts = np.asarray([row["bucket_count"] for row in rows], dtype=np.float64)
            group_medians = np.asarray([row["groups_per_bucket_median"] for row in rows], dtype=np.float64)
            activation_traffic = np.asarray([row["activation_traffic_ratio"] for row in rows])
            total_traffic = np.asarray([row["total_traffic_ratio"] for row in rows])
            packed_projection[qkey] = {
                "matmul_flops_b16": int(matmul_total),
                "screen_flops_b16": int(screen_total),
                "sort_flops_b16": int(sort_total),
                "packed_total_b16": int(packed_total_b16),
                "projected_raw_b28": float(projected_raw),
                "bucket_count_mean": float(np.mean(bucket_counts)),
                "bucket_count_max": int(np.max(bucket_counts)),
                "groups_per_bucket_median": float(np.median(group_medians)),
                "peak_bucket_bytes_b16": int(max(row["peak_bucket_bytes"] for row in rows)),
                "projected_peak_bucket_bytes_b28": int(
                    math.ceil(PROJECT_SCALE * max(row["peak_bucket_bytes"] for row in rows))
                ),
                "activation_traffic_ratio_mean": float(np.mean(activation_traffic)),
                "activation_traffic_ratio_max": float(np.max(activation_traffic)),
                "total_traffic_ratio_mean": float(np.mean(total_traffic)),
                "total_traffic_ratio_max": float(np.max(total_traffic)),
            }
        cert["projection"] = packed_projection

    # Fly transports results through log chunks.  The per-layer packed rows
    # above are useful while computing projections but are redundant after
    # aggregation: retain the per-layer decision curves and the compact
    # projected summaries, while dropping one dict per layer/quantum that
    # would make a single shard's JSON too large for reliable reconstruction.
    for cert_name in ("box", "combined"):
        cert = combo[cert_name]
        cert.pop("packing", None)
    for field in (
        "box_screen_flops",
        "low_rank_screen_flops",
        "sort_flops_box",
        "sort_flops_combined",
    ):
        combo.pop(field, None)
    combo["low_rank_only_violations_total"] = int(np.sum(combo["low_rank_only_violations"]))
    combo.pop("low_rank_only_violations", None)


def run_one(shard_index: int, bank_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    bank = np.load(bank_path)
    seeds = bank["seeds"]
    checksums = bank["weights_sha256"]
    seed = int(seeds[shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    checksum = weights_sha256(mlp.weights)
    expected_checksum = str(checksums[shard_index])
    if checksum != expected_checksum:
        raise ValueError(f"weight checksum mismatch for bank index {shard_index}")

    weights_f32 = [weight.astype(fnp.float32) for weight in mlp.weights]
    rng = fnp.random.default_rng(mlp.seed)
    x_half = _hadamard_sign_half_blocks(mlp, N_SAMPLES, rng).astype(fnp.float32)
    pre_half = _strassen_matmul(x_half, weights_f32[0], 3)
    y = fnp.concatenate((fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)), axis=0)

    w0 = mlp.weights[0]
    target_mean, target_cov = _zero_mean_relu_mean_cov(w0.T @ w0)
    sample_mean = fnp.mean(y, axis=0).astype(fnp.float64)
    centered = y - sample_mean[None, :]
    sample_cov = (
        _strassen_matmul(centered.T.astype(fnp.float32), centered.astype(fnp.float32), 3)
        / float(centered.shape[0])
    ).astype(fnp.float64)
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = fnp.eye(WIDTH)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered.astype(fnp.float32), recolor.astype(fnp.float32), 3)
    x = x + target_mean.astype(fnp.float32)[None, :]

    combos = {
        f"{strategy}:g{group_size}": _empty_combo(strategy, group_size)
        for strategy in STRATEGIES
        for group_size in GROUP_SIZES
    }

    for layer_idx, w_prop in enumerate(weights_f32[1:], start=1):
        pre = _strassen_matmul(x, w_prop, 3)
        if layer_idx >= 2:
            x_np = np.asarray(x, dtype=np.float64)
            pre_np = np.asarray(pre, dtype=np.float64)
            w_np = np.asarray(w_prop, dtype=np.float64)
            support = x_np > 0.0
            q, pc1_score, basis_flops = _power_basis(x_np)
            orders = _sort_orders(x_np, support, pc1_score)
            xq = x_np @ q
            x_qqt = xq @ q.T
            x_perp = x_np - x_qqt
            qtw = q.T @ w_np
            projected_w = xq @ qtw
            w_residual = w_np - q @ qtw
            w_residual_norm = np.nextafter(np.linalg.norm(w_residual, axis=0), np.inf)
            w_pos = np.maximum(w_np, 0.0)
            w_neg = np.minimum(w_np, 0.0)
            w_abs = np.abs(w_np)
            for strategy in STRATEGIES:
                order = orders[strategy]
                x_sorted = np.ascontiguousarray(x_np[order])
                pre_sorted = np.ascontiguousarray(pre_np[order])
                x_perp_sorted = np.ascontiguousarray(x_perp[order])
                projected_w_sorted = np.ascontiguousarray(projected_w[order])
                for group_size in GROUP_SIZES:
                    measured = _measure_groups(
                        x_sorted,
                        pre_sorted,
                        x_perp_sorted,
                        projected_w_sorted,
                        w_np,
                        w_pos,
                        w_neg,
                        w_abs,
                        w_residual_norm,
                        group_size,
                        strategy,
                        basis_flops,
                    )
                    _append_measurement(
                        combos[f"{strategy}:g{group_size}"],
                        layer_idx,
                        measured,
                    )
        x_next = fnp.maximum(pre, 0.0)
        if layer_idx == 1:
            pre_mean = fnp.mean(pre, axis=0).astype(fnp.float64)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(
                pre_mean,
                fnp.mean(pre_centered * pre_centered, axis=0).astype(fnp.float64),
            )
            successor_mean = fnp.mean(x_next, axis=0).astype(fnp.float64)
            successor_centered = x_next - successor_mean[None, :]
            sample_var = fnp.maximum(
                fnp.mean(successor_centered * successor_centered, axis=0).astype(fnp.float64),
                _MIN_VARIANCE,
            )
            scale = 1.0 + _DEEP_VARIANCE_MATCH_STRENGTH * (
                fnp.sqrt(target_var / sample_var) - 1.0
            )
            centered_apply = x_next - successor_mean.astype(fnp.float32)[None, :]
            x_next = (
                centered_apply * scale.astype(fnp.float32)[None, :]
                + successor_mean.astype(fnp.float32)[None, :]
            )
        x = x_next

    for combo in combos.values():
        _finalize_combo(combo)

    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": True,
        "weights_sha256": checksum,
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "blocks": BLOCKS,
            "n_samples": N_SAMPLES,
            "group_sizes": GROUP_SIZES,
            "strategies": STRATEGIES,
            "pad_quanta": PAD_QUANTA,
            "rank": RANK,
            "power_iterations": POWER_ITERS,
            "strassen_slack_factor": STRASSEN_SLACK_FACTOR,
            "project_blocks": PROJECT_BLOCKS,
            "baseline_raw_b16": BASELINE_RAW_B16,
            "nonreplaced_raw_b16": NONREPLACED_RAW_B16,
        },
        "combos": combos,
        "wall_time_s": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    result = run_one(args.shard_index, args.bank)
    text = json.dumps(result, sort_keys=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
