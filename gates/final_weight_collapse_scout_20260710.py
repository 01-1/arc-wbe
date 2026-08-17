#!/usr/bin/env python3
"""Read-only oracle-ceiling scout for final-weight cross-output smoothing.

This intentionally uses bank truths[idx, -2] as an illegal optimistic feature
anchor. It is a ceiling closeout, not an estimator candidate or promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_engine import build_mlp


PREFIX = "final_weight_collapse_scout_20260710"
FLY_JSONL = Path("paired_fly_logs/fingerprint_theory/terminal_mixture_readout_gate_20260710_fly.jsonl")
BANK_PATH = Path("analysis/truth_bank/truth_bank.npz")
RESULT_PATH = Path(f"paired_fly_logs/fingerprint_theory/{PREFIX}.json")
REPORT_PATH = Path(f"paired_fly_logs/fingerprint_theory/{PREFIX}.md")
N_MLPS = 100
N_REPS = 3
N_OUTPUTS = 256
N_FOLDS = 4
NN_K = 16
TINY = 1e-12
BASELINE_MATCH_RTOL = 1e-9
BASELINE_MATCH_ATOL = 1e-14

# Freeze this rule before consuming any computed smoother result.
CLOSE_BLEND_GAIN_LIMIT = 1.15
CLOSE_BLEND_MSE_FLOOR = 2.2e-6
CLOSE_ORACLE_CEILING_FLOOR = 1.6e-6
SURVIVE_BLEND_MSE = 1.8e-6
SURVIVE_BLEND_GAIN = 1.35
SURVIVE_ORACLE_CEILING = 1.2e-6
CLOSEOUT_RULE = {
    "close_if_every_noisy_smoother": (
        "baseline_M1/blend_M1 < 1.15 OR blend_M1 > 2.2e-6"
    ),
    "or_if_every_oracle_ceiling": "truth-response held-out M1 > 1.6e-6",
    "survives_if_any": (
        "noisy blend_M1 <= 1.8e-6 AND gain >= 1.35 AND "
        "truth-response held-out M1 <= 1.2e-6"
    ),
    "otherwise": "INCONCLUSIVE",
}


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.maximum(scale, TINY)
    return (train - mean[None, :]) / scale[None, :], (test - mean[None, :]) / scale[None, :]


def _polynomial_features(x: np.ndarray) -> np.ndarray:
    columns = []
    for degree in range(4):
        for indices in itertools.combinations_with_replacement(range(x.shape[1]), degree):
            if not indices:
                columns.append(np.ones(x.shape[0], dtype=np.float64))
            else:
                columns.append(np.prod(x[:, indices], axis=1))
    return np.column_stack(columns)


def _predict_polynomial(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_z, test_z = _standardize(train_x, test_x)
    x_train = _polynomial_features(train_z)
    x_test = _polynomial_features(test_z)
    p = x_train.shape[1]
    ridge = 0.1 * float(np.trace(x_train.T @ x_train)) / p
    beta = np.linalg.solve(x_train.T @ x_train + ridge * np.eye(p), x_train.T @ train_y)
    return x_test @ beta


def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    delta = x[:, None, :] - y[None, :, :]
    distances2 = np.sum(delta * delta, axis=2)
    return np.exp(-0.5 * distances2 / max(bandwidth * bandwidth, TINY))


def _predict_rbf(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_z, test_z = _standardize(train_x, test_x)
    delta = train_z[:, None, :] - train_z[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    nonzero = distances[distances > 0.0]
    bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    kernel_train = _rbf_kernel(train_z, train_z, bandwidth)
    ridge = 0.01 * float(np.trace(kernel_train)) / train_z.shape[0]
    alpha = np.linalg.solve(
        kernel_train + ridge * np.eye(train_z.shape[0]), train_y
    )
    return _rbf_kernel(test_z, train_z, bandwidth) @ alpha


def _predict_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_z, test_z = _standardize(train_x, test_x)
    delta = test_z[:, None, :] - train_z[None, :, :]
    distances2 = np.sum(delta * delta, axis=2)
    predictions = []
    k = min(NN_K, train_z.shape[0])
    for row_distances in distances2:
        nearest = np.argsort(row_distances, kind="stable")[:k]
        predictions.append(np.mean(train_y[nearest], axis=0))
    return np.asarray(predictions, dtype=np.float64)


def _cross_output_predict(
    features: dict[str, np.ndarray], responses: np.ndarray, smoother: str
) -> np.ndarray:
    predictions = np.empty_like(responses, dtype=np.float64)
    for fold in range(N_FOLDS):
        test_mask = (np.arange(N_OUTPUTS) % N_FOLDS) == fold
        train_mask = ~test_mask
        if smoother == "polynomial_ridge":
            predictions[test_mask] = _predict_polynomial(
                features["poly"][train_mask],
                responses[train_mask],
                features["poly"][test_mask],
            )
        elif smoother == "rbf_kernel_ridge":
            predictions[test_mask] = _predict_rbf(
                features["rbf"][train_mask],
                responses[train_mask],
                features["rbf"][test_mask],
            )
        elif smoother == "knn16":
            predictions[test_mask] = _predict_knn(
                features["rbf"][train_mask],
                responses[train_mask],
                features["rbf"][test_mask],
            )
        else:
            raise ValueError(smoother)
    return predictions


def _feature_bundle(anchor: np.ndarray, final_weight: np.ndarray) -> dict[str, np.ndarray]:
    a = anchor @ final_weight
    q = np.sqrt(np.sum(final_weight * final_weight, axis=0))
    u = a / np.maximum(q, TINY)
    log_q = np.log(np.maximum(q, TINY))
    return {
        "poly": np.column_stack((a, u, log_q)),
        "rbf": np.column_stack((u, log_q)),
    }


def _mse(vector: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((vector - truth) ** 2))


def _ratio_stats(baseline: list[float], comparison: list[float]) -> dict[str, float]:
    ratios = np.asarray(baseline, dtype=np.float64) / np.maximum(
        np.asarray(comparison, dtype=np.float64), TINY
    )
    return _stats(ratios.tolist())


def _load_rows(path: Path) -> tuple[list[dict], int, list[int], list[int]]:
    rows = {}
    failures = 0
    duplicates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok") is True:
            index = int(row["mlp_index"])
            if index in rows:
                duplicates.append(index)
            else:
                rows[index] = row
        else:
            failures += 1
    ordered = [rows[index] for index in sorted(rows)]
    pending = sorted(set(range(N_MLPS)) - set(rows))
    return ordered, failures, duplicates, pending


def _empty_method() -> dict[str, object]:
    return {
        "direct_prediction_mse_cells": [],
        "truth_response_ceiling_mse_cells": [],
        "baseline_mse_per_mlp": [],
        "direct_prediction_mse_per_mlp": [],
        "blend_mse_per_mlp": [],
        "oracle_ceiling_mse_per_mlp": [],
        "oracle_lambda_per_mlp": [],
        "noisy_predictions": [],
        "oracle_predictions": [],
        "responses": [],
        "truths": [],
    }


def _finish_method(method: dict[str, object]) -> dict[str, object]:
    y = np.asarray(method.pop("responses"), dtype=np.float64)
    p = np.asarray(method.pop("noisy_predictions"), dtype=np.float64)
    truth_predictions = np.asarray(method.pop("oracle_predictions"), dtype=np.float64)
    truths = np.asarray(method.pop("truths"), dtype=np.float64)
    baseline_per_mlp = np.asarray(method["baseline_mse_per_mlp"], dtype=np.float64)
    direct_per_mlp = np.mean((p - truths) ** 2, axis=1).reshape(N_MLPS, N_REPS).mean(axis=1)

    noisy_error = y - truths
    noisy_delta = p - y
    denominator = float(np.sum(noisy_delta * noisy_delta))
    global_lambda = -float(np.sum(noisy_error * noisy_delta)) / max(denominator, TINY)
    blended = y + global_lambda * noisy_delta
    blended_mse = float(np.mean((blended - truths) ** 2))
    blended_by_mlp = np.mean(
        np.mean((blended - truths) ** 2, axis=1).reshape(N_MLPS, N_REPS), axis=1
    )
    three_rep_blended = np.mean(blended.reshape(N_MLPS, N_REPS, N_OUTPUTS), axis=1)
    three_rep_truths = truths.reshape(N_MLPS, N_REPS, N_OUTPUTS)[:, 0, :]
    three_rep_blended_mse = float(np.mean((three_rep_blended - three_rep_truths) ** 2))

    per_mlp_lambdas = []
    per_mlp_blends = []
    for mlp_index in range(N_MLPS):
        start = mlp_index * N_REPS
        stop = start + N_REPS
        local_error = noisy_error[start:stop]
        local_delta = noisy_delta[start:stop]
        local_denominator = float(np.sum(local_delta * local_delta))
        local_lambda = -float(np.sum(local_error * local_delta)) / max(local_denominator, TINY)
        per_mlp_lambdas.append(local_lambda)
        per_mlp_blends.append(float(np.mean((y[start:stop] + local_lambda * local_delta - truths[start:stop]) ** 2)))

    oracle_mse = float(np.mean((truth_predictions - truths) ** 2))
    oracle_by_mlp = np.mean(
        np.mean((truth_predictions - truths) ** 2, axis=1).reshape(N_MLPS, N_REPS),
        axis=1,
    )
    method.update(
        {
            "direct_prediction_M1": float(np.mean((p - truths) ** 2)),
            "truth_response_heldout_M1": oracle_mse,
            "oracle_global_lambda": global_lambda,
            "oracle_global_blend_M1": blended_mse,
            "oracle_global_blend_three_rep_mean_MSE": three_rep_blended_mse,
            "oracle_per_mlp_lambda": _stats(per_mlp_lambdas),
            "oracle_per_mlp_blend_M1_descriptive": _stats(per_mlp_blends),
            "direct_prediction_mse_per_mlp": _stats(direct_per_mlp.tolist()),
            "truth_response_ceiling_mse_per_mlp": _stats(oracle_by_mlp.tolist()),
            "baseline_over_direct_ratio_per_mlp": _ratio_stats(
                baseline_per_mlp.tolist(), direct_per_mlp.tolist()
            ),
            "baseline_over_blend_ratio_per_mlp": _ratio_stats(
                baseline_per_mlp.tolist(), blended_by_mlp.tolist()
            ),
            "baseline_over_oracle_ceiling_ratio_per_mlp": _ratio_stats(
                baseline_per_mlp.tolist(), oracle_by_mlp.tolist()
            ),
            "direct_prediction_mse_cells": _stats(
                np.mean((p - truths) ** 2, axis=1).tolist()
            ),
            "truth_response_ceiling_mse_cells": _stats(
                np.mean((truth_predictions - truths) ** 2, axis=1).tolist()
            ),
            "finite": bool(
                np.all(np.isfinite(y))
                and np.all(np.isfinite(p))
                and np.all(np.isfinite(truth_predictions))
                and np.isfinite(global_lambda)
            ),
        }
    )
    return method


def _run(fly_path: Path, bank_path: Path) -> dict[str, object]:
    rows, failures, duplicates, pending = _load_rows(fly_path)
    bank = np.load(bank_path)
    methods = {name: _empty_method() for name in ("polynomial_ridge", "rbf_kernel_ridge", "knn16")}
    integrity = {
        "rows_100": len(rows) == N_MLPS and [row["mlp_index"] for row in rows] == list(range(N_MLPS)),
        "zero_failures": failures == 0,
        "zero_duplicates": len(duplicates) == 0,
        "zero_pending": len(pending) == 0,
        "checksum_valid": 0,
        "baseline_mse_matches": 0,
        "finite_rows": 0,
    }
    baseline_per_mlp = []
    baseline_three_rep_mses = []
    finite_row_flags = []

    for row in rows:
        index = int(row["mlp_index"])
        seed = int(bank["seeds"][index])
        mlp = build_mlp(N_OUTPUTS, 32, seed)
        weights = [np.asarray(weight, dtype=np.float32) for weight in mlp.weights]
        actual_checksum = _weights_sha256(weights)
        checksum_ok = (
            actual_checksum == str(bank["weights_sha256"][index])
            and actual_checksum == str(row.get("weights_sha256"))
            and row.get("checksum_ok") is True
        )
        integrity["checksum_valid"] += int(checksum_ok)
        truth = np.asarray(bank["truths"][index, -1], dtype=np.float64)
        anchor = np.asarray(bank["truths"][index, -2], dtype=np.float64)
        final_weight = weights[-1].astype(np.float64)
        features = _feature_bundle(anchor, final_weight)

        reps = row.get("replicates", [])
        row_baseline_mses = []
        row_finite = checksum_ok and len(reps) == N_REPS
        for rep_record in reps:
            y = np.asarray(rep_record["baseline_estimate"], dtype=np.float64)
            stored_mse = float(row["baseline_mses"][int(rep_record["rep"])])
            recomputed_mse = _mse(y, truth)
            row_baseline_mses.append(recomputed_mse)
            matches = bool(np.isclose(stored_mse, recomputed_mse, rtol=BASELINE_MATCH_RTOL, atol=BASELINE_MATCH_ATOL))
            integrity["baseline_mse_matches"] += int(matches)
            row_finite = row_finite and matches and y.shape == (N_OUTPUTS,) and np.all(np.isfinite(y))
            for name, method in methods.items():
                prediction = _cross_output_predict(features, y, name)
                oracle_prediction = _cross_output_predict(features, truth, name)
                method["responses"].append(y)
                method["noisy_predictions"].append(prediction)
                method["oracle_predictions"].append(oracle_prediction)
                method["truths"].append(truth)
        baseline_per_mlp.append(float(np.mean(row_baseline_mses)))
        baseline_three_rep_mses.append(_mse(np.mean([np.asarray(r["baseline_estimate"], dtype=np.float64) for r in reps], axis=0), truth))
        finite_row_flags.append(row_finite)

    integrity["finite_rows"] = int(sum(finite_row_flags))
    integrity["checksum_valid"] = int(integrity["checksum_valid"])
    integrity["baseline_mse_matches"] = int(integrity["baseline_mse_matches"])
    integrity["all_checks_pass"] = bool(
        integrity["rows_100"]
        and integrity["zero_failures"]
        and integrity["zero_duplicates"]
        and integrity["zero_pending"]
        and integrity["checksum_valid"] == N_MLPS
        and integrity["baseline_mse_matches"] == N_MLPS * N_REPS
        and integrity["finite_rows"] == N_MLPS
    )

    for method in methods.values():
        method["baseline_mse_per_mlp"] = baseline_per_mlp
        _finish_method(method)

    baseline_m1 = float(np.mean(baseline_per_mlp))
    baseline_three_rep_mean_mse = float(np.mean(baseline_three_rep_mses))
    method_summaries = {}
    for name, method in methods.items():
        blend_m1 = float(method["oracle_global_blend_M1"])
        ceiling_m1 = float(method["truth_response_heldout_M1"])
        method_summaries[name] = method

    noisy_close_conditions = {
        name: (
            baseline_m1 / max(float(method["oracle_global_blend_M1"]), TINY) < CLOSE_BLEND_GAIN_LIMIT
            or float(method["oracle_global_blend_M1"]) > CLOSE_BLEND_MSE_FLOOR
        )
        for name, method in method_summaries.items()
    }
    ceiling_close_conditions = {
        name: float(method["truth_response_heldout_M1"]) > CLOSE_ORACLE_CEILING_FLOOR
        for name, method in method_summaries.items()
    }
    survives_conditions = {
        name: (
            float(method["oracle_global_blend_M1"]) <= SURVIVE_BLEND_MSE
            and baseline_m1 / max(float(method["oracle_global_blend_M1"]), TINY) >= SURVIVE_BLEND_GAIN
            and float(method["truth_response_heldout_M1"]) <= SURVIVE_ORACLE_CEILING
        )
        for name, method in method_summaries.items()
    }
    if not integrity["all_checks_pass"]:
        closeout = "INCONCLUSIVE"
        closeout_reason = "integrity failure; closeout rule not applied as a valid scout"
    elif all(noisy_close_conditions.values()) or all(ceiling_close_conditions.values()):
        closeout = "CLOSE"
        closeout_reason = "frozen closeout condition satisfied"
    elif any(survives_conditions.values()):
        closeout = "SURVIVES"
        closeout_reason = "frozen survival condition satisfied"
    else:
        closeout = "INCONCLUSIVE"
        closeout_reason = "neither frozen closeout nor survival condition satisfied"

    return {
        "prefix": PREFIX,
        "inputs": {
            "fly_jsonl": str(fly_path),
            "truth_bank": str(bank_path),
            "allowed_rebuild": "local_engine.build_mlp only",
            "oracle_feature": "bank truths[idx,-2] only; illegal optimistic anchor",
        },
        "integrity": integrity,
        "baseline": {
            "M1": baseline_m1,
            "three_rep_mean_MSE": baseline_three_rep_mean_mse,
            "per_mlp_M1": _stats(baseline_per_mlp),
            "per_mlp_three_rep_mean_MSE": _stats(baseline_three_rep_mses),
        },
        "methods": method_summaries,
        "closeout_rule_frozen_before_results": CLOSEOUT_RULE,
        "closeout": closeout,
        "closeout_reason": closeout_reason,
        "no_promotion_authorization": "negative result closes the ceiling question; positive result only authorizes a later legal Fly gate",
    }


def _report(result: dict[str, object]) -> str:
    lines = [
        "# Final-weight collapse oracle-ceiling scout",
        "",
        "This is a read-only, optimistic oracle ceiling. The layer-30 bank truth is an illegal feature for estimator promotion. A negative result closes the ceiling question; a positive result only authorizes a later legal Fly gate.",
        "",
        "## Frozen closeout rule",
        "",
        f"`{json.dumps(result['closeout_rule_frozen_before_results'], sort_keys=True)}`",
        "",
        f"## Verdict: **{result['closeout']}**",
        "",
        f"{result['closeout_reason']}",
        "",
        f"Integrity: `{json.dumps(result['integrity'], sort_keys=True)}`",
        f"Baseline: `{json.dumps(result['baseline'], sort_keys=True)}`",
        "",
        "## Smoother results",
        "",
    ]
    for name, method in result["methods"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Direct prediction M1: `{method['direct_prediction_M1']:.12g}`.",
                f"- Truth-oracle global lambda: `{method['oracle_global_lambda']:.12g}`.",
                f"- Noisy-response truth-oracle global blend M1: `{method['oracle_global_blend_M1']:.12g}`; three-rep-mean MSE: `{method['oracle_global_blend_three_rep_mean_MSE']:.12g}`.",
                f"- Baseline/blend global gain: `{result['baseline']['M1'] / max(method['oracle_global_blend_M1'], TINY):.12g}`.",
                f"- Truth-response held-out approximation ceiling M1 (illegal/oracle-only): `{method['truth_response_heldout_M1']:.12g}`.",
                f"- Per-MLP baseline/blend ratio: `{json.dumps(method['baseline_over_blend_ratio_per_mlp'], sort_keys=True)}`.",
                f"- Per-MLP baseline/oracle-ceiling ratio: `{json.dumps(method['baseline_over_oracle_ceiling_ratio_per_mlp'], sort_keys=True)}`.",
            ]
        )
    lines.extend(
        [
            "",
            "The three smoothers were fixed in advance: cubic polynomial ridge on standardized `[a,u,logq]`, Gaussian RBF kernel ridge on standardized `[u,logq]`, and uniform 16-nearest-neighbor regression. Output folds are fixed by `j mod 4`; no truth-based model selection was used.",
            "",
            "No estimator was generated or scored locally, and no Fly run was launched.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fly-jsonl", type=Path, default=FLY_JSONL)
    parser.add_argument("--bank", type=Path, default=BANK_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    result = _run(args.fly_jsonl, args.bank)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"closeout": result["closeout"], "baseline_M1": result["baseline"]["M1"], "methods": {name: {"blend_M1": value["oracle_global_blend_M1"], "ceiling_M1": value["truth_response_heldout_M1"]} for name, value in result["methods"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
