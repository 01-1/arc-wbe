#!/usr/bin/env python3
"""Aggregate the frozen odd low-rank transport truth-bank gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N_MLPS = 100
N_REPS = 3
FINAL_SHAPE = (256,)
EXPECTED_SCRIPT_VERSION = "odd-lr-transport-v1-20260710"
EXPECTED_STATE_INDEXING = (
    "state0=post-recolor first ReLU; W1..W4 shared exact; "
    "candidate transport W5..W31; current exact W5..W31"
)
EXPECTED_REFRESHES = [(4, 64), (8, 32), (16, 16), (24, 8)]
EXPECTED_LAYERS = list(range(5, 32))
EXPECTED_LAYER_RANKS = {
    layer: 64 if layer <= 8 else 32 if layer <= 16 else 16 if layer <= 24 else 8
    for layer in EXPECTED_LAYERS
}
EXPECTED_ROUTE_STREAMS = [0x0DD1_0710, 0x0DD1_1710, 0x0DD1_2710]
EXPECTED_REFRESH_STREAMS = [0x6A40_0710, 0x6A40_1710, 0x6A40_2710, 0x6A40_3710]
EXPECTED_PAIRING = "positive_rows[p] paired with negative_rows[p] after global recolor"
EXPECTED_ODD_ENERGY_RESTORE = (
    "each layer uses Qgram=Q.T@Q and coordinate scale "
    "sqrt(max(sum(O^2),tiny)/max(sum(C*(Qgram@C)),tiny)); restored energy is "
    "checked by the same identity; no per-layer Q@C diagnostic apply and no clipping"
)
EXPECTED_FIRST_SUCCESSOR = (
    "fp64 moments; fp32 centered strength-1.5 scale application/writeback"
)
EXPECTED_PROPAGATION = "fp32 L3 Strassen"
EXPECTED_SUBSPACE_DTYPES = (
    "fp32 Omega/Y/Q0/B0/Q/Qgram/C/C_scaled/A/B/E/O; fp64 small Gram, Cholesky, "
    "solve, and eigh"
)
EXPECTED_RECOLOR_DTYPES = (
    "corrected current route: fp64 moments/Cholesky/recolor solve; fp32 centered "
    "recolor application/writeback"
)
EXPECTED_BLOCK_CONFIG = (
    "diagnostic-only statistics from 16x256 final pair-even block means centered "
    "per output coordinate over blocks, then pooled"
)
PAIR_RECON_TOL = 2e-7
Q_ORTH_TOL = 2e-5
PROJECTION_RANGE_TOL = 5e-5
PROJECTION_IDENTITY_TOL = 5e-4
SECOND_MOMENT_REL_TOL = 2e-4
TINY = 1e-12
MSE_ABS_TOL = 1e-15
MSE_REL_TOL = 1e-10


def _stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.1)),
        "q90": float(np.quantile(x, 0.9)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _read(path: Path):
    rows = {}
    failures = 0
    duplicate_indices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            index = int(row["mlp_index"])
            if index in rows:
                duplicate_indices.append(index)
            else:
                rows[index] = row
        else:
            failures += 1
    ordered = [rows[index] for index in sorted(rows)]
    pending = sorted(set(range(N_MLPS)) - set(rows))
    return ordered, failures, duplicate_indices, pending


def _schema_reasons(row: dict[str, object]) -> list[str]:
    reasons = []
    if row.get("script_version") != EXPECTED_SCRIPT_VERSION:
        reasons.append("script_version")
    if row.get("weights_sha256") != row.get("expected_weights_sha256"):
        reasons.append("weight_checksum_pair")
    if row.get("checksum_ok") is not True:
        reasons.append("checksum_ok")

    config = row.get("config")
    if not isinstance(config, dict):
        reasons.append("config_missing")
    else:
        expected = {
            "width": 256,
            "depth": 32,
            "blocks": 16,
            "pairs_per_block": 256,
            "positive_rows": 4096,
            "total_rows": 8192,
            "reps": 3,
            "oversample": 8,
            "state_indexing": EXPECTED_STATE_INDEXING,
            "prediction_scope": "final W31 pair-even mean vector only",
            "route_streams": EXPECTED_ROUTE_STREAMS,
            "refresh_streams": EXPECTED_REFRESH_STREAMS,
            "pairing": EXPECTED_PAIRING,
            "odd_energy_restore": EXPECTED_ODD_ENERGY_RESTORE,
            "first_successor": EXPECTED_FIRST_SUCCESSOR,
            "propagation": EXPECTED_PROPAGATION,
            "subspace_dtypes": EXPECTED_SUBSPACE_DTYPES,
            "recolor_dtypes": EXPECTED_RECOLOR_DTYPES,
            "block_correction": EXPECTED_BLOCK_CONFIG,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                reasons.append(f"config_{key}")
        schedule = config.get("schedule")
        expected_schedule = [
            {"refresh_after_layer": 4, "rank": 64, "used_for_layers": [5, 8]},
            {"refresh_after_layer": 8, "rank": 32, "used_for_layers": [9, 16]},
            {"refresh_after_layer": 16, "rank": 16, "used_for_layers": [17, 24]},
            {"refresh_after_layer": 24, "rank": 8, "used_for_layers": [25, 31]},
        ]
        if schedule != expected_schedule:
            reasons.append("config_schedule")
        tolerances = config.get("tolerances")
        expected_tolerances = {
            "pair_reconstruction_relative_fro_max": PAIR_RECON_TOL,
            "q_orthogonality_relative_fro_max": Q_ORTH_TOL,
            "projection_range_tolerance": PROJECTION_RANGE_TOL,
            "projection_identity_error_max": PROJECTION_IDENTITY_TOL,
            "post_scale_second_moment_relative_error_max": SECOND_MOMENT_REL_TOL,
            "gram_jitter_relative": 1e-7,
        }
        if tolerances != expected_tolerances:
            reasons.append("config_tolerances")

    truth = np.asarray(row.get("truth_final"), dtype=float)
    if truth.shape != FINAL_SHAPE:
        reasons.append("truth_shape")
    elif not np.all(np.isfinite(truth)):
        reasons.append("truth_nonfinite")

    reps = row.get("reps")
    if not isinstance(reps, list) or len(reps) != N_REPS:
        reasons.append("rep_count")
        return reasons
    if [rep.get("rep") for rep in reps] != [0, 1, 2]:
        reasons.append("rep_ids")
    for rep_index, rep in enumerate(reps):
        for method in ("current", "candidate"):
            vector = np.asarray(rep.get(f"{method}_vector"), dtype=float)
            if vector.shape != FINAL_SHAPE:
                reasons.append(f"rep{rep_index}_{method}_shape")
            elif not np.all(np.isfinite(vector)):
                reasons.append(f"rep{rep_index}_{method}_nonfinite")
            mse = rep.get("mse", {}).get(method)
            if mse is None or not np.isfinite(float(mse)):
                reasons.append(f"rep{rep_index}_{method}_mse")
            elif (
                vector.shape == FINAL_SHAPE
                and np.all(np.isfinite(vector))
                and truth.shape == FINAL_SHAPE
                and np.all(np.isfinite(truth))
            ):
                recomputed_mse = float(np.mean((vector - truth) ** 2))
                if not np.isclose(
                    float(mse),
                    recomputed_mse,
                    rtol=MSE_REL_TOL,
                    atol=MSE_ABS_TOL,
                ):
                    reasons.append(f"rep{rep_index}_{method}_mse_mismatch")
        diagnostics = rep.get("diagnostics")
        transport = diagnostics.get("transport") if isinstance(diagnostics, dict) else None
        refreshes = transport.get("refreshes") if isinstance(transport, dict) else None
        layers = transport.get("layers") if isinstance(transport, dict) else None
        if not isinstance(refreshes, list) or len(refreshes) != 4:
            reasons.append(f"rep{rep_index}_refresh_count")
        elif [(d.get("boundary"), d.get("rank")) for d in refreshes] != EXPECTED_REFRESHES:
            reasons.append(f"rep{rep_index}_refresh_schedule")
        if not isinstance(layers, list) or len(layers) != len(EXPECTED_LAYERS):
            reasons.append(f"rep{rep_index}_layer_count")
        else:
            labels = [d.get("layer") for d in layers]
            ranks = [d.get("rank") for d in layers]
            if labels != EXPECTED_LAYERS:
                reasons.append(f"rep{rep_index}_layer_labels")
            if ranks != [EXPECTED_LAYER_RANKS[layer] for layer in EXPECTED_LAYERS]:
                reasons.append(f"rep{rep_index}_layer_ranks")
        block = diagnostics.get("block_correction") if isinstance(diagnostics, dict) else None
        if not isinstance(block, dict) or block.get("diagnostic_only") is not True:
            reasons.append(f"rep{rep_index}_block_diagnostic")
    return reasons


def _decomposition(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    per_rep_values = [[] for _ in range(N_REPS)]
    per_mlp_m1 = []
    per_mlp_m3 = []
    for row in rows:
        reps = row["reps"]
        mses = [float(rep["mse"][method]) for rep in reps]
        for rep_index, mse in enumerate(mses):
            per_rep_values[rep_index].append(mse)
        per_mlp_m1.append(float(np.mean(mses)))
        vectors = np.asarray([rep[f"{method}_vector"] for rep in reps], dtype=float)
        truth = np.asarray(row["truth_final"], dtype=float)
        per_mlp_m3.append(float(np.mean((np.mean(vectors, axis=0) - truth) ** 2)))
    all_rep_mses = [value for values in per_rep_values for value in values]
    m1 = float(np.mean(all_rep_mses)) if all_rep_mses else float("inf")
    m3 = float(np.mean(per_mlp_m3)) if per_mlp_m3 else float("inf")
    bias2 = max((3.0 * m3 - m1) / 2.0, 0.0)
    var16 = max(m1 - bias2, 0.0)
    return {
        "M1": m1,
        "M3": m3,
        "bias2": bias2,
        "var16": var16,
        "projected_MSE": {
            str(blocks): bias2 + var16 * 16.0 / blocks for blocks in (25, 26, 27)
        },
        "per_rep_MSE": [
            float(np.mean(values)) if values else float("inf")
            for values in per_rep_values
        ],
        "per_mlp_M1": per_mlp_m1,
        "per_mlp_M3": per_mlp_m3,
    }


def _diagnostics(rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, bool]]:
    pair_values = []
    refresh_values = {
        key: []
        for key in (
            "projection_relative_fro",
            "captured_energy",
            "projection_identity_error",
            "q_orthogonality_relative_fro",
            "gram_min_eigenvalue",
            "gram_jitter",
            "gram_min_plus_jitter",
            "range_retained_min_eigenvalue",
            "range_max_eigenvalue",
        )
    }
    refresh_by_boundary = {
        str(boundary): {key: [] for key in refresh_values}
        for boundary, _ in EXPECTED_REFRESHES
    }
    layer_values = {
        key: []
        for key in (
            "post_scale_second_moment_relative_error_max",
            "post_scale_second_moment_relative_error_mean",
            "scale_min",
            "scale_median",
            "scale_max",
        )
    }
    layer_by_layer = {
        str(layer): {key: [] for key in layer_values} for layer in EXPECTED_LAYERS
    }
    block_values = {
        key: []
        for key in (
            "correction_over_exact_variance",
            "exact_candidate_correlation",
            "candidate_over_exact_variance",
            "exact_block_variance",
            "candidate_block_variance",
            "correction_block_variance",
        )
    }
    finite_flags = []
    pair_sane = []
    q_sane = []
    projection_sane = []
    gram_sane = []
    restore_sane = []
    block_sane = []

    for row in rows:
        for rep in row["reps"]:
            diagnostics = rep["diagnostics"]
            transport = diagnostics["transport"]
            pair = transport["pair_reconstruction"]
            pair_value = float(pair["max_relative_fro"])
            pair_values.append(pair_value)
            pair_sane.append(bool(pair["finite"]) and pair_value <= PAIR_RECON_TOL)

            for refresh in transport["refreshes"]:
                boundary_values = refresh_by_boundary[str(refresh["boundary"])]
                for key in refresh_values:
                    value = float(refresh[key])
                    refresh_values[key].append(value)
                    boundary_values[key].append(value)
                q_sane.append(
                    bool(refresh["finite"])
                    and float(refresh["q_orthogonality_relative_fro"]) <= Q_ORTH_TOL
                )
                residual = float(refresh["projection_relative_fro"])
                captured = float(refresh["captured_energy"])
                projection_sane.append(
                    -PROJECTION_RANGE_TOL <= residual <= 1.0 + PROJECTION_RANGE_TOL
                    and -PROJECTION_RANGE_TOL <= captured <= 1.0 + PROJECTION_RANGE_TOL
                    and float(refresh["projection_identity_error"])
                    <= PROJECTION_IDENTITY_TOL
                )
                gram_sane.append(
                    float(refresh["gram_jitter"]) > 0.0
                    and float(refresh["gram_min_plus_jitter"]) > 0.0
                    and float(refresh["range_retained_min_eigenvalue"])
                    >= -PROJECTION_RANGE_TOL
                )

            for layer in transport["layers"]:
                per_layer_values = layer_by_layer[str(layer["layer"])]
                for key in layer_values:
                    value = float(layer[key])
                    layer_values[key].append(value)
                    per_layer_values[key].append(value)
                restore_sane.append(
                    bool(layer["finite"])
                    and float(layer["post_scale_second_moment_relative_error_max"])
                    <= SECOND_MOMENT_REL_TOL
                    and float(layer["scale_min"]) > 0.0
                    and float(layer["scale_min"]) <= float(layer["scale_median"])
                    <= float(layer["scale_max"])
                )

            block = diagnostics["block_correction"]
            for key in block_values:
                block_values[key].append(float(block[key]))
            block_sane.append(
                bool(block["finite"])
                and float(block["exact_block_variance"]) > TINY
                and float(block["candidate_block_variance"]) >= 0.0
                and float(block["correction_block_variance"]) >= 0.0
                and -1.0 - PROJECTION_RANGE_TOL
                <= float(block["exact_candidate_correlation"])
                <= 1.0 + PROJECTION_RANGE_TOL
            )
            scalar_bundle = list(pair.values())[:-1]
            scalar_bundle += [value for refresh in transport["refreshes"] for value in refresh.values() if isinstance(value, (int, float))]
            scalar_bundle += [value for layer in transport["layers"] for value in layer.values() if isinstance(value, (int, float))]
            finite_flags.append(
                bool(diagnostics["state0_finite"])
                and bool(diagnostics["layer4_finite"])
                and bool(diagnostics["target_mean_finite"])
                and bool(diagnostics["target_cov_finite"])
                and bool(transport["finite"])
                and bool(np.all(np.isfinite(np.asarray(scalar_bundle, dtype=float))))
            )

    summary = {
        "pair_reconstruction_max_relative_fro": _stats(pair_values),
        "refresh_all": {key: _stats(values) for key, values in refresh_values.items()},
        "refresh_by_boundary": {
            boundary: {key: _stats(values) for key, values in metrics.items()}
            for boundary, metrics in refresh_by_boundary.items()
        },
        "odd_energy_restore_layers": {
            key: _stats(values) for key, values in layer_values.items()
        },
        "odd_energy_restore_by_layer": {
            layer: {key: _stats(values) for key, values in metrics.items()}
            for layer, metrics in layer_by_layer.items()
        },
        "block_correction_diagnostic_only": {
            key: _stats(values) for key, values in block_values.items()
        },
    }
    checks = {
        "all_finite": bool(finite_flags) and all(finite_flags),
        "pair_reconstruction_sane": bool(pair_sane) and all(pair_sane),
        "q_orthogonality_sane": bool(q_sane) and all(q_sane),
        "projection_diagnostics_sane": bool(projection_sane) and all(projection_sane),
        "gram_diagnostics_sane": bool(gram_sane) and all(gram_sane),
        "odd_energy_restoration_sane": bool(restore_sane) and all(restore_sane),
        "block_diagnostics_sane": bool(block_sane) and all(block_sane),
    }
    return summary, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows, failures, duplicate_indices, pending = _read(args.jsonl)
    schema_invalid = {
        str(row["mlp_index"]): reasons
        for row in rows
        if (reasons := _schema_reasons(row))
    }
    schema_rows = [
        row for row in rows if str(row["mlp_index"]) not in schema_invalid
    ]
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == N_MLPS and indices == list(range(N_MLPS))
    checksum_valid_rows = sum(
        bool(row.get("checksum_ok"))
        and row.get("weights_sha256") == row.get("expected_weights_sha256")
        for row in rows
    )
    checksums = complete and checksum_valid_rows == N_MLPS
    current = _decomposition(schema_rows, "current")
    candidate = _decomposition(schema_rows, "candidate")
    ratios = (
        np.asarray(current["per_mlp_M1"], dtype=float)
        / np.maximum(np.asarray(candidate["per_mlp_M1"], dtype=float), 1e-300)
        if schema_rows
        else np.asarray([], dtype=float)
    )
    ratio_stats = _stats(ratios.tolist())
    diagnostics, numerical_checks = _diagnostics(schema_rows)
    block_diagnostic_integrity = numerical_checks.pop("block_diagnostics_sane", False)

    integrity_gates = {
        "complete_100": complete,
        "checksums_100": checksums,
        "zero_failures": failures == 0,
        "zero_pending": len(pending) == 0,
        "zero_duplicates": len(duplicate_indices) == 0,
        "schema_valid": len(schema_invalid) == 0,
    }
    candidate_m1 = float(candidate["M1"])
    current_m1 = float(current["M1"])
    science_gates = {
        "candidate_M1": candidate_m1 <= 2.80e-6,
        "global_current_over_candidate_M1": current_m1 / max(candidate_m1, 1e-300)
        >= 0.90,
        "candidate_bias2": float(candidate["bias2"]) <= 2.5e-7,
        "candidate_projected_B27": float(candidate["projected_MSE"]["27"])
        <= 1.52e-6,
        **numerical_checks,
    }
    overall = all(integrity_gates.values()) and all(science_gates.values())
    compact_current = {
        key: value
        for key, value in current.items()
        if key not in ("per_mlp_M1", "per_mlp_M3")
    }
    compact_candidate = {
        key: value
        for key, value in candidate.items()
        if key not in ("per_mlp_M1", "per_mlp_M3")
    }
    result = {
        "script_version": "odd-lr-transport-v1-20260710-aggregate",
        "returned": len(rows),
        "failures": failures,
        "pending_indices": pending,
        "duplicate_count": len(duplicate_indices),
        "duplicate_indices": sorted(set(duplicate_indices)),
        "checksum_valid_rows": checksum_valid_rows,
        "schema_valid_rows": len(schema_rows),
        "schema_invalid": schema_invalid,
        "integrity_gates": integrity_gates,
        "science_gates": science_gates,
        "diagnostic_integrity": {
            "block_correction_diagnostic_only": block_diagnostic_integrity,
        },
        "overall_pass": overall,
        "verdict": "PASS" if overall else "FAIL",
        "current": compact_current,
        "candidate": compact_candidate,
        "current_over_candidate_global_M1": current_m1 / max(candidate_m1, 1e-300),
        "current_over_candidate_per_mlp_ratio": ratio_stats,
        "diagnostics": diagnostics,
        "authorization": "PASS permits only a separate estimator implementation/economics audit; no automatic follow-up",
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Odd low-rank transport Stage A",
        "",
        f"Rows `{len(rows)}/100`; failures `{failures}`; pending `{pending}`; duplicates `{len(duplicate_indices)}`; checksum-valid `{checksum_valid_rows}`; schema-valid `{len(schema_rows)}`.",
        f"Schema-invalid: `{json.dumps(schema_invalid, sort_keys=True)}`",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        f"Current sanity decomposition: `{json.dumps(compact_current, sort_keys=True)}`",
        f"Candidate decomposition: `{json.dumps(compact_candidate, sort_keys=True)}`",
        f"Global current/candidate M1 ratio: `{result['current_over_candidate_global_M1']}`",
        f"Per-MLP current/candidate M1 ratio: `{json.dumps(ratio_stats, sort_keys=True)}`",
        f"Diagnostic-only block integrity: `{'PASS' if block_diagnostic_integrity else 'FAIL'}` (excluded from verdict).",
        "",
        "## Numerical and label-free diagnostics",
        "",
        f"`{json.dumps(diagnostics, sort_keys=True)}`",
        "",
        "The block correction statistics are diagnostic-only and do not define a second candidate or alter PASS thresholds.",
        "",
        "## Integrity gates",
    ]
    lines.extend(
        f"- `{name}`: **{'PASS' if value else 'FAIL'}**."
        for name, value in integrity_gates.items()
    )
    lines += ["", "## Science gates"]
    lines.extend(
        f"- `{name}`: **{'PASS' if value else 'FAIL'}**."
        for name, value in science_gates.items()
    )
    lines += [
        "",
        "A PASS authorizes only a separate estimator implementation/economics audit; it does not authorize a mode, Stage B, rerun, or automatic follow-up.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "returned": len(rows),
                "failures": failures,
                "pending": pending,
                "duplicates": len(duplicate_indices),
                "verdict": result["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
