#!/usr/bin/env python3
"""Aggregate the frozen odd-state Rao--Blackwell K8 Stage-A gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N_MLPS = 100
N_REPS = 3
TOL = 1e-12
EXPECTED_SCRIPT_VERSION = "odd-rb-k8-v1"
EXPECTED_STATE_INDEXING = "state0=post-recolor first ReLU; W1..W8 prefix; branch after W8; suffix W9..W31"
EXPECTED_LAYERS = list(range(9, 32))
FINAL_SHAPE = (256,)


def _stats(values) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
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
    for line in path.read_text().splitlines():
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
    ordered = [rows[i] for i in sorted(rows)]
    pending = sorted(set(range(N_MLPS)) - set(rows))
    return ordered, failures, duplicate_indices, pending


def _schema_reasons(row) -> list[str]:
    reasons = []
    if row.get("script_version") != EXPECTED_SCRIPT_VERSION:
        reasons.append("script_version")
    config = row.get("config")
    if not isinstance(config, dict):
        reasons.append("config_missing")
    else:
        expected = {
            "width": 256,
            "depth": 32,
            "blocks": 16,
            "reps": 3,
            "state_indexing": EXPECTED_STATE_INDEXING,
            "prediction_scope": "final W31 mean only; all 23 W9..W31 closure steps computed and diagnosed",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                reasons.append(f"config_{key}")
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
    for rep_idx, rep in enumerate(reps):
        for method in ("current", "candidate"):
            vector = np.asarray(rep.get(f"{method}_vector"), dtype=float)
            if vector.shape != FINAL_SHAPE:
                reasons.append(f"rep{rep_idx}_{method}_shape")
            elif not np.all(np.isfinite(vector)):
                reasons.append(f"rep{rep_idx}_{method}_nonfinite")
            mse = rep.get("mse", {}).get(method)
            if mse is None or not np.isfinite(float(mse)):
                reasons.append(f"rep{rep_idx}_{method}_mse")
        diagnostics = rep.get("diagnostics")
        closure = diagnostics.get("closure") if isinstance(diagnostics, dict) else None
        layers = closure.get("layers") if isinstance(closure, dict) else None
        if not isinstance(layers, list) or len(layers) != len(EXPECTED_LAYERS):
            reasons.append(f"rep{rep_idx}_closure_count")
        elif [layer.get("layer") for layer in layers] != EXPECTED_LAYERS:
            reasons.append(f"rep{rep_idx}_closure_labels")
    return reasons


def _decomposition(rows, method: str):
    per_rep = []
    per_rep_values = [[] for _ in range(N_REPS)]
    per_mlp_m1 = []
    per_mlp_m3 = []
    for row in rows:
        reps = row.get("reps", [])
        mses = [float(rep["mse"][method]) for rep in reps]
        per_rep.extend(mses)
        for rep_idx, mse in enumerate(mses[:N_REPS]):
            per_rep_values[rep_idx].append(mse)
        per_mlp_m1.append(float(np.mean(mses)))
        vectors = np.asarray([rep[f"{method}_vector"] for rep in reps], dtype=float)
        truth = np.asarray(row["truth_final"], dtype=float)
        per_mlp_m3.append(float(np.mean((np.mean(vectors, axis=0) - truth) ** 2)))
    M1 = float(np.mean(per_rep)) if per_rep else float("inf")
    M3 = float(np.mean(per_mlp_m3)) if per_mlp_m3 else float("inf")
    bias2 = max((3.0 * M3 - M1) / 2.0, 0.0)
    var16 = max(M1 - bias2, 0.0)
    projected = {str(B): bias2 + var16 * 16.0 / B for B in (25, 26, 27)}
    return {
        "M1": M1,
        "M3": M3,
        "bias2": bias2,
        "var16": var16,
        "projected_MSE": projected,
        "per_rep_MSE": [float(np.mean(values)) if values else float("inf") for values in per_rep_values],
        "per_mlp_M1": per_mlp_m1,
        "per_mlp_M3": per_mlp_m3,
    }


def _diagnostics(rows):
    residuals = []
    s2_mins = []
    vo_raw_mins = []
    vo_mins = []
    initial_residuals = []
    finite_flags = []
    s2_flags = []
    vo_flags = []
    for row in rows:
        for rep in row.get("reps", []):
            rep_diag = rep["diagnostics"]
            closure = rep_diag["closure"]
            initial_residuals.append(float(closure["initial_factor_residual"]))
            scalar_values = [
                closure["initial_factor_residual"], closure["initial_g"],
                closure["initial_r_mean"], closure["initial_c_mean"],
            ]
            for layer in closure["layers"]:
                scalar_values.extend(
                    layer[key] for key in ("factor_residual", "g", "s2_min", "vo_raw_min", "vo_min")
                )
            finite_flags.append(
                bool(rep_diag["branch_finite"])
                and bool(rep_diag["target_mean_finite"])
                and bool(rep_diag["target_cov_finite"])
                and bool(closure["finite"])
                and bool(np.all(np.isfinite(np.asarray(scalar_values, dtype=float))))
            )
            s2_flags.append(bool(closure["s2_positive"]))
            vo_flags.append(bool(closure["vo_nonnegative"]))
            for layer in closure["layers"]:
                residuals.append(float(layer["factor_residual"]))
                s2_mins.append(float(layer["s2_min"]))
                vo_raw_mins.append(float(layer["vo_raw_min"]))
                vo_mins.append(float(layer["vo_min"]))
    return {
        "initial_factor_residual": _stats(initial_residuals) if initial_residuals else {},
        "closure_factor_residual": _stats(residuals) if residuals else {},
        "s2_min": _stats(s2_mins) if s2_mins else {},
        "vo_raw_min": _stats(vo_raw_mins) if vo_raw_mins else {},
        "vo_min": _stats(vo_mins) if vo_mins else {},
        "all_finite": bool(all(finite_flags)) if finite_flags else False,
        "all_s2_positive": bool(all(s2_flags)) if s2_flags else False,
        "all_vo_nonnegative": bool(all(vo_flags)) if vo_flags else False,
    }


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
    schema_rows = [row for row in rows if str(row["mlp_index"]) not in schema_invalid]
    schema_valid = len(schema_invalid) == 0
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == N_MLPS and indices == list(range(N_MLPS))
    checksum_valid_rows = sum(bool(row.get("checksum_ok")) for row in rows)
    checksums = complete and checksum_valid_rows == N_MLPS
    current = _decomposition(schema_rows, "current")
    candidate = _decomposition(schema_rows, "candidate")
    current_m1 = current["M1"]
    candidate_m1 = candidate["M1"]
    ratios = np.asarray(current["per_mlp_M1"], dtype=float) / np.maximum(
        np.asarray(candidate["per_mlp_M1"], dtype=float), 1e-300
    ) if schema_rows else np.array([])
    ratio_stats = _stats(ratios) if schema_rows else {}
    diagnostics = _diagnostics(schema_rows)
    integrity_gates = {
        "complete_100": complete,
        "checksums": checksums,
        "zero_failures": failures == 0,
        "zero_pending": len(pending) == 0,
        "zero_duplicates": len(duplicate_indices) == 0,
        "schema_valid": schema_valid,
    }
    science_gates = {
        "candidate_M1": candidate_m1 <= 2.65e-6,
        "global_current_over_candidate_M1": current_m1 / max(candidate_m1, 1e-300) >= 0.98,
        "candidate_bias2": candidate["bias2"] <= 2.5e-7,
        "candidate_projected_B27": candidate["projected_MSE"]["27"] <= 1.52e-6,
        "all_finite": diagnostics["all_finite"],
        "all_s2_positive": diagnostics["all_s2_positive"],
        "all_vo_nonnegative": diagnostics["all_vo_nonnegative"],
    }
    overall = all(integrity_gates.values()) and all(science_gates.values())
    result = {
        "script_version": "odd-rb-k8-v1-aggregate",
        "n_rows": len(rows),
        "returned": len(rows),
        "failures": failures,
        "pending_indices": pending,
        "duplicate_count": len(duplicate_indices),
        "duplicate_indices": sorted(set(duplicate_indices)),
        "checksum_valid_rows": checksum_valid_rows,
        "schema_valid_rows": len(schema_rows),
        "schema_invalid": schema_invalid,
        "complete": complete,
        "checksums": checksums,
        "integrity_gates": integrity_gates,
        "science_gates": science_gates,
        "overall_pass": overall,
        "verdict": "PASS" if overall else "FAIL",
        "current": {k: v for k, v in current.items() if k not in ("per_mlp_M1", "per_mlp_M3")},
        "candidate": {k: v for k, v in candidate.items() if k not in ("per_mlp_M1", "per_mlp_M3")},
        "current_over_candidate_global_M1": current_m1 / max(candidate_m1, 1e-300),
        "per_mlp_ratio": ratio_stats,
        "diagnostics": diagnostics,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Antipodal odd-state Rao--Blackwell K8 Stage A",
        "",
        f"Rows `{len(rows)}/100`; failures `{failures}`; pending `{pending}`; duplicates `{len(duplicate_indices)}`; checksum-valid `{checksum_valid_rows}`; schema-valid `{len(schema_rows)}`.",
        f"Schema-invalid: `{json.dumps(schema_invalid, sort_keys=True)}`",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        f"Current decomposition: `{json.dumps(current, sort_keys=True)}`",
        f"Candidate decomposition: `{json.dumps(candidate, sort_keys=True)}`",
        f"Current/candidate per-MLP ratio: `{json.dumps(ratio_stats, sort_keys=True)}`",
        f"Diagnostics: `{json.dumps(diagnostics, sort_keys=True)}`",
        "",
        "## Integrity gates",
    ]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in integrity_gates.items())
    lines += ["", "## Science gates"]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in science_gates.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"returned": len(rows), "failures": failures, "pending": pending, "duplicates": len(duplicate_indices), "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
