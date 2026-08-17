#!/usr/bin/env python3
"""Exact vector-MSE aggregation and frozen gate decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _summarize(rows: list[dict[str, object]], stage: str) -> dict[str, object]:
    valid = [r for r in rows if r.get("ok") and r.get("checksum_ok")]
    failures = [r for r in rows if r not in valid]
    rep_rows = []
    by_mlp: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in valid:
        idx = int(row["mlp_index"])
        for rep in row.get("rep_results", []):
            rep_rows.append(rep)
            by_mlp[idx].append(rep)
    current = np.array([float(r["current_mse"]) for r in rep_rows], dtype=float)
    candidate = np.array([float(r["candidate_mse"]) for r in rep_rows], dtype=float)
    per_mlp_ratio = np.array(
        [
            np.mean([float(x["current_mse"]) for x in vals])
            / max(np.mean([float(x["candidate_mse"]) for x in vals]), 1e-300)
            for vals in by_mlp.values()
        ],
        dtype=float,
    )
    lambdas = np.array(
        [float(r["lambda_eb"]) for r in rep_rows], dtype=float
    )
    condition_max = np.array(
        [float(r["ridge_condition_max"]) for r in rep_rows], dtype=float
    )
    condition_mean = np.array(
        [float(r["ridge_condition_mean"]) for r in rep_rows], dtype=float
    )
    ridge_lambda = np.array(
        [float(r["ridge_lambda_mean"]) for r in rep_rows], dtype=float
    )
    feature_min = np.array(
        [float(r["feature_scale_min"]) for r in rep_rows], dtype=float
    )
    feature_max = np.array(
        [float(r["feature_scale_max"]) for r in rep_rows], dtype=float
    )
    result: dict[str, object] = {
        "stage": stage,
        "rows_returned": len(rows),
        "valid_checksums": len(valid),
        "failures": [
            {"mlp_index": r.get("mlp_index"), "error": r.get("error"), "ok": r.get("ok")}
            for r in failures
        ],
        "rep_rows": len(rep_rows),
        "current_mse_mean": float(np.mean(current)) if len(current) else None,
        "candidate_mse_mean": float(np.mean(candidate)) if len(candidate) else None,
        "current_candidate_global_ratio": (
            float(np.mean(current) / max(np.mean(candidate), 1e-300)) if len(current) else None
        ),
        "per_mlp_ratio_median": float(np.median(per_mlp_ratio)) if len(per_mlp_ratio) else None,
        "per_mlp_ratio_q10": float(np.quantile(per_mlp_ratio, 0.10)) if len(per_mlp_ratio) else None,
        "per_mlp_ratio_min": float(np.min(per_mlp_ratio)) if len(per_mlp_ratio) else None,
        "lambda_eb_mean": float(np.mean(lambdas)) if len(lambdas) else None,
        "lambda_eb_median": float(np.median(lambdas)) if len(lambdas) else None,
        "lambda_eb_q10": float(np.quantile(lambdas, 0.10)) if len(lambdas) else None,
        "lambda_eb_min": float(np.min(lambdas)) if len(lambdas) else None,
        "lambda_eb_max": float(np.max(lambdas)) if len(lambdas) else None,
        "ridge_condition_max_observed": float(np.max(condition_max)) if len(condition_max) else None,
        "ridge_condition_mean": float(np.mean(condition_mean)) if len(condition_mean) else None,
        "ridge_lambda_mean": float(np.mean(ridge_lambda)) if len(ridge_lambda) else None,
        "feature_scale_min_mean": float(np.mean(feature_min)) if len(feature_min) else None,
        "feature_scale_max_mean": float(np.mean(feature_max)) if len(feature_max) else None,
        "three_rep_squared_bias_proxy_mean": (
            float(np.mean([float(r["three_rep_squared_bias_proxy"]) for r in valid]))
            if stage == "b" and valid
            else None
        ),
    }
    if stage == "a":
        checks = {
            "100_checksums": len(valid) == 100 and not failures,
            "candidate_mse": result["candidate_mse_mean"] is not None and result["candidate_mse_mean"] <= 1.8e-6,
            "global_ratio": result["current_candidate_global_ratio"] is not None and result["current_candidate_global_ratio"] >= 1.25,
            "median_ratio": result["per_mlp_ratio_median"] is not None and result["per_mlp_ratio_median"] >= 1.10,
            "q10_ratio": result["per_mlp_ratio_q10"] is not None and result["per_mlp_ratio_q10"] >= 0.85,
            "minimum_ratio": result["per_mlp_ratio_min"] is not None and result["per_mlp_ratio_min"] >= 0.65,
            "mean_lambda_range": result["lambda_eb_mean"] is not None and 0.02 <= result["lambda_eb_mean"] <= 0.95,
        }
    else:
        checks = {
            "100_checksums": len(valid) == 100 and not failures,
            "candidate_mse": result["candidate_mse_mean"] is not None and result["candidate_mse_mean"] <= 1.6e-6,
            "global_ratio": result["current_candidate_global_ratio"] is not None and result["current_candidate_global_ratio"] >= 1.35,
            "median_ratio": result["per_mlp_ratio_median"] is not None and result["per_mlp_ratio_median"] >= 1.20,
            "q10_ratio": result["per_mlp_ratio_q10"] is not None and result["per_mlp_ratio_q10"] >= 0.90,
            "minimum_ratio": result["per_mlp_ratio_min"] is not None and result["per_mlp_ratio_min"] >= 0.70,
            "three_rep_bias_proxy": result["three_rep_squared_bias_proxy_mean"] is not None and result["three_rep_squared_bias_proxy_mean"] <= 1e-6,
        }
    result["gates"] = checks
    result["pass"] = bool(all(checks.values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary = _summarize(_load(args.input), args.stage)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# Cross-output EB Stage {args.stage.upper()} aggregate",
        "",
        f"- Rows returned: {summary['rows_returned']}",
        f"- Valid checksums: {summary['valid_checksums']}",
        f"- Current MSE mean: {summary['current_mse_mean']}",
        f"- Candidate MSE mean: {summary['candidate_mse_mean']}",
        f"- Global current/candidate ratio: {summary['current_candidate_global_ratio']}",
        f"- Per-MLP ratio median/q10/min: {summary['per_mlp_ratio_median']} / {summary['per_mlp_ratio_q10']} / {summary['per_mlp_ratio_min']}",
        f"- Lambda EB mean/median/q10/min/max: {summary['lambda_eb_mean']} / {summary['lambda_eb_median']} / {summary['lambda_eb_q10']} / {summary['lambda_eb_min']} / {summary['lambda_eb_max']}",
        f"- Ridge condition max observed: {summary['ridge_condition_max_observed']}",
        f"- Ridge condition mean / ridge lambda mean: {summary['ridge_condition_mean']} / {summary['ridge_lambda_mean']}",
        f"- Feature scale min/max means: {summary['feature_scale_min_mean']} / {summary['feature_scale_max_mean']}",
        f"- Three-rep squared-bias proxy mean: {summary['three_rep_squared_bias_proxy_mean']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in summary["gates"].items())
    lines.extend(["", f"**Verdict: {'PASS' if summary['pass'] else 'FAIL'}**"])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
