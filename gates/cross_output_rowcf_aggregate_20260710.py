#!/usr/bin/env python3
"""Aggregate the preregistered row-cross-fitted James–Stein gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9)), "min": float(np.min(x)), "max": float(np.max(x))}


def _load(path: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            row = obj.get("result") if isinstance(obj.get("result"), dict) else obj
            if isinstance(row, dict) and row.get("ok") is True:
                rows[int(row["mlp_index"])] = row
    return [rows[i] for i in sorted(rows)]


def _summarize(rows: list[dict[str, object]], stage: str) -> dict[str, object]:
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == 100 and indices == list(range(100))
    checksums = bool(rows) and all(bool(row.get("checksum_ok")) for row in rows)
    rep_rows = [rep for row in rows for rep in row.get("rep_results", [])]
    current = np.asarray([float(rep["current_mse"]) for rep in rep_rows], dtype=np.float64)
    candidate = np.asarray([float(rep["candidate_mse"]) for rep in rep_rows], dtype=np.float64)
    by_mlp: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_mlp[int(row["mlp_index"])] = list(row.get("rep_results", []))
    ratios = np.asarray([
        np.mean([float(rep["current_mse"]) for rep in reps])
        / max(np.mean([float(rep["candidate_mse"]) for rep in reps]), 1e-300)
        for reps in by_mlp.values()
    ], dtype=np.float64)
    lambdas = np.asarray([float(rep["lambda_mean"]) for rep in rep_rows], dtype=np.float64)
    lambda_a = np.asarray([float(rep["lambda_a"]) for rep in rep_rows], dtype=np.float64)
    lambda_b = np.asarray([float(rep["lambda_b"]) for rep in rep_rows], dtype=np.float64)
    predictor_discrepancy = np.asarray([float(rep["predictor_discrepancy"]) for rep in rep_rows], dtype=np.float64)
    bias_proxy = [float(row["three_rep_squared_bias_proxy"]) for row in rows if row.get("three_rep_squared_bias_proxy") is not None]
    result: dict[str, object] = {
        "stage": stage,
        "rows_returned": len(rows),
        "valid_checksums": sum(bool(row.get("checksum_ok")) for row in rows),
        "rep_rows": len(rep_rows),
        "complete_100": complete,
        "checksums": checksums,
        "current_mse_mean": float(np.mean(current)) if len(current) else None,
        "candidate_mse_mean": float(np.mean(candidate)) if len(candidate) else None,
        "current_candidate_global_ratio": float(np.mean(current) / max(np.mean(candidate), 1e-300)) if len(current) else None,
        "per_mlp_ratio_stats": _stats(ratios.tolist()) if len(ratios) else None,
        "lambda_mean_stats": _stats(lambdas.tolist()) if len(lambdas) else None,
        "lambda_a_stats": _stats(lambda_a.tolist()) if len(lambda_a) else None,
        "lambda_b_stats": _stats(lambda_b.tolist()) if len(lambda_b) else None,
        "predictor_discrepancy_stats": _stats(predictor_discrepancy.tolist()) if len(predictor_discrepancy) else None,
        "three_rep_squared_bias_proxy_mean": float(np.mean(bias_proxy)) if bias_proxy else None,
    }
    if stage == "a":
        gates = {
            "100_checksums": complete and checksums,
            "candidate_mse": result["candidate_mse_mean"] is not None and result["candidate_mse_mean"] <= 1.8e-6,
            "global_ratio": result["current_candidate_global_ratio"] is not None and result["current_candidate_global_ratio"] >= 1.25,
            "median_ratio": result["per_mlp_ratio_stats"] is not None and result["per_mlp_ratio_stats"]["median"] >= 1.10,
            "q10_ratio": result["per_mlp_ratio_stats"] is not None and result["per_mlp_ratio_stats"]["q10"] >= 0.85,
            "minimum_ratio": result["per_mlp_ratio_stats"] is not None and result["per_mlp_ratio_stats"]["min"] >= 0.65,
            "mean_lambda_range": result["lambda_mean_stats"] is not None and 0.02 <= result["lambda_mean_stats"]["mean"] <= 0.95,
        }
    else:
        gates = {
            "100_checksums": complete and checksums,
            "candidate_mse": result["candidate_mse_mean"] is not None and result["candidate_mse_mean"] <= 1.6e-6,
            "global_ratio": result["current_candidate_global_ratio"] is not None and result["current_candidate_global_ratio"] >= 1.35,
            "median_ratio": result["per_mlp_ratio_stats"] is not None and result["per_mlp_ratio_stats"]["median"] >= 1.20,
            "q10_ratio": result["per_mlp_ratio_stats"] is not None and result["per_mlp_ratio_stats"]["q10"] >= 0.90,
            "minimum_ratio": result["per_mlp_ratio_stats"] is not None and result["per_mlp_ratio_stats"]["min"] >= 0.70,
            "three_rep_bias_proxy": result["three_rep_squared_bias_proxy_mean"] is not None and result["three_rep_squared_bias_proxy_mean"] <= 1e-6,
        }
    result["gates"] = gates
    result["pass"] = bool(all(gates.values()))
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
    lines = [f"# Cross-output row-CF Stage {args.stage.upper()} aggregate", "", f"- Rows returned: {summary['rows_returned']}", f"- Valid checksums: {summary['valid_checksums']}", f"- Rep rows: {summary['rep_rows']}", f"- Current MSE mean: {summary['current_mse_mean']}", f"- Candidate MSE mean: {summary['candidate_mse_mean']}", f"- Global current/candidate ratio: {summary['current_candidate_global_ratio']}", f"- Per-MLP ratio mean/median/q10/min: {summary['per_mlp_ratio_stats']}", f"- Lambda mean/median/q10/min/max: {summary['lambda_mean_stats']}", f"- Lambda A stats: {summary['lambda_a_stats']}", f"- Lambda B stats: {summary['lambda_b_stats']}", f"- Predictor discrepancy stats: {summary['predictor_discrepancy_stats']}", f"- Three-rep squared-bias proxy mean: {summary['three_rep_squared_bias_proxy_mean']}", "", "## Gates", ""]
    lines.extend(f"- `{name}`: **{'PASS' if ok else 'FAIL'}**" for name, ok in summary["gates"].items())
    lines.extend(["", f"**Verdict: {'PASS' if summary['pass'] else 'FAIL'}**"])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
