#!/usr/bin/env python3
"""Aggregate the preregistered Haar-sphere fold-CV Fly payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("current", "raw_haar", "haar_cv")


def qstats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "q10": float(np.quantile(x, 0.10)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _rows(path: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        row = obj.get("result") if isinstance(obj.get("result"), dict) else obj
        if isinstance(row, dict) and row.get("ok") is True and "reps" in row:
            rows[int(row["mlp_index"])] = row
    return [rows[index] for index in sorted(rows)]


def _decomposition(row: dict[str, object], method: str) -> dict[str, float]:
    truth = np.asarray(row["truth_final"], dtype=np.float64)
    reps = row["reps"]
    estimates = np.asarray(
        [rep["estimates"][method] for rep in reps], dtype=np.float64
    )
    mean_estimate = np.mean(estimates, axis=0)
    bias_sq = float(np.mean((mean_estimate - truth) ** 2))
    variance = float(np.mean(np.mean((estimates - mean_estimate[None, :]) ** 2, axis=1)))
    return {"bias_squared": bias_sq, "variance": variance, "total": bias_sq + variance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = _rows(args.jsonl)
    complete = len(rows) == 100 and [int(row["mlp_index"]) for row in rows] == list(range(100))
    checksums = bool(rows) and all(bool(row.get("checksum_ok")) for row in rows)
    per_mlp: list[dict[str, object]] = []
    for row in rows:
        rep_mse = {
            method: [float(rep["mse"][method]) for rep in row["reps"]]
            for method in METHODS
        }
        mean_mse = {method: float(np.mean(values)) for method, values in rep_mse.items()}
        per_mlp.append(
            {
                "mlp_index": int(row["mlp_index"]),
                "mse": mean_mse,
                "rep_mse": rep_mse,
                "current_over_cv": mean_mse["current"] / max(mean_mse["haar_cv"], 1e-300),
                "raw_over_cv": mean_mse["raw_haar"] / max(mean_mse["haar_cv"], 1e-300),
                "decomposition": {
                    method: _decomposition(row, method) for method in METHODS
                },
            }
        )

    mean_mse = {
        method: float(np.mean([item["mse"][method] for item in per_mlp]))
        for method in METHODS
    }
    exact_rep_mean_mse = {
        method: float(
            np.mean(
                [mse for item in per_mlp for mse in item["rep_mse"][method]]
            )
        )
        for method in METHODS
    }
    current_over_cv = [float(item["current_over_cv"]) for item in per_mlp]
    raw_over_cv = [float(item["raw_over_cv"]) for item in per_mlp]
    ratio_stats = {
        "current_over_cv": qstats(current_over_cv),
        "raw_over_cv": qstats(raw_over_cv),
    }
    mse_stats = {
        method: qstats([float(item["mse"][method]) for item in per_mlp])
        for method in METHODS
    }
    decomposition = {
        method: {
            component: qstats(
                [float(item["decomposition"][method][component]) for item in per_mlp]
            )
            for component in ("bias_squared", "variance", "total")
        }
        for method in METHODS
    }

    cv_mean_pass = mean_mse.get("haar_cv", float("inf")) <= 1.6e-6
    current_ratio_pass = mean_mse.get("current", 0.0) / max(mean_mse.get("haar_cv", float("inf")), 1e-300) >= 1.35
    median_pass = ratio_stats.get("current_over_cv", {}).get("median", 0.0) >= 1.20
    q10_pass = ratio_stats.get("current_over_cv", {}).get("q10", 0.0) >= 0.90
    no_bad_ratio_pass = min(
        ratio_stats.get("current_over_cv", {}).get("min", 0.0),
        ratio_stats.get("raw_over_cv", {}).get("min", 0.0),
    ) >= 0.70
    bias_pass = decomposition.get("haar_cv", {}).get("bias_squared", {}).get("mean", float("inf")) <= 1.0e-6
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "haar_cv_mean_mse": cv_mean_pass,
        "current_over_cv_mean_ratio": current_ratio_pass,
        "current_over_cv_median_ratio": median_pass,
        "current_over_cv_q10_ratio": q10_pass,
        "all_ratios_at_least_0.70": no_bad_ratio_pass,
        "haar_cv_mean_squared_bias": bias_pass,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "script_version": "haar-sphere-foldcv-aggregate-v1",
        "n_mlps": len(rows),
        "complete": complete,
        "checksums": checksums,
        "verdict": verdict,
        "gates": gates,
        "mean_mse_per_mlp_then_reps": mean_mse,
        "exact_rep_mean_mse": exact_rep_mean_mse,
        "mse_stats_per_mlp": mse_stats,
        "ratio_stats": ratio_stats,
        "decomposition": decomposition,
        "per_mlp": per_mlp,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Haar-sphere first-layer fold-CV gate",
        "",
        f"Successful shards: `{len(rows)}/100`; checksums: `{'PASS' if checksums else 'FAIL'}`.",
        "",
        f"**{verdict}**",
        "",
        "## Exact MSE summary",
        "",
        "| method | exact rep mean MSE | per-MLP mean MSE | median | q10 | q90 | worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        stats = mse_stats.get(method, {})
        lines.append(
            f"| {method} | {exact_rep_mean_mse.get(method, float('nan')):.6e} | "
            f"{mean_mse.get(method, float('nan')):.6e} | {stats.get('median', float('nan')):.6e} | "
            f"{stats.get('q10', float('nan')):.6e} | {stats.get('q90', float('nan')):.6e} | "
            f"{stats.get('max', float('nan')):.6e} |"
        )
    lines.extend(
        [
            "",
            "## MSE ratio summary",
            "",
            "| ratio | mean | median | q10 | q90 | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in ratio_stats.items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['q10']:.4f} | "
            f"{stats['q90']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "## Three-rep Haar-CV decomposition", "", "| component | mean | median | q10 | q90 |", "|---|---:|---:|---:|---:|"])
    for component in ("bias_squared", "variance", "total"):
        stats = decomposition.get("haar_cv", {}).get(component, {})
        lines.append(
            f"| haar_cv {component} | {stats.get('mean', float('nan')):.6e} | "
            f"{stats.get('median', float('nan')):.6e} | {stats.get('q10', float('nan')):.6e} | "
            f"{stats.get('q90', float('nan')):.6e} |"
        )
    lines.extend(["", "## Frozen gate decisions", ""])
    for name, passed in gates.items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**.")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_mlps": len(rows), "verdict": verdict, "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
