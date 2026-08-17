#!/usr/bin/env python3
"""Aggregate the fixed Hadamard-oriented LHS gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("current", "lhs_independent", "lhs_hadamard")


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


def _read(path: Path) -> tuple[list[dict[str, object]], int]:
    rows: dict[int, dict[str, object]] = {}
    failures = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            rows[int(row["mlp_index"])] = row
        else:
            failures += 1
    return [rows[i] for i in sorted(rows)], failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, failures = _read(args.jsonl)
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == 100 and indices == list(range(100))
    checksums = bool(rows) and all(bool(row.get("checksum_ok")) for row in rows)
    mse_by_mlp = {method: [float(row["mse"][method]) for row in rows] for method in METHODS}
    mean_mse = {method: float(np.mean(values)) for method, values in mse_by_mlp.items()}
    ratios = {
        "current_over_candidate": np.asarray(mse_by_mlp["current"]) / np.maximum(np.asarray(mse_by_mlp["lhs_hadamard"]), 1e-300),
        "independent_over_candidate": np.asarray(mse_by_mlp["lhs_independent"]) / np.maximum(np.asarray(mse_by_mlp["lhs_hadamard"]), 1e-300),
    }
    ratio_stats = {name: _stats(values) for name, values in ratios.items()}
    diagnostics = {}
    for method in ("lhs_independent", "lhs_hadamard"):
        keys = rows[0]["diagnostics"][method].keys() if rows else ()
        diagnostics[method] = {
            key: all(bool(row["diagnostics"][method][key]) for row in rows)
            if isinstance(rows[0]["diagnostics"][method][key], bool)
            else _stats([row["diagnostics"][method][key] for row in rows])
            for key in keys
        }
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "no_failures": failures == 0,
        "candidate_mse": mean_mse.get("lhs_hadamard", float("inf")) <= 1.8e-6,
        "current_over_candidate_global": mean_mse.get("current", 0.0) / max(mean_mse.get("lhs_hadamard", float("inf")), 1e-300) >= 1.25,
        "current_over_candidate_median": ratio_stats["current_over_candidate"]["median"] >= 1.10,
        "current_over_candidate_q10": ratio_stats["current_over_candidate"]["q10"] >= 0.85,
        "current_over_candidate_min": ratio_stats["current_over_candidate"]["min"] >= 0.65,
        "independent_strata": diagnostics.get("lhs_independent", {}).get("strata_exact", False),
        "hadamard_strata": diagnostics.get("lhs_hadamard", {}).get("strata_exact", False),
    }
    result = {
        "script_version": "hadamard-lhs-aggregate-v1",
        "n_mlps": len(rows),
        "failures": failures,
        "complete": complete,
        "checksums": checksums,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mean_mse_per_mlp": mean_mse,
        "mse_stats_per_mlp": {method: _stats(values) for method, values in mse_by_mlp.items()},
        "ratio_stats_per_mlp": ratio_stats,
        "diagnostic_stats": diagnostics,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Hadamard-oriented LHS gate",
        "",
        f"Rows: `{len(rows)}/100`; failures: `{failures}`; checksums: `{'PASS' if checksums else 'FAIL'}`.",
        "",
        f"**{result['verdict']}**",
        "",
        "| method | mean MSE | median | q10 | q90 | min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        stats = result["mse_stats_per_mlp"][method]
        lines.append(
            f"| {method} | {mean_mse[method]:.6e} | {stats['median']:.6e} | "
            f"{stats['q10']:.6e} | {stats['q90']:.6e} | {stats['min']:.6e} |"
        )
    lines += ["", "| ratio | mean | median | q10 | q90 | min |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats in ratio_stats.items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['q10']:.4f} | {stats['q90']:.4f} | {stats['min']:.4f} |"
        )
    lines += ["", "## LHS diagnostics", ""]
    for method, values in diagnostics.items():
        lines.append(f"- `{method}`: `{json.dumps(values, sort_keys=True)}`")
    lines += ["", "## Gate decisions", ""]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_mlps": len(rows), "failures": failures, "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
