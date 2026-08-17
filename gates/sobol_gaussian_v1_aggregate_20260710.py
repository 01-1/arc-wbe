#!/usr/bin/env python3
"""Aggregate the frozen all-100 Gaussian-Sobol Stage-A gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("current", "iid_gaussian", "sobol_gaussian")
DIAGNOSTICS = ("radius_mean", "radius_std", "radius_q10", "radius_q90", "cov_rel_fro", "cov_max_offdiag", "coordinate_mean_max_abs", "antipode_max_abs")


def _stats(values) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.1)),
        "q90": float(np.quantile(x, 0.9)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _read(path: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") is True:
                rows[int(row["mlp_index"])] = row
    return [rows[i] for i in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.jsonl)
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == 100 and indices == list(range(100))
    checksums = bool(rows) and all(bool(row.get("checksum_ok")) for row in rows)
    mse_by_mlp = {method: [float(row["reps"][0]["mse"][method]) for row in rows] for method in METHODS}
    mean_mse = {method: float(np.mean(values)) if values else None for method, values in mse_by_mlp.items()}
    ratios = {
        "current_over_sobol": [a / max(b, 1e-300) for a, b in zip(mse_by_mlp["current"], mse_by_mlp["sobol_gaussian"])],
        "iid_over_sobol": [a / max(b, 1e-300) for a, b in zip(mse_by_mlp["iid_gaussian"], mse_by_mlp["sobol_gaussian"])],
    }
    ratio_stats = {name: _stats(values) for name, values in ratios.items()}
    diagnostics = {method: {key: _stats([float(row["reps"][0]["diagnostics"][method][key]) for row in rows]) for key in DIAGNOSTICS} for method in ("sobol_gaussian", "iid_gaussian")}
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "sobol_mean_mse": mean_mse.get("sobol_gaussian", float("inf")) <= 1.8e-6,
        "current_over_sobol_global": mean_mse.get("current", 0.0) / max(mean_mse.get("sobol_gaussian", float("inf")), 1e-300) >= 1.25,
        "current_over_sobol_median": ratio_stats["current_over_sobol"].get("median", 0.0) >= 1.10,
        "current_over_sobol_q10": ratio_stats["current_over_sobol"].get("q10", 0.0) >= 0.85,
        "current_over_sobol_min": ratio_stats["current_over_sobol"].get("min", 0.0) >= 0.65,
    }
    result = {"script_version": "sobol-gaussian-v1-aggregate", "n_mlps": len(rows), "complete": complete, "checksums": checksums, "verdict": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "mean_mse": mean_mse, "mse_stats_per_mlp": {method: _stats(values) for method, values in mse_by_mlp.items()}, "ratio_stats_per_mlp": ratio_stats, "input_diagnostics": diagnostics}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Unnormalized Gaussian Sobol RQMC Stage-A gate", "", f"Successful shards: `{len(rows)}/100`; checksums: `{'PASS' if checksums else 'FAIL'}`.", "", f"**{result['verdict']}**", "", "| method | mean MSE | median | q10 | q90 | min |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        stats = result["mse_stats_per_mlp"][method]
        if not stats:
            lines.append(f"| {method} | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {method} | {mean_mse[method]:.6e} | {stats['median']:.6e} | {stats['q10']:.6e} | {stats['q90']:.6e} | {stats['min']:.6e} |")
    lines += ["", "| ratio | mean | median | q10 | q90 | min |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats in ratio_stats.items():
        if not stats:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['q10']:.4f} | {stats['q90']:.4f} | {stats['min']:.4f} |")
    lines += ["", "## Label-free input diagnostics", "", "Radius, covariance, and coordinate-mean statistics are computed on positive representatives only; `antipode_max_abs` is computed on the constructed positive/negative pair and should be exactly zero."]
    for method, entries in diagnostics.items():
        lines.append(f"### {method}")
        for key, stats in entries.items():
            lines.append(f"- `{key}` mean `{stats['mean']:.6e}`, median `{stats['median']:.6e}`, q10 `{stats['q10']:.6e}`, q90 `{stats['q90']:.6e}`.")
    lines += ["", "## Frozen gate decisions", ""]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_mlps": len(rows), "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
