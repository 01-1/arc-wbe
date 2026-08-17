#!/usr/bin/env python3
"""Aggregate the preregistered Stage-A/Stage-B paired LT gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("current", "sobol_unrotated", "sobol_lt")


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


def _read(path: Path) -> tuple[list[dict[str, object]], int, int]:
    rows: dict[int, dict[str, object]] = {}
    total = failures = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        if row.get("ok"):
            rows[int(row["mlp_index"])] = row
        else:
            failures += 1
    return [rows[i] for i in sorted(rows)], total, failures


def _ratio_stats(mse_by_mlp: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    lt = np.asarray(mse_by_mlp["sobol_lt"], dtype=float)
    ratios = {
        "current_over_lt": np.asarray(mse_by_mlp["current"], dtype=float) / np.maximum(lt, 1e-300),
        "unrotated_over_lt": np.asarray(mse_by_mlp["sobol_unrotated"], dtype=float) / np.maximum(lt, 1e-300),
    }
    return {name: _stats(values) for name, values in ratios.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows, total_rows, failures = _read(args.jsonl)
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == 100 and indices == list(range(100))
    checksums = bool(rows) and all(bool(row.get("checksum_ok")) for row in rows)
    reps = sorted({len(row["reps"]) for row in rows})
    mse_by_mlp = {
        method: [
            float(np.mean([rep["mse"][method] for rep in row["reps"]]))
            for row in rows
        ]
        for method in METHODS
    }
    mean_mse = {method: float(np.mean(values)) for method, values in mse_by_mlp.items()}
    ratio_stats = _ratio_stats(mse_by_mlp)
    decompositions = {}
    for method in METHODS:
        bias_values = []
        variance_values = []
        for row in rows:
            truth = np.asarray(row["truth_final"], dtype=float)
            estimates = np.asarray([rep["estimates"][method] for rep in row["reps"]], dtype=float)
            mean_estimate = np.mean(estimates, axis=0)
            bias_values.append(float(np.mean((mean_estimate - truth) ** 2)))
            variance_values.append(float(np.mean(np.mean((estimates - mean_estimate[None, :]) ** 2, axis=1))))
        decompositions[method] = {
            "bias_squared": _stats(bias_values),
            "variance": _stats(variance_values),
        }

    pilot_concentration = []
    for row in rows:
        for rep in row["reps"]:
            pilot_concentration.append(rep["pilot_singular_concentration_top_1_2_4_8"])
    pilot_concentration_stats = {
        f"top_{k}": _stats([values[idx] for values in pilot_concentration])
        for idx, k in enumerate((1, 2, 4, 8))
    }

    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "returned_rows_no_failures": total_rows == 100 and failures == 0,
        "one_fixed_replication_stage_shape": reps == [1] if args.stage == "A" else reps == [3],
        "lt_mean_mse": mean_mse.get("sobol_lt", float("inf")) <= (1.8e-6 if args.stage == "A" else 1.6e-6),
        "current_over_lt_global": mean_mse.get("current", 0.0) / max(mean_mse.get("sobol_lt", float("inf")), 1e-300) >= (1.25 if args.stage == "A" else 1.35),
        "unrotated_over_lt_global": mean_mse.get("sobol_unrotated", 0.0) / max(mean_mse.get("sobol_lt", float("inf")), 1e-300) >= 1.15 if args.stage == "A" else True,
        "current_over_lt_median": ratio_stats["current_over_lt"]["median"] >= (1.10 if args.stage == "A" else 1.20),
        "current_over_lt_q10": ratio_stats["current_over_lt"]["q10"] >= (0.85 if args.stage == "A" else 0.90),
        "current_over_lt_min": ratio_stats["current_over_lt"]["min"] >= (0.65 if args.stage == "A" else 0.70),
    }
    if args.stage == "B":
        gates["lt_bias_proxy"] = decompositions["sobol_lt"]["bias_squared"]["mean"] <= 1e-6

    result = {
        "script_version": "sobol-jacobian-lt-aggregate-v1",
        "stage": args.stage,
        "n_mlps": len(rows),
        "total_rows": total_rows,
        "failures": failures,
        "complete": complete,
        "checksums": checksums,
        "reps_observed": reps,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mean_mse_per_mlp_then_reps": mean_mse,
        "mse_stats_per_mlp": {method: _stats(values) for method, values in mse_by_mlp.items()},
        "ratio_stats_per_mlp": ratio_stats,
        "decomposition": decompositions,
        "pilot_singular_concentration": pilot_concentration_stats,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Sobol pilot-Jacobian LT Stage {args.stage}",
        "",
        f"Returned rows: `{len(rows)}/100`; failures: `{failures}`; checksums: `{'PASS' if checksums else 'FAIL'}`.",
        f"Observed replication counts: `{reps}`.",
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
    lines += ["", "## Bias/variance proxy", ""]
    for method in METHODS:
        d = decompositions[method]
        lines.append(
            f"- `{method}`: bias² mean `{d['bias_squared']['mean']:.6e}`, "
            f"variance mean `{d['variance']['mean']:.6e}`."
        )
    lines += ["", "## Pilot singular-value concentration", ""]
    for name, stats in pilot_concentration_stats.items():
        lines.append(f"- `{name}`: mean `{stats['mean']:.6f}`, median `{stats['median']:.6f}`.")
    lines += ["", "## Gate decisions", ""]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"stage": args.stage, "n_mlps": len(rows), "failures": failures, "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
