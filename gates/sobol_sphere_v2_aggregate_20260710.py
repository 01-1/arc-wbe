#!/usr/bin/env python3
"""Aggregate the v2 paired Sobol-sphere Fly gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("current", "iid_sphere_recolor", "sobol_sphere_recolor")


def _stats(values) -> dict[str, float]:
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


def _read(path: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok"):
                rows[int(row["mlp_index"])] = row
    return [rows[i] for i in sorted(rows)]


def _decomp(row: dict[str, object], method: str) -> dict[str, float]:
    truth = np.asarray(row["truth_final"], dtype=float)
    estimates = np.asarray(
        [rep["estimates"][method] for rep in row["reps"]], dtype=float
    )
    mean_estimate = np.mean(estimates, axis=0)
    bias_squared = float(np.mean((mean_estimate - truth) ** 2))
    variance = float(np.mean(np.mean((estimates - mean_estimate[None, :]) ** 2, axis=1)))
    return {"bias_squared": bias_squared, "variance": variance, "total": bias_squared + variance}


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
    mse_by_mlp = {
        method: [
            float(np.mean([rep["mse"][method] for rep in row["reps"]]))
            for row in rows
        ]
        for method in METHODS
    }
    mean_mse = {
        method: (float(np.mean(values)) if values else None)
        for method, values in mse_by_mlp.items()
    }
    ratios = {
        "current_over_sobol": [
            current / max(sobol, 1e-300)
            for current, sobol in zip(
                mse_by_mlp["current"], mse_by_mlp["sobol_sphere_recolor"]
            )
        ],
        "iid_over_sobol": [
            iid / max(sobol, 1e-300)
            for iid, sobol in zip(
                mse_by_mlp["iid_sphere_recolor"], mse_by_mlp["sobol_sphere_recolor"]
            )
        ],
    }
    decompositions = {
        method: {
            component: _stats([_decomp(row, method)[component] for row in rows])
            for component in ("bias_squared", "variance", "total")
        }
        for method in METHODS
    }
    ratio_stats = {name: _stats(values) for name, values in ratios.items()}
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "sobol_mean_mse": (
            mean_mse["sobol_sphere_recolor"] is not None
            and mean_mse["sobol_sphere_recolor"] <= 1.6e-6
        ),
        "current_over_sobol_global": (
            mean_mse["current"] / max(mean_mse["sobol_sphere_recolor"], 1e-300)
            if mean_mse["current"] is not None and mean_mse["sobol_sphere_recolor"] is not None
            else False
        )
        >= 1.35,
        "current_over_sobol_median": ratio_stats["current_over_sobol"].get("median", 0.0) >= 1.20,
        "current_over_sobol_q10": ratio_stats["current_over_sobol"].get("q10", 0.0) >= 0.90,
        "current_over_sobol_min": ratio_stats["current_over_sobol"].get("min", 0.0) >= 0.70,
        "sobol_bias_proxy": decompositions["sobol_sphere_recolor"]["bias_squared"].get(
            "mean", float("inf")
        )
        <= 1e-6,
    }
    result = {
        "script_version": "sobol-sphere-aggregate-v2",
        "n_mlps": len(rows),
        "complete": complete,
        "checksums": checksums,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mean_mse_per_mlp_then_reps": mean_mse,
        "mse_stats_per_mlp": {method: _stats(values) for method, values in mse_by_mlp.items()},
        "ratio_stats_per_mlp": ratio_stats,
        "decomposition_3rep": decompositions,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Sobol-sphere recolor v2 gate",
        "",
        "The v1 gate produced zero estimates because the stable image lacked SciPy. "
        "This v2 run uses the reviewed dependency-free generator, validated against "
        "SciPy 1.16.2 at the Sobol-uniform level with maximum error 0.0.",
        "",
        f"Successful shards: `{len(rows)}/100`; checksums: `{'PASS' if checksums else 'FAIL'}`.",
        "",
        f"**{result['verdict']}**",
        "",
        "| method | mean MSE | median | q10 | q90 | min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        lines.append("| no successful rows | n/a | n/a | n/a | n/a | n/a |")
    for method in METHODS:
        stats = result["mse_stats_per_mlp"][method]
        mean_text = "n/a" if mean_mse[method] is None else f"{mean_mse[method]:.6e}"
        lines.append(
            f"| {method} | {mean_text} | "
            f"{stats.get('median', 'n/a') if stats else 'n/a'} | "
            f"{stats.get('q10', 'n/a') if stats else 'n/a'} | "
            f"{stats.get('q90', 'n/a') if stats else 'n/a'} | "
            f"{stats.get('min', 'n/a') if stats else 'n/a'} |"
        )
    lines += [
        "",
        "| ratio | mean | median | q10 | q90 | min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in ratio_stats.items():
        lines.append(
            f"| {name} | {stats.get('mean', 'n/a')} | {stats.get('median', 'n/a')} | "
            f"{stats.get('q10', 'n/a')} | {stats.get('q90', 'n/a')} | "
            f"{stats.get('min', 'n/a')} |"
        )
    lines += ["", "## 3-rep bias/variance", ""]
    if not rows:
        lines.append("No 3-rep estimates were returned.")
    else:
        for method in METHODS:
            d = decompositions[method]
            lines.append(
                f"- `{method}`: bias² mean `{d['bias_squared']['mean']:.6e}`, "
                f"variance mean `{d['variance']['mean']:.6e}`, total mean `{d['total']['mean']:.6e}`."
            )
    lines += ["", "## Gate decisions", ""]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_mlps": len(rows), "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
