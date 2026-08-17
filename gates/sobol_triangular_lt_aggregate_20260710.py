#!/usr/bin/env python3
"""Aggregate the preregistered Sobol triangular Stage-A payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("current", "sobol_sphere", "sobol_triangular")


def _stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _read(path: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        row = obj.get("result") if isinstance(obj.get("result"), dict) else obj
        if isinstance(row, dict) and row.get("ok") is True:
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
    mse_by_mlp = {
        method: [float(row["mse"][method]) for row in rows]
        for method in METHODS
    }
    mean_mse = {method: float(np.mean(values)) for method, values in mse_by_mlp.items()}
    ratios = {
        "current_over_triangular": [
            current / max(triangular, 1e-300)
            for current, triangular in zip(
                mse_by_mlp["current"], mse_by_mlp["sobol_triangular"]
            )
        ],
        "sobol_over_triangular": [
            sobol / max(triangular, 1e-300)
            for sobol, triangular in zip(
                mse_by_mlp["sobol_sphere"], mse_by_mlp["sobol_triangular"]
            )
        ],
    }
    ratio_stats = {name: _stats(values) for name, values in ratios.items()}
    importance_shares = {
        str(k): _stats([float(row["importance_shares"][str(k)]) for row in rows])
        for k in (1, 2, 4, 8, 16, 32)
    }
    qr_abs = [float(row["qr_error"]["max_abs"]) for row in rows]
    qr_rel = [float(row["qr_error"]["max_rel"]) for row in rows]
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "triangular_mean_mse": mean_mse.get("sobol_triangular", float("inf")) <= 1.8e-6,
        "current_over_triangular_global": mean_mse.get("current", 0.0)
        / max(mean_mse.get("sobol_triangular", float("inf")), 1e-300)
        >= 1.25,
        "sobol_over_triangular_global": mean_mse.get("sobol_sphere", 0.0)
        / max(mean_mse.get("sobol_triangular", float("inf")), 1e-300)
        >= 1.15,
        "current_over_triangular_median": ratio_stats["current_over_triangular"].get("median", 0.0) >= 1.10,
        "current_over_triangular_q10": ratio_stats["current_over_triangular"].get("q10", 0.0) >= 0.85,
        "all_ratios_min": min(
            ratio_stats["current_over_triangular"].get("min", 0.0),
            ratio_stats["sobol_over_triangular"].get("min", 0.0),
        ) >= 0.65,
    }
    result = {
        "script_version": "sobol-triangular-lt-aggregate-v1",
        "n_mlps": len(rows),
        "complete": complete,
        "checksums": checksums,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mean_mse": mean_mse,
        "mse_stats_per_mlp": {method: _stats(values) for method, values in mse_by_mlp.items()},
        "ratio_stats_per_mlp": ratio_stats,
        "importance_share_stats": importance_shares,
        "qr_error_stats": {"max_abs": _stats(qr_abs), "max_rel": _stats(qr_rel)},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Sobol triangular linear-transform Stage-A gate",
        "",
        f"Successful shards: `{len(rows)}/100`; checksums: `{'PASS' if checksums else 'FAIL'}`.",
        "",
        f"**{result['verdict']}**",
        "",
        "| method | mean MSE | median | q10 | q90 | worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        stats = result["mse_stats_per_mlp"][method]
        lines.append(
            f"| {method} | {mean_mse[method]:.6e} | {stats['median']:.6e} | "
            f"{stats['q10']:.6e} | {stats['q90']:.6e} | {stats['max']:.6e} |"
        )
    lines += [
        "",
        "| ratio | mean | median | q10 | q90 | min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in ratio_stats.items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['q10']:.4f} | {stats['q90']:.4f} | {stats['min']:.4f} |"
        )
    lines += ["", "## Importance concentration", "", "| top k | mean | median | q10 | q90 |", "|---:|---:|---:|---:|---:|"]
    for k, stats in importance_shares.items():
        lines.append(f"| {k} | {stats['mean']:.6f} | {stats['median']:.6f} | {stats['q10']:.6f} | {stats['q90']:.6f} |")
    lines += [
        "",
        f"QR max-absolute error: mean `{(float(np.mean(qr_abs)) if qr_abs else float('nan')):.6e}`, worst `{(max(qr_abs) if qr_abs else float('nan')):.6e}`.",
        f"QR max-relative error: mean `{(float(np.mean(qr_rel)) if qr_rel else float('nan')):.6e}`, worst `{(max(qr_rel) if qr_rel else float('nan')):.6e}`.",
        "",
        "## Stage-A gate decisions",
        "",
    ]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_mlps": len(rows), "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
