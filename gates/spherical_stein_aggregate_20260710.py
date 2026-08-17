#!/usr/bin/env python3
"""Aggregate the spherical Stein Haar fold-CV gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("current", "raw_haar", "stein")


def _stats(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9)), "min": float(np.min(x)), "max": float(np.max(x))}


def _read(path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok"):
                rows[int(row["mlp_index"])] = row
    return [rows[i] for i in sorted(rows)]


def _decomp(row, method):
    truth = np.asarray(row["truth_final"], float)
    est = np.asarray([rep["estimates"][method] for rep in row["reps"]], float)
    mean = np.mean(est, axis=0)
    bias = float(np.mean((mean - truth) ** 2))
    var = float(np.mean(np.mean((est - mean[None, :]) ** 2, axis=1)))
    return {"bias_squared": bias, "variance": var, "total": bias + var}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.jsonl)
    complete = len(rows) == 100 and [int(r["mlp_index"]) for r in rows] == list(range(100))
    checksums = bool(rows) and all(bool(r.get("checksum_ok")) for r in rows)
    mse_by_mlp = [{m: float(np.mean([rep["mse"][m] for rep in r["reps"]])) for m in METHODS} for r in rows]
    ratios = {"current_over_stein": [x["current"] / max(x["stein"], 1e-300) for x in mse_by_mlp], "raw_over_stein": [x["raw_haar"] / max(x["stein"], 1e-300) for x in mse_by_mlp]}
    mean_mse = {m: float(np.mean([x[m] for x in mse_by_mlp])) for m in METHODS}
    decomp = {m: {c: _stats([_decomp(r, m)[c] for r in rows]) for c in ("bias_squared", "variance", "total")} for m in METHODS}
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "stein_mean_mse": mean_mse.get("stein", float("inf")) <= 1.6e-6,
        "current_over_stein_mean": mean_mse.get("current", 0.0) / max(mean_mse.get("stein", float("inf")), 1e-300) >= 1.35,
        "current_over_stein_median": _stats(ratios["current_over_stein"]).get("median", 0.0) >= 1.20,
        "current_over_stein_q10": _stats(ratios["current_over_stein"]).get("q10", 0.0) > 0.90,
        "all_ratio_min": min(_stats(ratios["current_over_stein"])["min"], _stats(ratios["raw_over_stein"])["min"]) >= 0.70,
        "stein_bias": decomp["stein"]["bias_squared"]["mean"] <= 1.0e-6,
    }
    result = {"script_version": "spherical-stein-aggregate-v1", "n_mlps": len(rows), "complete": complete, "checksums": checksums, "verdict": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "mean_mse_per_mlp_then_reps": mean_mse, "mse_stats": {m: _stats([x[m] for x in mse_by_mlp]) for m in METHODS}, "ratio_stats": {k: _stats(v) for k, v in ratios.items()}, "decomposition": decomp}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Spherical Stein Haar fold-CV gate", "", f"Successful shards: `{len(rows)}/100`; checksums: `{'PASS' if checksums else 'FAIL'}`.", "", f"**{result['verdict']}**", "", "| method | mean MSE | median | q10 | q90 |", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        s = result["mse_stats"][m]
        lines.append(f"| {m} | {mean_mse[m]:.6e} | {s['median']:.6e} | {s['q10']:.6e} | {s['q90']:.6e} |")
    lines += ["", "| ratio | mean | median | q10 | q90 | min |", "|---|---:|---:|---:|---:|---:|"]
    for k, s in result["ratio_stats"].items():
        lines.append(f"| {k} | {s['mean']:.4f} | {s['median']:.4f} | {s['q10']:.4f} | {s['q90']:.4f} | {s['min']:.4f} |")
    lines += ["", "## Gate decisions", ""] + [f"- `{k}`: **{'PASS' if v else 'FAIL'}**." for k, v in gates.items()]
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_mlps": len(rows), "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
