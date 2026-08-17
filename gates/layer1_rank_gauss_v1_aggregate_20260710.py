#!/usr/bin/env python3
"""Aggregate the pre-registered layer-1 Gaussian rank gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("current", "rank_gaussian")


def _stats(values) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)), "q90": float(np.quantile(x, 0.9)), "min": float(np.min(x)), "max": float(np.max(x))}


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
    checksums = len(rows) == 100 and all(bool(row.get("checksum_ok")) for row in rows)
    mse = {method: [float(row["reps"][0]["mse"][method]) for row in rows] for method in METHODS}
    means = {method: float(np.mean(values)) if values else float("inf") for method, values in mse.items()}
    ratios = np.asarray(mse["current"]) / np.maximum(np.asarray(mse["rank_gaussian"]), 1e-300) if rows else np.array([])
    ratio_stats = _stats(ratios) if rows else {key: float("nan") for key in ("mean", "median", "q10", "q90", "min", "max")}
    diag = {}
    for method in METHODS:
        diag[method] = {}
        if rows:
            keys = rows[0]["reps"][0]["diagnostics"]["relu"][method].keys()
            for key in keys:
                values = [
                    row["reps"][0]["diagnostics"]["relu"][method][key]
                    for row in rows
                ]
                if isinstance(values[0], dict):
                    diag[method][key] = {
                        field: _stats([value[field] for value in values])
                        for field in values[0]
                    }
                else:
                    diag[method][key] = _stats(values)
    transport_diag = {}
    if rows:
        dkeys = rows[0]["reps"][0]["diagnostics"].keys()
        for key in dkeys:
            if key == "relu":
                continue
            vals = [row["reps"][0]["diagnostics"][key] for row in rows]
            if isinstance(vals[0], dict):
                transport_diag[key] = {
                    field: _stats([value[field] for value in vals])
                    for field in vals[0]
                }
            elif isinstance(vals[0], bool):
                transport_diag[key] = all(bool(v) for v in vals)
            else:
                transport_diag[key] = _stats(vals)
    gates = {
        "complete_100": complete,
        "checksums": checksums,
        "no_failures": failures == 0,
        "candidate_mse": means.get("rank_gaussian", float("inf")) <= 1.8e-6,
        "current_over_candidate_global": means.get("current", 0.0) / max(means.get("rank_gaussian", float("inf")), 1e-300) >= 1.25,
        "current_over_candidate_median": ratio_stats["median"] >= 1.10,
        "current_over_candidate_q10": ratio_stats["q10"] >= 0.85,
        "current_over_candidate_min": ratio_stats["min"] >= 0.65,
        "transported_antipode_max": transport_diag.get("transported_antipode_max_abs", {"max": float("inf")}).get("max", float("inf")) <= 1e-5,
        "transport_sorted_target_exact": transport_diag.get("transport_sorted_target_max_abs", {"max": float("inf")}).get("max", float("inf")) == 0.0,
        "magnitude_rank_roundtrip_exact": transport_diag.get("magnitude_rank_roundtrip_max_abs", {"max": float("inf")}).get("max", float("inf")) == 0.0,
        "no_zero_values": transport_diag.get("zero_value_count", {"max": 1}).get("max", 1) == 0,
        "negligible_magnitude_ties": transport_diag.get("raw_magnitude_tie_fraction", {"max": float("inf")}).get("max", float("inf")) <= 1e-3,
    }
    result = {
        "script_version": "layer1-rank-gauss-aggregate-v1",
        "n_mlps": len(rows), "failures": failures, "complete": complete, "checksums": checksums,
        "verdict": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "mean_mse": means, "mse_stats": {method: _stats(values) for method, values in mse.items()} if rows else {},
        "current_over_rank_gaussian": ratio_stats, "transport_diagnostics": transport_diag,
        "relu_diagnostics": diag,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Layer-1 Gaussian rank transport gate", "", f"Rows: `{len(rows)}/100`; failures: `{failures}`; checksums: `{'PASS' if checksums else 'FAIL'}`.", "", f"**{result['verdict']}**", "", "| method | mean MSE | median | q10 | min |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        s = result["mse_stats"].get(method, {})
        lines.append(f"| {method} | {means.get(method, float('nan')):.6e} | {s.get('median', float('nan')):.6e} | {s.get('q10', float('nan')):.6e} | {s.get('min', float('nan')):.6e} |")
    lines += ["", f"Current/candidate ratio: `{json.dumps(ratio_stats, sort_keys=True)}`", "", "## Transport and pre-recolor diagnostics", "", f"`{json.dumps(transport_diag, sort_keys=True)}`", "", f"`{json.dumps(diag, sort_keys=True)}`", "", "## Gate decisions", ""]
    lines.extend(f"- `{name}`: **{'PASS' if value else 'FAIL'}**." for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_mlps": len(rows), "failures": failures, "verdict": result["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
