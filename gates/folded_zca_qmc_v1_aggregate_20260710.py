#!/usr/bin/env python3
"""Aggregate folded ZCA QMC/LHS Stage-A results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("current", "lhs_base", "lhs_zca", "sobol_base", "sobol_zca")
CANDIDATES = (("lhs", "lhs_base", "lhs_zca"), ("sobol", "sobol_base", "sobol_zca"))


def _stats(values) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9)), "min": float(np.min(x)), "max": float(np.max(x))}


def _read(path: Path):
    rows = {}
    failures = 0
    duplicate_indices = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            index = int(row["mlp_index"])
            if index in rows:
                duplicate_indices.append(index)
            else:
                rows[index] = row
        else:
            failures += 1
    ordered = [rows[i] for i in sorted(rows)]
    return ordered, failures, duplicate_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, failures, duplicate_indices = _read(args.jsonl)
    indices = [int(r["mlp_index"]) for r in rows]
    complete = len(rows) == 100 and indices == list(range(100))
    checksums = complete and all(bool(r.get("checksum_ok")) for r in rows)
    mse = {m: [float(r["reps"][0]["mse"][m]) for r in rows] for m in METHODS}
    means = {m: float(np.mean(v)) if v else float("inf") for m, v in mse.items()}
    ratio_stats = {}
    for family, base, candidate in CANDIDATES:
        ratio_stats[f"current_over_{candidate}"] = _stats(np.asarray(mse["current"]) / np.maximum(mse[candidate], 1e-300)) if rows else {}
        ratio_stats[f"{base}_over_{candidate}"] = _stats(np.asarray(mse[base]) / np.maximum(mse[candidate], 1e-300)) if rows else {}

    diagnostic_summary = {}
    for family, base, candidate in CANDIDATES:
        diagnostic_summary[family] = {}
        if rows:
            vals = [r["reps"][0]["diagnostics"][family]["zca"] for r in rows]
            keys = ("pre_cov_rel_fro", "pre_diag_max_abs_error", "pre_offdiag_rms", "pre_offdiag_max_abs", "post_cov_rel_fro", "zca_post_cov_rel_fro", "post_diag_max_abs_error", "post_offdiag_rms", "post_offdiag_max_abs", "condition", "pre_radius_mean", "pre_radius_std", "post_radius_rms", "base_antipode_max_abs", "zca_antipode_max_abs", "preactivation_cov_rel_fro_base", "preactivation_cov_rel_fro_zca")
            for key in keys:
                diagnostic_summary[family][key] = _stats([v[key] for v in vals])
            diagnostic_summary[family]["eigenvalues"] = {
                field: _stats([v["eigenvalues"][field] for v in vals])
                for field in ("min", "median", "max")
            }

    integrity_gates = {"complete_100": complete, "checksums": checksums, "no_failures": failures == 0, "no_duplicates": len(duplicate_indices) == 0}
    candidate_gates = {}
    for family, base, candidate in CANDIDATES:
        prefix = family + "_zca"
        candidate_gates[family] = {
            "mse": means.get(candidate, float("inf")) <= 1.8e-6,
            "current_global": means.get("current", 0.0) / max(means.get(candidate, float("inf")), 1e-300) >= 1.25,
            "current_median": ratio_stats.get(f"current_over_{candidate}", {}).get("median", -float("inf")) >= 1.10,
            "current_q10": ratio_stats.get(f"current_over_{candidate}", {}).get("q10", -float("inf")) >= .85,
            "current_min": ratio_stats.get(f"current_over_{candidate}", {}).get("min", -float("inf")) >= .65,
            "beats_base_global": means.get(base, 0.0) / max(means.get(candidate, float("inf")), 1e-300) >= 1.15,
            "post_cov_max": diagnostic_summary.get(family, {}).get("post_cov_rel_fro", {}).get("max", float("inf")) <= 1e-4,
            "antipode_max": diagnostic_summary.get(family, {}).get("zca_antipode_max_abs", {}).get("max", float("inf")) <= 1e-5,
        }

    flat_gates = dict(integrity_gates)
    for family, family_gates in candidate_gates.items():
        flat_gates.update({f"{family}_zca_{name}": value for name, value in family_gates.items()})
    candidate_verdicts = {family: "PASS" if all(values.values()) else "FAIL" for family, values in candidate_gates.items()}
    overall_pass = all(integrity_gates.values()) and any(verdict == "PASS" for verdict in candidate_verdicts.values())

    result = {"script_version": "folded-zca-qmc-v1-aggregate", "n_mlps": len(rows), "failures": failures, "duplicate_count": len(duplicate_indices), "duplicate_indices": sorted(set(duplicate_indices)), "complete": complete, "checksums": checksums, "verdict": "PASS" if overall_pass else "FAIL", "integrity_gates": integrity_gates, "candidate_gates": candidate_gates, "candidate_verdicts": candidate_verdicts, "gates": flat_gates, "mean_mse": means, "mse_stats": {m: _stats(v) for m, v in mse.items()} if rows else {}, "ratio_stats": ratio_stats, "diagnostics": diagnostic_summary}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Folded ZCA Gaussian-QMC/LHS v1 Stage A", "", f"Rows: `{len(rows)}/100`; failures: `{failures}`; duplicates: `{len(duplicate_indices)}`; checksums: `{'PASS' if checksums else 'FAIL'}`.", "", f"**{result['verdict']}**", "", "| method | mean MSE | median | q10 | min |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        s = result["mse_stats"].get(method, {})
        lines.append(f"| {method} | {means.get(method, float('nan')):.6e} | {s.get('median', float('nan')):.6e} | {s.get('q10', float('nan')):.6e} | {s.get('min', float('nan')):.6e} |")
    lines += ["", f"Ratios: `{json.dumps(ratio_stats, sort_keys=True)}`", "", f"Diagnostics: `{json.dumps(diagnostic_summary, sort_keys=True)}`", "", f"Candidate verdicts: `{json.dumps(candidate_verdicts, sort_keys=True)}`", "", "## Gates"]
    lines.extend(f"- `{k}`: **{'PASS' if v else 'FAIL'}**." for k, v in flat_gates.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_mlps": len(rows), "failures": failures, "verdict": result["verdict"], "gates": flat_gates}, indent=2))


if __name__ == "__main__":
    main()
