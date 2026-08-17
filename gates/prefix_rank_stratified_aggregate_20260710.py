#!/usr/bin/env python3
"""Aggregate corrected prefix-rank stratified gate results exactly."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KS = (2, 4, 8, 12)


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows, stage):
    valid = [r for r in rows if r.get("ok") and r.get("checksum_ok")]
    failures = [r for r in rows if r not in valid]
    out = {"stage": stage, "rows_returned": len(rows), "valid_checksums": len(valid), "failures": [r.get("mlp_index") for r in failures], "ks": {}}
    for k in KS:
        reps = []
        by_mlp = defaultdict(list)
        for row in valid:
            for rep in row.get("rep_results", []):
                item = rep.get("k_results", {}).get(str(k))
                if item is not None:
                    reps.append(item)
                    by_mlp[int(row["mlp_index"])].append(item)
        def arr(name): return np.array([float(x[name]) for x in reps], dtype=float)
        cand, cur, ctrl, raw = (arr(n) for n in ("candidate_mse", "current_mse", "control_mse", "raw_rank_mse"))
        ratios = np.array([np.mean([x["current_mse"] for x in v]) / max(np.mean([x["candidate_mse"] for x in v]), 1e-300) for v in by_mlp.values()])
        out["ks"][str(k)] = {
            "rep_rows": len(reps),
            "candidate_mse_mean": float(np.mean(cand)) if len(cand) else None,
            "current_mse_mean": float(np.mean(cur)) if len(cur) else None,
            "control_greg_mse_mean": float(np.mean(ctrl)) if len(ctrl) else None,
            "raw_rank_mse_mean": float(np.mean(raw)) if len(raw) else None,
            "full_pool_ceiling_mse_mean": float(np.mean([x["full_pool_ceiling_mse"] for x in reps])) if reps else None,
            "global_ratio": float(np.mean(cur) / max(np.mean(cand), 1e-300)) if len(cand) else None,
            "median_ratio": float(np.median(ratios)) if len(ratios) else None,
            "q10_ratio": float(np.quantile(ratios, .1)) if len(ratios) else None,
            "min_ratio": float(np.min(ratios)) if len(ratios) else None,
            "corr_median": float(np.median([x["main_predictor_corr"] for x in reps])) if reps else None,
            "corr_q10": float(np.quantile([x["main_predictor_corr"] for x in reps], .1)) if reps else None,
            "projected_raw_flops": float(np.mean([x["projected_raw_flops"] for x in reps])) if reps else None,
            "projected_dense_ratio": float(np.mean([x["compute_dense_ratio"] for x in reps])) if reps else None,
            "ridge_lambda_mean": float(np.mean([x["ridge_lambda"] for x in reps])) if reps else None,
            "ridge_condition_mean": float(np.mean([x["ridge_condition"] for x in reps])) if reps else None,
            "selected_count": int(reps[0]["selected_count"]) if reps else None,
            "pilot_weight": float(reps[0]["pilot_weight"]) if reps else None,
        }
        g = out["ks"][str(k)]
        g["gates"] = {
            "100_checksums": len(valid) == 100 and not failures,
            "candidate_mse": g["candidate_mse_mean"] is not None and g["candidate_mse_mean"] <= (1.6e-6 if stage == "b" else 1.8e-6),
            "global_ratio": g["global_ratio"] is not None and g["global_ratio"] >= (1.35 if stage == "b" else 1.25),
            "median_ratio": g["median_ratio"] is not None and g["median_ratio"] >= (1.20 if stage == "b" else 1.10),
            "q10_ratio": g["q10_ratio"] is not None and g["q10_ratio"] >= (.90 if stage == "b" else .85),
            "min_ratio": g["min_ratio"] is not None and g["min_ratio"] >= (.70 if stage == "b" else .65),
            "beats_random_greg": g["candidate_mse_mean"] is not None and g["candidate_mse_mean"] < g["control_greg_mse_mean"],
            "beats_raw_rank": g["candidate_mse_mean"] is not None and g["candidate_mse_mean"] < g["raw_rank_mse_mean"],
            "compute_anchor": g["projected_raw_flops"] is not None and g["projected_raw_flops"] <= 2.535e10,
            "predictor_corr": g["corr_median"] is not None and g["corr_median"] >= .90,
        }
        g["pass"] = bool(all(g["gates"].values()))
    out["pass_k"] = [int(k) for k, g in out["ks"].items() if g["pass"]]
    out["pass"] = bool(out["pass_k"])
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("input", type=Path); p.add_argument("--stage", choices=("a", "b"), required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--report", type=Path, required=True); a=p.parse_args()
    s = summarize(load(a.input), a.stage)
    a.output.write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
    lines = [f"# Prefix-rank stratified Stage {a.stage.upper()} aggregate", "", f"Rows={s['rows_returned']} valid_checksums={s['valid_checksums']} failures={s['failures']}", ""]
    for k, g in s["ks"].items():
        lines += [f"## K={k}", f"candidate/current/control-GREG/raw-rank MSE = {g['candidate_mse_mean']} / {g['current_mse_mean']} / {g['control_greg_mse_mean']} / {g['raw_rank_mse_mean']}", f"global/median/q10/min ratio = {g['global_ratio']} / {g['median_ratio']} / {g['q10_ratio']} / {g['min_ratio']}", f"corr median/q10 = {g['corr_median']} / {g['corr_q10']}; projected FLOPs/ratio = {g['projected_raw_flops']} / {g['projected_dense_ratio']}; full-pool ceiling MSE={g['full_pool_ceiling_mse_mean']}", "Gates: " + ", ".join(f"{n}={'PASS' if v else 'FAIL'}" for n,v in g["gates"].items()), f"K verdict: {'PASS' if g['pass'] else 'FAIL'}", ""]
    lines += [f"Passing K: {s['pass_k']}", f"Overall verdict: {'PASS' if s['pass'] else 'FAIL'}"]
    a.report.write_text("\n".join(lines) + "\n")
    print(json.dumps(s, indent=2, sort_keys=True))


if __name__ == "__main__": main()
