#!/usr/bin/env python3
"""Exact Stage-A/Stage-B aggregation for vector-GREG v3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KS = (4, 6, 8)


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def summarize(rows, stage):
    valid = [r for r in rows if r.get("ok") and r.get("checksum_ok")]
    failures = [r.get("mlp_index") for r in rows if r not in valid]
    out = {"stage": stage, "rows_returned": len(rows), "valid_checksums": len(valid), "failures": failures, "ks": {}}
    for k in KS:
        reps = []
        by_mlp = defaultdict(list)
        for row in valid:
            for rep in row.get("rep_results", []):
                item = rep.get("k_results", {}).get(str(k))
                if item is not None:
                    reps.append(item); by_mlp[int(row["mlp_index"])].append(item)
        def vals(name): return np.array([float(x[name]) for x in reps], dtype=float)
        cand, cur, ctrl, raw = (vals(n) for n in ("candidate_mse", "current_mse", "control_mse", "raw_rank_mse"))
        ratios = np.array([np.mean([x["current_mse"] for x in v]) / max(np.mean([x["candidate_mse"] for x in v]), 1e-300) for v in by_mlp.values()])
        r2 = np.array([float(x["full_main_r2"]) for x in reps])
        projected = np.array([float(x["projected_raw_flops"]) for x in reps])
        bias = [float(x["three_rep_bias_proxy"]) for x in reps if x.get("three_rep_bias_proxy") is not None]
        breakdown = {}
        if reps:
            for name in reps[0]["regression_flops"]:
                breakdown[name] = float(np.mean([x["regression_flops"][name] for x in reps]))
        g = {
            "rep_rows": len(reps),
            "candidate_mse_mean": float(np.mean(cand)) if len(cand) else None,
            "current_mse_mean": float(np.mean(cur)) if len(cur) else None,
            "random_vector_greg_mse_mean": float(np.mean(ctrl)) if len(ctrl) else None,
            "raw_rank_mse_mean": float(np.mean(raw)) if len(raw) else None,
            "full_pool_ceiling_mse_mean": float(np.mean([x["full_pool_ceiling_mse"] for x in reps])) if reps else None,
            "global_ratio": float(np.mean(cur) / max(np.mean(cand), 1e-300)) if len(cand) else None,
            "median_ratio": float(np.median(ratios)) if len(ratios) else None,
            "q10_ratio": float(np.quantile(ratios, .1)) if len(ratios) else None,
            "min_ratio": float(np.min(ratios)) if len(ratios) else None,
            "full_main_r2_median": float(np.median(r2)) if len(r2) else None,
            "full_main_r2_q10": float(np.quantile(r2, .1)) if len(r2) else None,
            "projected_raw_flops_mean": float(np.mean(projected)) if len(projected) else None,
            "projected_raw_flops_max": float(np.max(projected)) if len(projected) else None,
            "projected_dense_ratio": float(np.mean([x["compute_dense_ratio"] for x in reps])) if reps else None,
            "regression_flops_mean": breakdown,
            "three_rep_bias_proxy_mean": float(np.mean(bias)) if bias else None,
            "selected_count": int(reps[0]["selected_count"]) if reps else None,
        }
        g["gates"] = {
            "100_checksums": len(valid) == 100 and not failures,
            "candidate_mse": g["candidate_mse_mean"] is not None and g["candidate_mse_mean"] <= (1.6e-6 if stage == "b" else 1.8e-6),
            "global_ratio": g["global_ratio"] is not None and g["global_ratio"] >= (1.35 if stage == "b" else 1.25),
            "median_ratio": g["median_ratio"] is not None and g["median_ratio"] >= (1.20 if stage == "b" else 1.10),
            "q10_ratio": g["q10_ratio"] is not None and g["q10_ratio"] >= (.90 if stage == "b" else .85),
            "min_ratio": g["min_ratio"] is not None and g["min_ratio"] >= (.70 if stage == "b" else .65),
            "beats_random_vector_greg": g["candidate_mse_mean"] is not None and g["candidate_mse_mean"] < g["random_vector_greg_mse_mean"],
            "beats_raw_rank": g["candidate_mse_mean"] is not None and g["candidate_mse_mean"] < g["raw_rank_mse_mean"],
            "compute_anchor": g["projected_raw_flops_max"] is not None and g["projected_raw_flops_max"] <= 2.535e10,
            "full_main_r2_median": g["full_main_r2_median"] is not None and g["full_main_r2_median"] >= .90,
            "full_main_r2_q10": g["full_main_r2_q10"] is not None and g["full_main_r2_q10"] >= .75,
            "three_rep_bias_proxy": (stage != "b") or (g["three_rep_bias_proxy_mean"] is not None and g["three_rep_bias_proxy_mean"] <= 1e-6),
        }
        g["pass"] = bool(all(g["gates"].values()))
        out["ks"][str(k)] = g
    out["pass_k"] = [int(k) for k,g in out["ks"].items() if g["pass"]]
    out["pass"] = bool(out["pass_k"])
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("--stage",choices=("a","b"),required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    s=summarize(load(a.input),a.stage); a.output.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
    lines=[f"# Prefix vector-GREG v3 Stage {a.stage.upper()} aggregate","",f"Rows={s['rows_returned']} valid_checksums={s['valid_checksums']} failures={s['failures']}",""]
    for k,g in s["ks"].items():
        lines += [f"## K={k}",f"MSE candidate/current/random-GREG/raw-rank = {g['candidate_mse_mean']} / {g['current_mse_mean']} / {g['random_vector_greg_mse_mean']} / {g['raw_rank_mse_mean']}",f"ratio global/median/q10/min = {g['global_ratio']} / {g['median_ratio']} / {g['q10_ratio']} / {g['min_ratio']}",f"full-main R2 median/q10 = {g['full_main_r2_median']} / {g['full_main_r2_q10']}; projected raw FLOPs mean/max={g['projected_raw_flops_mean']} / {g['projected_raw_flops_max']}; dense ratio={g['projected_dense_ratio']}",f"full-pool ceiling MSE={g['full_pool_ceiling_mse_mean']}; regression breakdown={g['regression_flops_mean']}","Gates: "+", ".join(f"{n}={'PASS' if v else 'FAIL'}" for n,v in g['gates'].items()),f"K verdict: {'PASS' if g['pass'] else 'FAIL'}",""]
    lines += [f"Passing K: {s['pass_k']}",f"Overall verdict: {'PASS' if s['pass'] else 'FAIL'}"]
    a.report.write_text("\n".join(lines)+"\n"); print(json.dumps(s,indent=2,sort_keys=True))


if __name__ == "__main__": main()
