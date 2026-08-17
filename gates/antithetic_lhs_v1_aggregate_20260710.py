#!/usr/bin/env python3
"""Aggregate antithetic Gaussian-LHS Stage A results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    valid=[r for r in rows if r.get("ok") and r.get("checksum_ok")]
    failures=[r.get("mlp_index") for r in rows if r not in valid]
    current=[]; lhs=[]; iid=[]; lhs_ratios=[]; iid_ratios=[]; iid_lhs_ratios=[]; lhs_diag=[]; iid_diag=[]
    for r in valid:
        for rep in r["reps"]:
            m=rep["mse"]; current.append(m["current"]); lhs.append(m["antithetic_gaussian_lhs"]); iid.append(m["iid_gaussian"])
            lhs_ratios.append(m["current"] / max(m["antithetic_gaussian_lhs"],1e-300)); iid_ratios.append(m["current"] / max(m["iid_gaussian"],1e-300)); iid_lhs_ratios.append(m["iid_gaussian"] / max(m["antithetic_gaussian_lhs"],1e-300)); lhs_diag.append(rep["lhs_diagnostics"]); iid_diag.append(rep["iid_diagnostics"])
    current=np.asarray(current); lhs=np.asarray(lhs); iid=np.asarray(iid); lhs_ratios=np.asarray(lhs_ratios); iid_ratios=np.asarray(iid_ratios); iid_lhs_ratios=np.asarray(iid_lhs_ratios)
    def d(name):
        v=np.asarray([x[name] for x in lhs_diag],dtype=float); return {"mean":float(np.mean(v)),"median":float(np.median(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def d_iid(name):
        v=np.asarray([x[name] for x in iid_diag],dtype=float); return {"mean":float(np.mean(v)),"median":float(np.median(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    s={"rows_returned":len(rows),"valid_checksums":len(valid),"failures":failures,"current_mse_mean":float(np.mean(current)),"lhs_mse_mean":float(np.mean(lhs)),"iid_mse_mean":float(np.mean(iid)),"current_lhs_global_ratio":float(np.mean(current)/np.mean(lhs)),"current_iid_global_ratio":float(np.mean(current)/np.mean(iid)),"iid_lhs_global_ratio":float(np.mean(iid)/np.mean(lhs)),"lhs_ratio_median":float(np.median(lhs_ratios)),"lhs_ratio_q10":float(np.quantile(lhs_ratios,.1)),"lhs_ratio_min":float(np.min(lhs_ratios)),"iid_ratio_median":float(np.median(iid_ratios)),"iid_ratio_q10":float(np.quantile(iid_ratios,.1)),"iid_ratio_min":float(np.min(iid_ratios)),"iid_lhs_ratio_median":float(np.median(iid_lhs_ratios)),"iid_lhs_ratio_q10":float(np.quantile(iid_lhs_ratios,.1)),"iid_lhs_ratio_min":float(np.min(iid_lhs_ratios)),"lhs_diag":{"exact_strata_all":all(x["exact_strata"] for x in lhs_diag),"antipode_max_abs":d("antipode_max_abs"),"max_full_coordinate_mean_abs":d("max_full_coordinate_mean_abs"),"radius_mean":d("radius_mean"),"radius_median":d("radius_median"),"radius_min":d("radius_min"),"radius_max":d("radius_max"),"diag_second_moment_min":d("diag_second_moment_min"),"diag_second_moment_median":d("diag_second_moment_median"),"diag_second_moment_max":d("diag_second_moment_max"),"cov_frobenius_relative_error":d("cov_frobenius_relative_error"),"cov_max_abs_offdiag":d("cov_max_abs_offdiag")},"iid_diag":{"antipode_max_abs":d_iid("antipode_max_abs"),"max_full_coordinate_mean_abs":d_iid("max_full_coordinate_mean_abs"),"radius_mean":d_iid("radius_mean"),"radius_median":d_iid("radius_median"),"radius_min":d_iid("radius_min"),"radius_max":d_iid("radius_max"),"diag_second_moment_min":d_iid("diag_second_moment_min"),"diag_second_moment_median":d_iid("diag_second_moment_median"),"diag_second_moment_max":d_iid("diag_second_moment_max"),"cov_frobenius_relative_error":d_iid("cov_frobenius_relative_error"),"cov_max_abs_offdiag":d_iid("cov_max_abs_offdiag")}}
    s["gates"]={"100_checksums":len(valid)==100 and not failures,"lhs_mse":s["lhs_mse_mean"]<=1.8e-6,"global_ratio":s["current_lhs_global_ratio"]>=1.25,"median_ratio":s["lhs_ratio_median"]>=1.10,"q10_ratio":s["lhs_ratio_q10"]>=.85,"min_ratio":s["lhs_ratio_min"]>=.65}; s["pass"]=all(s["gates"].values())
    a.output.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
    lines=["# Antithetic Gaussian-LHS v1 Stage A", "",f"Rows={s['rows_returned']} valid_checksums={s['valid_checksums']} failures={s['failures']}",f"Current/LHS/IID MSE={s['current_mse_mean']} / {s['lhs_mse_mean']} / {s['iid_mse_mean']}",f"Current/LHS global ratio={s['current_lhs_global_ratio']}; per-MLP median/q10/min={s['lhs_ratio_median']} / {s['lhs_ratio_q10']} / {s['lhs_ratio_min']}",f"Current/IID global ratio={s['current_iid_global_ratio']}; per-MLP median/q10/min={s['iid_ratio_median']} / {s['iid_ratio_q10']} / {s['iid_ratio_min']}",f"IID/LHS global ratio={s['iid_lhs_global_ratio']}; per-MLP median/q10/min={s['iid_lhs_ratio_median']} / {s['iid_lhs_ratio_q10']} / {s['iid_lhs_ratio_min']}",f"LHS diagnostics={s['lhs_diag']}",f"IID diagnostics={s['iid_diag']}","","Gates:"]+[f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in s['gates'].items()]+["",f"Verdict: {'PASS' if s['pass'] else 'FAIL'}"]
    a.report.write_text("\n".join(lines)+"\n"); print(json.dumps(s,indent=2,sort_keys=True))


if __name__ == "__main__": main()
