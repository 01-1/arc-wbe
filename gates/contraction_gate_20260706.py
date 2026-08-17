#!/usr/bin/env python3
"""Aggregate Fly JSONL for the the reference entrant contraction gate and write the memo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/i/e")
OUT_DIR = ROOT / "paired_fly_logs" / "fingerprint_theory"
DEFAULT_JSONL = OUT_DIR / "contraction_gate_20260706_fly_results.jsonl"
RESULTS_JSON = OUT_DIR / "contraction_gate_20260706_results.json"
MEMO_MD = OUT_DIR / "contraction_gate_20260706.md"
PROFILE_JSON = OUT_DIR / "profile_forensics_v2_20260706_results.json"
KEENAN_BAND = (-0.09394441757937719, -0.039370948894800195)
INJECTION_LAYERS = (2, 8, 16, 24)
PERTURBATION_TYPES = ("iid_gaussian", "bias_all", "top2_bias", "orthogonal_bias")


def qdict(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(arr, 0.1)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.9)),
        "mean": float(np.mean(arr)),
    }


def slope(xs: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(y) & (y > 0)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return float("nan")
    ly = np.log(y)
    xm = float(x.mean())
    ym = float(ly.mean())
    return float(np.sum((x - xm) * (ly - ym)) / np.sum((x - xm) ** 2))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_jsonls(paths: list[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in paths:
        payloads.extend(load_jsonl(path))
    return payloads


def flatten(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records_by_index: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    for payload in payloads:
        config.update(payload.get("config") or {})
        for record in payload.get("records") or []:
            records_by_index[int(record["bank_index"])] = record
        failures.extend(payload.get("failures") or [])
    records = [records_by_index[index] for index in sorted(records_by_index)]
    q1 = []
    q2 = []
    for record in records:
        q1.extend(record.get("q1") or [])
        q2.extend(record.get("q2") or [])
    return records, q1, q2, {"config": config, "failures": failures}


def summarize_q1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in INJECTION_LAYERS:
        out[str(k)] = {}
        for ptype in PERTURBATION_TYPES:
            sub = [r for r in rows if r["injection_layer"] == k and r["perturbation_type"] == ptype]
            out[str(k)][ptype] = {
                "n_mlps": len(sub),
                "mse_factor_per_layer": qdict([r["mse_factor_per_layer"] for r in sub]),
                "mse_log_slope": qdict([r["mse_log_slope"] for r in sub]),
                "amplitude_factor_per_layer": qdict([r["amplitude_factor_per_layer"] for r in sub]),
                "terminal_over_first_mse": qdict([r["terminal_over_first_mse"] for r in sub]),
            }
    return out


def summarize_q2(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    ref_layers = profile["reference_curves"]["layers"]
    plain = np.asarray([r["plain_var_median"] for r in ref_layers], dtype=np.float64)
    anti = np.asarray([r["antithetic_effective_var_median"] for r in ref_layers], dtype=np.float64)
    the reference entrant = np.asarray(
        [r["median"] for r in profile["leaderboard"]["entries"]["the reference entrant"]["per_layer_profile"]],
        dtype=np.float64,
    )
    hidden = np.arange(2, 31)
    plain_log = np.log(plain[hidden] / plain[2])
    anti_log = np.log(anti[hidden] / anti[2])
    keenan_log = np.log(the reference entrant[hidden] / the reference entrant[2])
    out: dict[str, Any] = {}
    for toy in sorted({r["toy"] for r in rows}):
        sub = [r for r in rows if r["toy"] == toy]
        mat = np.asarray([r["mse_by_layer"] for r in sub], dtype=np.float64)
        med = np.median(mat, axis=0)
        toy_log = np.log(med[hidden] / max(med[2], 1e-300))
        plain_resid = toy_log - plain_log
        out[toy] = {
            "n_mlps": len(sub),
            "per_mlp_slope": qdict([r["hidden_log_mse_slope_layers_2_30"] for r in sub]),
            "aggregate_median_slope": slope(hidden, med[2:31]),
            "aggregate_median_mse_factor_per_layer": float(math.exp(slope(hidden, med[2:31]))),
            "terminal_layer31_over_layer30": qdict([r["terminal_layer31_over_layer30"] for r in sub]),
            "final_mse": qdict([r["final_mse"] for r in sub]),
            "shape_correlations_log_layers_2_30": {
                "plain": pearson(toy_log, plain_log),
                "antithetic": pearson(toy_log, anti_log),
                "the reference entrant": pearson(toy_log, keenan_log),
            },
            "plain_residual_log_slope_layers_2_30": slope(hidden, np.exp(plain_resid)),
            "plain_residual_log_rms": float(np.sqrt(np.mean(plain_resid * plain_resid))),
            "median_mse_by_layer": [float(v) for v in med],
        }
    return out


def decide(q1: dict[str, Any], q2: dict[str, Any]) -> dict[str, Any]:
    relevant = []
    for k in ("8", "16", "24"):
        for ptype in ("bias_all", "top2_bias", "orthogonal_bias"):
            relevant.append(q1[k][ptype]["mse_factor_per_layer"]["median"])
    q1_med = float(np.median(relevant))
    q1_ok = 0.88 <= q1_med <= 0.97
    any_toy_ok = False
    any_toy_distinct = False
    toy_calls = {}
    for toy, row in q2.items():
        med_slope = row["per_mlp_slope"]["median"]
        in_band = KEENAN_BAND[0] <= med_slope <= KEENAN_BAND[1]
        distinct = (
            abs(row["plain_residual_log_slope_layers_2_30"]) > 0.015
            or row["plain_residual_log_rms"] > 0.20
            or row["shape_correlations_log_layers_2_30"]["plain"] < 0.96
        )
        any_toy_ok = any_toy_ok or in_band
        any_toy_distinct = any_toy_distinct or (in_band and distinct)
        toy_calls[toy] = {"slope_in_keenan_band": in_band, "measurably_distinct_from_plain": distinct}
    if q1_ok and any_toy_distinct:
        overall = "CONSISTENT"
    elif (not q1_ok) or (not any_toy_ok):
        overall = "INCONSISTENT"
    else:
        overall = "INCONCLUSIVE"
    return {
        "q1_median_relevant_mse_factor": q1_med,
        "q1_contracts_near_preregistered_scale": q1_ok,
        "toy_calls": toy_calls,
        "overall": overall,
    }


def write_memo(results: dict[str, Any]) -> None:
    q1 = results["q1_summary"]
    q2 = results["q2_summary"]
    dec = results["decision"]
    ck = results["checksum_verification"]
    lines = [
        "# the reference entrant Contraction Gate (truth-bank edition)",
        "",
        "Fly-bank research run. Each Machine rebuilt its bank MLP from seed, loaded only its truth-bank row, ran the perturbation and toy-state computations in place, and returned summary JSON. No estimator was run or scored.",
        "",
        "## Checksum rebuild",
        "",
        f"Checksum rows verified: {ck['ok_count']}/{ck['n_records']} matched. First row index `{ck['first_index']}` seed `{ck['first_seed']}` local SHA256 `{ck['first_local_sha256']}` vs bank `{ck['first_bank_sha256']}`.",
        "",
        "## Q1. Perturbation contraction",
        "",
        "| inject layer | type | MSE factor median [q10,q90] | amplitude factor median [q10,q90] | terminal/first MSE median |",
        "|---:|---|---:|---:|---:|",
    ]
    for k in INJECTION_LAYERS:
        for ptype in PERTURBATION_TYPES:
            row = q1[str(k)][ptype]
            mf = row["mse_factor_per_layer"]
            af = row["amplitude_factor_per_layer"]
            tf = row["terminal_over_first_mse"]
            lines.append(
                f"| {k} | {ptype} | {mf['median']:.3f} [{mf['q10']:.3f},{mf['q90']:.3f}] | "
                f"{af['median']:.3f} [{af['q10']:.3f},{af['q90']:.3f}] | {tf['median']:.3g} |"
            )
    lines.extend([
        "",
        "## Q2. Crude state-propagated toys",
        "",
        "| toy | per-MLP slope median [q10,q90] | aggregate slope | MSE factor/layer | L31/L30 median [q10,q90] | final MSE median |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for toy, row in q2.items():
        s = row["per_mlp_slope"]
        t = row["terminal_layer31_over_layer30"]
        fm = row["final_mse"]
        lines.append(
            f"| {toy} | {s['median']:.4f} [{s['q10']:.4f},{s['q90']:.4f}] | "
            f"{row['aggregate_median_slope']:.4f} | {row['aggregate_median_mse_factor_per_layer']:.3f} | "
            f"{t['median']:.3f} [{t['q10']:.3f},{t['q90']:.3f}] | {fm['median']:.3g} |"
        )
    lines.extend([
        "",
        "## Q3. Distinguishability from plain sampling",
        "",
        "| toy | corr plain | corr antithetic | corr the reference entrant | residual slope vs plain | residual log RMS | distinct call |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for toy, row in q2.items():
        c = row["shape_correlations_log_layers_2_30"]
        call = dec["toy_calls"][toy]
        lines.append(
            f"| {toy} | {c['plain']:.3f} | {c['antithetic']:.3f} | {c['the reference entrant']:.3f} | "
            f"{row['plain_residual_log_slope_layers_2_30']:.4f} | {row['plain_residual_log_rms']:.3f} | "
            f"{call['measurably_distinct_from_plain']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"Overall preregistered call: **{dec['overall']}**. Relevant Q1 median MSE factor was `{dec['q1_median_relevant_mse_factor']:.3f}`/layer; Q1 contract-scale call `{dec['q1_contracts_near_preregistered_scale']}`.",
        "",
        "Terminal-drop note: none of these crude propagated-state mechanisms directly supplies the reference entrant's additional ~25x final-layer discontinuity. That still requires an explicit final-layer allocation, refinement, or readout switch beyond smooth deep contraction.",
        "",
        "## Run metadata",
        "",
        f"- Fly JSONL: `{results['source_jsonl']}`",
        f"- Bank MLP records: {results['n_records']}",
        f"- Failures: {len(results['failures'])}",
        f"- Machine-side Q1 samples: {results['machine_config'].get('q1_samples')}",
        f"- Machine-side Q2 particles: {results['machine_config'].get('q2_samples')}",
    ])
    MEMO_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, nargs="+", default=[DEFAULT_JSONL])
    parser.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    args = parser.parse_args()
    payloads = load_jsonls(args.jsonl)
    records, q1_rows, q2_rows, extra = flatten(payloads)
    profile = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    q1_summary = summarize_q1(q1_rows)
    q2_summary = summarize_q2(q2_rows, profile)
    decision = decide(q1_summary, q2_summary)
    checksum_ok = [bool(r.get("checksum_ok")) for r in records]
    first = sorted(records, key=lambda r: r["bank_index"])[0] if records else {}
    results = {
        "created_at": "2026-07-06",
        "source_jsonl": [str(path) for path in args.jsonl],
        "n_payloads": len(payloads),
        "n_records": len(records),
        "failures": extra["failures"],
        "machine_config": extra["config"],
        "checksum_verification": {
            "n_records": len(records),
            "ok_count": int(sum(checksum_ok)),
            "all_ok": bool(all(checksum_ok)),
            "first_index": first.get("bank_index"),
            "first_seed": first.get("seed"),
            "first_local_sha256": first.get("weights_sha256"),
            "first_bank_sha256": first.get("bank_weights_sha256"),
        },
        "q1_rows": q1_rows,
        "q1_summary": q1_summary,
        "q2_rows": q2_rows,
        "q2_summary": q2_summary,
        "decision": decision,
    }
    args.results_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_memo(results)
    print("=== the reference entrant contraction gate summary ===")
    print(f"checksum ok: {results['checksum_verification']['ok_count']}/{len(records)}")
    for k in INJECTION_LAYERS:
        bits = []
        for ptype in PERTURBATION_TYPES:
            mf = q1_summary[str(k)][ptype]["mse_factor_per_layer"]
            bits.append(f"{ptype} {mf['median']:.3f} [{mf['q10']:.3f},{mf['q90']:.3f}]")
        print(f"Q1 layer {k}: " + "; ".join(bits))
    for toy, row in q2_summary.items():
        s = row["per_mlp_slope"]
        c = row["shape_correlations_log_layers_2_30"]
        print(
            f"Q2/Q3 {toy}: slope median {s['median']:.4f} [{s['q10']:.4f},{s['q90']:.4f}], "
            f"corr plain={c['plain']:.3f}, the reference entrant={c['the reference entrant']:.3f}, "
            f"distinct={decision['toy_calls'][toy]['measurably_distinct_from_plain']}"
        )
    print(f"overall: {decision['overall']}")
    print(f"wrote {args.results_json}")
    print(f"wrote {MEMO_MD}")


if __name__ == "__main__":
    main()
