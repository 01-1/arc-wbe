#!/usr/bin/env python3
"""Aggregate readout-smoothing Fly-bank gate JSONL into JSON and memo."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.truth_bank import load_bank  # noqa: E402

OUT_DIR = ROOT / "paired_fly_logs" / "fingerprint_theory"
DEFAULT_JSONL = OUT_DIR / "readout_smoothing_gate_20260706_fly.jsonl"
RESULTS_PATH = OUT_DIR / "readout_smoothing_gate_20260706_results.json"
REPORT_PATH = OUT_DIR / "readout_smoothing_gate_20260706.md"
DEPTH = 32
EPS = 1.0e-12


def qstats(values: np.ndarray) -> dict[str, float]:
    return {
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "mean": float(np.mean(values)),
    }


def layer_qstats(matrix: np.ndarray) -> list[dict[str, float]]:
    return [qstats(matrix[:, layer]) for layer in range(matrix.shape[1])]


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    records = []
    failures = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            payloads.append(payload)
            records.extend(record for record in payload.get("records", []) if isinstance(record, dict))
            failures.extend(failure for failure in payload.get("failures", []) if isinstance(failure, dict))
    records.sort(key=lambda record: int(record["bank_index"]))
    return payloads, records, failures


def summarize(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    rows = {int(row["mlp_index"]): row for row in metadata["rows"]}
    summary: dict[str, Any] = {
        "n_mlps": len(records),
        "bank_indices": [int(record["bank_index"]) for record in records],
        "sample_counts": [1024, 4096, 8192],
    }

    for mode_name, anti in (("iid", False), ("antithetic", True)):
        summary[mode_name] = {}
        for n in summary["sample_counts"]:
            direct = []
            smooth = []
            ratios = []
            for record in records:
                match = next(
                    item
                    for item in record["replicates"]
                    if int(item["n"]) == n and bool(item["antithetic"]) is anti
                )
                d = np.asarray(match["direct_mse"], dtype=np.float64)
                s = np.asarray(match["smooth_mse"], dtype=np.float64)
                direct.append(d)
                smooth.append(s)
                ratios.append(s / np.maximum(d, EPS))
            summary[mode_name][str(n)] = {
                "completed_mlps": len(ratios),
                "direct_mse_by_layer": layer_qstats(np.asarray(direct)),
                "smooth_mse_by_layer": layer_qstats(np.asarray(smooth)),
                "ratio_by_layer": layer_qstats(np.asarray(ratios)),
            }

    skew = np.asarray([record["skew_abs_q"] for record in records], dtype=np.float64)
    kurt = np.asarray([record["excess_kurtosis_q"] for record in records], dtype=np.float64)
    smooth_bias2 = np.asarray([record["smooth_bias2"] for record in records], dtype=np.float64)
    anti_eff_var = np.asarray([record["anti_eff_var"] for record in records], dtype=np.float64)
    direct_large = np.asarray([record["direct_large_error2"] for record in records], dtype=np.float64)
    sample_count = np.asarray([rows[int(record["bank_index"])]["sample_count"] for record in records], dtype=np.float64)
    floor = anti_eff_var / sample_count[:, None]
    corrected = np.maximum(smooth_bias2 - floor, 0.0)
    summary["p1"] = {
        "completed_mlps": len(records),
        "skew_abs_layer_unit_q_across_mlp": [
            {
                "q10": float(np.quantile(skew[:, layer, 0], 0.10)),
                "median": float(np.quantile(skew[:, layer, 1], 0.50)),
                "q90": float(np.quantile(skew[:, layer, 2], 0.90)),
            }
            for layer in range(DEPTH)
        ],
        "excess_kurtosis_layer_unit_q_across_mlp": [
            {
                "q10": float(np.quantile(kurt[:, layer, 0], 0.10)),
                "median": float(np.quantile(kurt[:, layer, 1], 0.50)),
                "q90": float(np.quantile(kurt[:, layer, 2], 0.90)),
            }
            for layer in range(DEPTH)
        ],
    }
    summary["p3"] = {
        "completed_mlps": len(records),
        "smooth_bias2_by_layer_raw": layer_qstats(smooth_bias2),
        "truth_noise_floor_by_layer": layer_qstats(floor),
        "smooth_bias2_by_layer_floor_subtracted": layer_qstats(corrected),
        "direct_large_n_error2_by_layer": layer_qstats(direct_large),
        "floor_limited_mlp_layer_count": int(np.sum(smooth_bias2 <= floor)),
    }
    return summary


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    p2_l31 = summary["antithetic"]["8192"]["ratio_by_layer"][-1]
    p2_pass = bool(p2_l31["median"] <= 1 / 1.5 and p2_l31["q90"] < 0.9)
    saved = (
        summary["antithetic"]["8192"]["direct_mse_by_layer"][-1]["median"]
        - summary["antithetic"]["8192"]["smooth_mse_by_layer"][-1]["median"]
    )
    raw = summary["p3"]["smooth_bias2_by_layer_raw"][-1]["median"]
    floor = summary["p3"]["truth_noise_floor_by_layer"][-1]["median"]
    corrected = summary["p3"]["smooth_bias2_by_layer_floor_subtracted"][-1]["median"]
    floor_limited = raw <= floor
    bias_for_gate = raw if floor_limited else corrected
    p3_pass = bool(bias_for_gate < 0.2 * saved)
    if p2_pass and p3_pass:
        verdict = "ALIVE"
    elif p2_l31["median"] >= 0.9 or not p3_pass:
        verdict = "DEAD"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "p2_pass": p2_pass,
        "p3_pass": p3_pass,
        "p3_floor_limited": bool(floor_limited),
        "p3_saved_variance_l31_median": float(saved),
        "p3_bias2_l31_median_for_gate": float(bias_for_gate),
        "decision": verdict,
    }


def write_report(results: dict[str, Any]) -> None:
    summary = results["summary"]
    decision = results["decision"]
    checksum_ok = results["checksum_rebuild"]["ok"]
    missing = sorted(set(range(100)) - set(int(index) for index in summary["bank_indices"]))
    lines = [
        "# Readout-Smoothing Gate (truth-bank edition)",
        "",
        "Fly-bank-style machine-side gate. Each Machine rebuilt its bank MLP from the bank seed, verified the bank checksum, ran the readout comparison locally on that Machine, and returned only compact summary statistics. The local step only aggregates JSON and applies the truth-noise-floor subtraction from `metadata.json` sample counts.",
        "",
        f"- Returned MLPs: {summary['n_mlps']}",
        f"- Missing bank indices: {', '.join(str(index) for index in missing) if missing else 'none'}",
        f"- Candidate n: {', '.join(str(n) for n in summary['sample_counts'])}",
        f"- Replicates per n/mode: {results['config'].get('reps')}",
        f"- Large-n P1/P3 pass: {results['config'].get('bias_n')}",
        f"- Checksum rebuild: {'PASS' if checksum_ok else 'FAIL'} ({results['checksum_rebuild']['ok_count']}/{summary['n_mlps']})",
        "",
        "## Final-layer ratio table",
        "",
        "| mode | n | smoothed/direct median | q10 | q90 | completed MLPs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("iid", "antithetic"):
        for n in summary["sample_counts"]:
            row = summary[mode][str(n)]
            q = row["ratio_by_layer"][-1]
            lines.append(f"| {mode} | {n} | {q['median']:.6g} | {q['q10']:.6g} | {q['q90']:.6g} | {row['completed_mlps']} |")

    p1_skew = summary["p1"]["skew_abs_layer_unit_q_across_mlp"][-1]
    p1_kurt = summary["p1"]["excess_kurtosis_layer_unit_q_across_mlp"][-1]
    p2_l31 = summary["antithetic"]["8192"]["ratio_by_layer"][-1]
    p3_raw = summary["p3"]["smooth_bias2_by_layer_raw"][-1]
    p3_floor = summary["p3"]["truth_noise_floor_by_layer"][-1]
    p3_corr = summary["p3"]["smooth_bias2_by_layer_floor_subtracted"][-1]
    p1_pass = bool(p1_skew["q90"] < 0.5 and abs(p1_kurt["median"]) < 0.5 and abs(p1_kurt["q90"]) < 1.0)

    lines += [
        "",
        "## Premise verdicts",
        "",
        f"- P1 {'PASS' if p1_pass else 'FAIL'}: layer 31 abs-skew median/q90 {p1_skew['median']:.6g}/{p1_skew['q90']:.6g}; excess-kurtosis median [q10,q90] {p1_kurt['median']:.6g} [{p1_kurt['q10']:.6g},{p1_kurt['q90']:.6g}].",
        f"- P2 {'PASS' if decision['p2_pass'] else 'FAIL'}: antithetic layer 31 ratio at n=8192 median {p2_l31['median']:.6g}, q90 {p2_l31['q90']:.6g}; gate requires median <= 0.667 and q90 < 0.9.",
        f"- P3 {'PASS' if decision['p3_pass'] else 'FAIL'}: layer 31 smooth bias^2 raw median {p3_raw['median']:.6g}, floor median {p3_floor['median']:.6g}, floor-subtracted median {p3_corr['median']:.6g}; saved variance median {decision['p3_saved_variance_l31_median']:.6g}.",
        "",
        f"## Decision: {decision['decision']}",
        "",
        "Bias values at or below the estimated truth-noise floor are interpreted as upper bounds. No estimator.py path was imported or scored by this gate entrypoint.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = parser.parse_args()
    _, _, metadata = load_bank()
    payloads, records, failures = load_records(args.jsonl)
    if failures:
        raise SystemExit(f"gate failures present: {failures[:3]}")
    if not records:
        raise SystemExit("no records found")
    config = payloads[0].get("config", {})
    summary = summarize(records, metadata)
    checksum_ok_count = sum(1 for record in records if record.get("checksum_ok") is True)
    results = {
        "source_jsonl": str(args.jsonl),
        "config": config,
        "checksum_rebuild": {
            "ok": checksum_ok_count == len(records),
            "ok_count": checksum_ok_count,
            "n_records": len(records),
            "first": {
                "bank_index": int(records[0]["bank_index"]),
                "seed": int(records[0]["seed"]),
                "weights_sha256": records[0]["weights_sha256"],
            },
        },
        "summary": summary,
    }
    results["decision"] = decide(summary)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(results)

    print("FINAL SUMMARY")
    print(
        f"checksum rebuild: {'PASS' if results['checksum_rebuild']['ok'] else 'FAIL'} "
        f"{checksum_ok_count}/{len(records)}; first index={records[0]['bank_index']} seed={records[0]['seed']}"
    )
    for mode in ("iid", "antithetic"):
        print(f"{mode} final-layer ratios:")
        for n in summary["sample_counts"]:
            q = summary[mode][str(n)]["ratio_by_layer"][-1]
            print(f"  n={n}: median={q['median']:.6g} q10={q['q10']:.6g} q90={q['q90']:.6g}")
    p1 = summary["p1"]["skew_abs_layer_unit_q_across_mlp"][-1]
    p2 = results["decision"]["p2_pass"]
    p3 = results["decision"]["p3_pass"]
    print(f"P1 l31 abs-skew median={p1['median']:.6g} q90={p1['q90']:.6g}")
    print(f"P2 pass={p2}")
    print(
        f"P3 l31 bias2_for_gate={results['decision']['p3_bias2_l31_median_for_gate']:.6g} "
        f"saved={results['decision']['p3_saved_variance_l31_median']:.6g} pass={p3}"
    )
    print(f"DECISION {results['decision']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
