#!/usr/bin/env python3
"""Aggregate tail-aware projection proxy gate JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.truth_bank import load_bank  # noqa: E402

OUT_DIR = ROOT / "paired_fly_logs" / "fingerprint_theory"
DEFAULT_JSONL = OUT_DIR / "tail_projection_proxy_gate_20260707_fly.jsonl"
RESULTS_PATH = OUT_DIR / "tail_projection_proxy_gate_20260707_results.json"
REPORT_PATH = OUT_DIR / "tail_projection_proxy_gate_20260707.md"


def qstats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "mean": float(np.mean(values)),
    }


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    records = []
    failures = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            payloads.append(payload)
            records.extend(row for row in payload.get("records", []) if isinstance(row, dict))
            failures.extend(row for row in payload.get("failures", []) if isinstance(row, dict))
    records.sort(key=lambda row: int(row["bank_index"]))
    return payloads, records, failures


def floor_by_index(metadata: dict[str, Any]) -> dict[int, float]:
    out = {}
    for row in metadata["rows"]:
        idx = int(row["mlp_index"])
        # Truth-bank labels are antithetic sample means. Use the documented
        # final-layer floor scale; gates compare reductions, but corrected MSE
        # levels are still reported.
        out[idx] = float(row.get("final_layer_truth_mse_floor", row.get("truth_mse_floor", 2.6e-8)))
    return out


def summarize(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    floors = floor_by_index(metadata)
    rows = []
    for record in records:
        floor = floors.get(int(record["bank_index"]), 2.6e-8)
        for row in record["rows"]:
            item = dict(row)
            item["bank_index"] = int(record["bank_index"])
            item["floor"] = floor
            for key in ("base_final_mse", "local_top_mse", "tail_top_mse", "successor_top_mse"):
                item[f"{key}_floor_subtracted"] = max(float(item[key]) - floor, 0.0)
            item["local_reduction_floor_subtracted"] = item["base_final_mse_floor_subtracted"] - item["local_top_mse_floor_subtracted"]
            item["tail_reduction_floor_subtracted"] = item["base_final_mse_floor_subtracted"] - item["tail_top_mse_floor_subtracted"]
            item["successor_reduction_floor_subtracted"] = item["base_final_mse_floor_subtracted"] - item["successor_top_mse_floor_subtracted"]
            rows.append(item)

    tail = np.asarray([r["tail_reduction_floor_subtracted"] for r in rows], dtype=np.float64)
    local = np.asarray([r["local_reduction_floor_subtracted"] for r in rows], dtype=np.float64)
    succ = np.asarray([r["successor_reduction_floor_subtracted"] for r in rows], dtype=np.float64)
    local_corr = np.asarray([r["spearman_local"] for r in rows], dtype=np.float64)
    tail_corr = np.asarray([r["spearman_tail"] for r in rows], dtype=np.float64)
    succ_corr = np.asarray([r["spearman_successor"] for r in rows], dtype=np.float64)
    eps = 1.0e-30
    summary = {
        "n_mlps": len(records),
        "n_rows": len(rows),
        "bank_indices": [int(r["bank_index"]) for r in records],
        "checksum_ok_count": sum(1 for r in records if r.get("checksum_ok") is True),
        "base_final_mse_floor_subtracted": qstats(np.asarray([r["base_final_mse"] - floors.get(int(r["bank_index"]), 2.6e-8) for r in records])),
        "tail_reduction": qstats(tail),
        "local_reduction": qstats(local),
        "successor_reduction": qstats(succ),
        "tail_over_local_reduction": qstats(tail / np.maximum(local, eps)),
        "tail_over_successor_reduction": qstats(tail / np.maximum(succ, eps)),
        "tail_minus_local_reduction": qstats(tail - local),
        "tail_minus_successor_reduction": qstats(tail - succ),
        "tail_win_local_fraction": float(np.mean(tail > local)),
        "tail_win_successor_fraction": float(np.mean(tail > succ)),
        "spearman_local": qstats(local_corr[np.isfinite(local_corr)]),
        "spearman_tail": qstats(tail_corr[np.isfinite(tail_corr)]),
        "spearman_successor": qstats(succ_corr[np.isfinite(succ_corr)]),
        "spearman_tail_minus_local": qstats((tail_corr - local_corr)[np.isfinite(tail_corr - local_corr)]),
        "spearman_tail_minus_successor": qstats((tail_corr - succ_corr)[np.isfinite(tail_corr - succ_corr)]),
        "by_layer": {},
    }
    for layer in sorted({int(r["layer"]) for r in rows}):
        lr = [r for r in rows if int(r["layer"]) == layer]
        lt = np.asarray([r["tail_reduction_floor_subtracted"] for r in lr], dtype=np.float64)
        ll = np.asarray([r["local_reduction_floor_subtracted"] for r in lr], dtype=np.float64)
        ls = np.asarray([r["successor_reduction_floor_subtracted"] for r in lr], dtype=np.float64)
        summary["by_layer"][str(layer)] = {
            "n": len(lr),
            "tail_over_local_reduction": qstats(lt / np.maximum(ll, eps)),
            "tail_over_successor_reduction": qstats(lt / np.maximum(ls, eps)),
            "tail_win_local_fraction": float(np.mean(lt > ll)),
            "tail_win_successor_fraction": float(np.mean(lt > ls)),
        }
    return summary


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    p1_pass = bool(summary["tail_over_local_reduction"]["median"] >= 1.10 and summary["tail_win_local_fraction"] >= 0.60)
    p2_pass = bool(summary["spearman_tail"]["median"] >= 0.15 and summary["spearman_tail_minus_local"]["median"] >= 0.05)
    p3_pass = bool(summary["tail_over_successor_reduction"]["median"] >= 1.05 and summary["tail_win_successor_fraction"] >= 0.55)
    if p1_pass and (p2_pass or p3_pass):
        verdict = "ALIVE"
    elif (not p1_pass) and (not p2_pass) and (not p3_pass):
        verdict = "DEAD"
    else:
        verdict = "INCONCLUSIVE"
    return {"p1_pass": p1_pass, "p2_pass": p2_pass, "p3_pass": p3_pass, "verdict": verdict}


def write_report(results: dict[str, Any]) -> None:
    s = results["summary"]
    d = results["decision"]
    missing = sorted(set(range(100)) - set(s["bank_indices"]))
    lines = [
        "# Tail-Aware Projection Proxy Gate (truth-bank edition)",
        "",
        "Fly-bank measurement gate. Each Machine rebuilt its bank MLP from seed, verified the weight checksum, sampled antithetic particles, computed a suffix Hutchinson diagonal `L^2` kernel from concrete remaining weights/ReLU masks, and measured final-layer MSE changes from coordinate mean corrections. No estimator behavior was changed or scored.",
        "",
        f"- Returned MLPs: {s['n_mlps']}",
        f"- Missing bank indices: {', '.join(str(i) for i in missing) if missing else 'none'}",
        f"- Layer rows: {s['n_rows']}",
        f"- Checksum rebuild: {'PASS' if s['checksum_ok_count'] == s['n_mlps'] else 'FAIL'} ({s['checksum_ok_count']}/{s['n_mlps']})",
        "",
        "## Premise verdicts",
        "",
        f"- P1 {'PASS' if d['p1_pass'] else 'FAIL'}: tail/local top-32 reduction ratio median {s['tail_over_local_reduction']['median']:.6g} [{s['tail_over_local_reduction']['q10']:.6g},{s['tail_over_local_reduction']['q90']:.6g}], win fraction {s['tail_win_local_fraction']:.3f}; gate requires median >= 1.10 and wins >= 0.60.",
        f"- P2 {'PASS' if d['p2_pass'] else 'FAIL'}: Spearman tail median {s['spearman_tail']['median']:.6g} [{s['spearman_tail']['q10']:.6g},{s['spearman_tail']['q90']:.6g}], tail-local median delta {s['spearman_tail_minus_local']['median']:.6g}; gate requires tail >= 0.15 and delta >= 0.05.",
        f"- P3 {'PASS' if d['p3_pass'] else 'FAIL'}: tail/successor top-32 reduction ratio median {s['tail_over_successor_reduction']['median']:.6g} [{s['tail_over_successor_reduction']['q10']:.6g},{s['tail_over_successor_reduction']['q90']:.6g}], win fraction {s['tail_win_successor_fraction']:.3f}; gate requires median >= 1.05 and wins >= 0.55.",
        "",
        "## By-layer tail/local ratio",
        "",
        "| layer | median | q10 | q90 | win frac |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer, row in s["by_layer"].items():
        q = row["tail_over_local_reduction"]
        lines.append(f"| {layer} | {q['median']:.6g} | {q['q10']:.6g} | {q['q90']:.6g} | {row['tail_win_local_fraction']:.3f} |")
    lines += [
        "",
        f"## Decision: {d['verdict']}",
        "",
        "All MSE levels and reductions are aggregated after subtracting the truth-bank floor. Small absolute changes near the floor are interpreted through paired ratios/win fractions rather than standalone levels.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = parser.parse_args()
    _, _, metadata = load_bank()
    payloads, records, failures = load_jsonl(args.jsonl)
    if failures:
        raise SystemExit(f"gate failures present: {failures[:3]}")
    if not records:
        raise SystemExit("no records found")
    summary = summarize(records, metadata)
    results = {
        "source_jsonl": str(args.jsonl),
        "config": payloads[0].get("config", {}) if payloads else {},
        "summary": summary,
        "decision": decide(summary),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(results)
    print("FINAL SUMMARY")
    print(f"records={summary['n_mlps']} rows={summary['n_rows']} checksum={summary['checksum_ok_count']}/{summary['n_mlps']}")
    print(f"P1 pass={results['decision']['p1_pass']} ratio={summary['tail_over_local_reduction']['median']:.6g} win={summary['tail_win_local_fraction']:.3f}")
    print(f"P2 pass={results['decision']['p2_pass']} tail_spearman={summary['spearman_tail']['median']:.6g} delta={summary['spearman_tail_minus_local']['median']:.6g}")
    print(f"P3 pass={results['decision']['p3_pass']} ratio={summary['tail_over_successor_reduction']['median']:.6g} win={summary['tail_win_successor_fraction']:.3f}")
    print(f"DECISION {results['decision']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
