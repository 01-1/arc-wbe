#!/usr/bin/env python3
"""Aggregate ReLU region granularity truth-bank gate JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paired_fly_logs" / "fingerprint_theory"
DEFAULT_JSONL = OUT_DIR / "region_granularity_gate_20260707_fly.jsonl"
RESULTS_PATH = OUT_DIR / "region_granularity_gate_20260707_results.json"
REPORT_PATH = OUT_DIR / "region_granularity_gate_20260707.md"


def qstats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "mean": float(np.mean(arr)),
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[str, dict[str, Any]] = {}
    for layer in range(32):
        rows = [record["layers"][layer] for record in records]
        by_layer[str(layer)] = {
            "breakpoints_per_sigma": qstats([row["breakpoints_per_sigma"] for row in rows]),
            "live_hyperplanes": qstats([row["live_hyperplanes"] for row in rows]),
            "frozen_fraction": qstats([row["frozen_fraction"] for row in rows]),
            "within_region_variance_share": qstats([row["within_region_variance_share"] for row in rows]),
            "within_region_variance_share_total": qstats([row["within_region_variance_share_total"] for row in rows]),
            "flip_interval_fraction": qstats([row["flip_interval_fraction"] for row in rows]),
            "cooccurring_flip_interval_fraction": qstats([row["cooccurring_flip_interval_fraction"] for row in rows]),
        }
    return {
        "n_mlps": len(records),
        "bank_indices": [int(row["bank_index"]) for row in records],
        "checksum_ok_count": sum(1 for row in records if row.get("checksum_ok") is True),
        "breakpoints_per_sigma_total": qstats([row["breakpoints_per_sigma_total"] for row in records]),
        "typical_region_extent_sigma": qstats([row["typical_region_extent_sigma"] for row in records]),
        "effective_live_hyperplanes_total": qstats([row["effective_live_hyperplanes_total"] for row in records]),
        "layer31_live_hyperplanes": qstats([row["layer31_live_hyperplanes"] for row in records]),
        "layer31_within_region_variance_share": qstats([row["layers"][31]["within_region_variance_share"] for row in records]),
        "layer31_within_region_variance_share_total": qstats([row["layers"][31]["within_region_variance_share_total"] for row in records]),
        "max_layer_within_region_variance_share_median": max(by_layer[str(layer)]["within_region_variance_share"]["median"] for layer in range(32)),
        "deep_flip_cooccur_fraction": qstats([row["deep_flip_cooccur_fraction"] for row in records]),
        "deep_flip_events_per_flipping_interval_median": qstats([row["deep_flip_events_per_flipping_interval_median"] for row in records]),
        "by_layer": by_layer,
    }


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    p1_pass = summary["breakpoints_per_sigma_total"]["median"] <= 512.0
    p2_pass = (
        summary["layer31_within_region_variance_share"]["median"] >= 0.05
        or summary["max_layer_within_region_variance_share_median"] >= 0.10
    )
    p3_pass = (
        summary["effective_live_hyperplanes_total"]["median"] <= 2048.0
        or summary["layer31_live_hyperplanes"]["median"] <= 128.0
    )
    verdict = "ALIVE" if p1_pass and p2_pass and p3_pass else "DEAD"
    return {
        "p1_pass": bool(p1_pass),
        "p2_pass": bool(p2_pass),
        "p3_pass": bool(p3_pass),
        "verdict": verdict,
        "thresholds": {
            "p1_breakpoints_per_sigma_total_max": 512.0,
            "p2_layer31_within_share_min": 0.05,
            "p2_any_layer_within_share_min": 0.10,
            "p3_live_hyperplanes_total_max": 2048.0,
            "p3_layer31_live_hyperplanes_max": 128.0,
        },
    }


def write_report(results: dict[str, Any]) -> None:
    s = results["summary"]
    d = results["decision"]
    missing = sorted(set(range(100)) - set(s["bank_indices"]))
    layer31 = s["by_layer"]["31"]
    lines = [
        "# ReLU Region Granularity Gate (truth-bank edition)",
        "",
        "Pre-registered Fly-bank structural measurement. Each Machine rebuilt its truth-bank MLP from seed, verified the weight checksum, sampled Gaussian-bulk chords `x(t)=x0+t*u` over `[-1,1]`, counted sampled ReLU gate flips, and decomposed per-layer chord output variation into same-pattern affine interval variance versus between-interval variance. No estimator behavior was changed or scored.",
        "",
        f"- Returned MLPs: {s['n_mlps']}",
        f"- Missing bank indices: {', '.join(str(i) for i in missing) if missing else 'none'}",
        f"- Checksum rebuild: {'PASS' if s['checksum_ok_count'] == s['n_mlps'] else 'FAIL'} ({s['checksum_ok_count']}/{s['n_mlps']})",
        "",
        "## Premise verdicts",
        "",
        f"- P1 {'PASS' if d['p1_pass'] else 'FAIL'}: total breakpoint density median {s['breakpoints_per_sigma_total']['median']:.6g} [{s['breakpoints_per_sigma_total']['q10']:.6g},{s['breakpoints_per_sigma_total']['q90']:.6g}] per sigma; implied region extent median {s['typical_region_extent_sigma']['median']:.6g} sigma. Gate requires <= 512 per sigma.",
        f"- P2 {'PASS' if d['p2_pass'] else 'FAIL'}: layer-31 within-region variance share median {s['layer31_within_region_variance_share']['median']:.6g} [{s['layer31_within_region_variance_share']['q10']:.6g},{s['layer31_within_region_variance_share']['q90']:.6g}], max layer median {s['max_layer_within_region_variance_share_median']:.6g}. Gate requires layer 31 >= 0.05 or any layer >= 0.10.",
        f"- P3 {'PASS' if d['p3_pass'] else 'FAIL'}: effective live hyperplanes median {s['effective_live_hyperplanes_total']['median']:.6g} [{s['effective_live_hyperplanes_total']['q10']:.6g},{s['effective_live_hyperplanes_total']['q90']:.6g}] vs nominal 8192; layer-31 live median {s['layer31_live_hyperplanes']['median']:.6g}. Gate requires total <= 2048 or layer 31 <= 128.",
        "",
        "## Layer snapshot",
        "",
        "| layer | bp/sigma med | live med | frozen med | within-share med | flip-interval med | coflip-interval med |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in (0, 1, 3, 7, 15, 23, 30, 31):
        row = s["by_layer"][str(layer)]
        lines.append(
            f"| {layer} | {row['breakpoints_per_sigma']['median']:.6g} | "
            f"{row['live_hyperplanes']['median']:.6g} | {row['frozen_fraction']['median']:.6g} | "
            f"{row['within_region_variance_share']['median']:.6g} | "
            f"{row['flip_interval_fraction']['median']:.6g} | "
            f"{row['cooccurring_flip_interval_fraction']['median']:.6g} |"
        )
    lines += [
        "",
        f"Layer 31 within-region variance share using total chord variance denominator: {s['layer31_within_region_variance_share_total']['median']:.6g} [{s['layer31_within_region_variance_share_total']['q10']:.6g},{s['layer31_within_region_variance_share_total']['q90']:.6g}].",
        f"Deep-layer flip co-occurrence fraction: {s['deep_flip_cooccur_fraction']['median']:.6g} [{s['deep_flip_cooccur_fraction']['q10']:.6g},{s['deep_flip_cooccur_fraction']['q90']:.6g}].",
        "",
        f"## Decision: {d['verdict']}",
        "",
        "The within-region share uses only adjacent grid intervals whose sampled full layer gate pattern is unchanged, so it is conservative with respect to missed sub-grid crossings. The breakpoint density is a sampled line-chord density, not an exact arrangement count.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = parser.parse_args()
    payloads, records, failures = load_jsonl(args.jsonl)
    if failures:
        raise SystemExit(f"gate failures present: {failures[:3]}")
    if not records:
        raise SystemExit("no records found")
    summary = summarize(records)
    results = {
        "source_jsonl": str(args.jsonl),
        "config": payloads[0].get("config", {}) if payloads else {},
        "summary": summary,
        "decision": decide(summary),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(results)
    print("FINAL SUMMARY")
    print(f"records={summary['n_mlps']} checksum={summary['checksum_ok_count']}/{summary['n_mlps']}")
    print(f"P1 pass={results['decision']['p1_pass']} density={summary['breakpoints_per_sigma_total']['median']:.6g}")
    print(f"P2 pass={results['decision']['p2_pass']} layer31_share={summary['layer31_within_region_variance_share']['median']:.6g}")
    print(f"P3 pass={results['decision']['p3_pass']} live={summary['effective_live_hyperplanes_total']['median']:.6g}")
    print(f"DECISION {results['decision']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
