#!/usr/bin/env python3
"""Aggregate the active-subspace reflection covariance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def qstats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "q10": float(np.quantile(x, 0.10)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("ok") is True and obj.get("metrics"):
            rows.append(obj)
        elif isinstance(obj.get("result"), dict) and obj["result"].get("ok"):
            rows.append(obj["result"])
    rows = list({int(row["mlp_index"]): row for row in rows}.values())
    rows.sort(key=lambda row: int(row["mlp_index"]))
    complete = len(rows) == 100 and [int(row["mlp_index"]) for row in rows] == list(range(100))
    checksums = all(bool(row.get("checksum_ok")) for row in rows)
    groups: dict[tuple[str, int, str, int], list[dict[str, float]]] = {}
    for row in rows:
        for metric in row["metrics"]:
            key = (str(metric["family"]), int(metric["rank"]), str(metric["law"]), int(metric["reference_rep"]))
            groups.setdefault(key, []).append(metric)

    summaries = []
    for key, metrics in groups.items():
        family, rank, law, reference_rep = key
        gains = [float(m["gain"]) for m in metrics]
        rhos = [float(m["rho"]) for m in metrics]
        sanity = [float(m["mean_difference_stat"]) for m in metrics]
        summaries.append({
            "family": family,
            "rank": rank,
            "law": law,
            "reference_rep": reference_rep,
            "gain": qstats(gains),
            "rho": qstats(rhos),
            "sanity": qstats(sanity),
        })
    summaries.sort(key=lambda item: item["gain"]["mean"], reverse=True)

    by_plan: dict[tuple[str, int, str], list[dict[str, float]]] = {}
    for item in summaries:
        by_plan.setdefault((item["family"], item["rank"], item["law"]), []).append(item)
    replication = []
    for key, items in by_plan.items():
        if len(items) != 2:
            continue
        replication.append({
            "family": key[0], "rank": key[1], "law": key[2],
            "rep0_mean_gain": items[0]["gain"]["mean"],
            "rep1_mean_gain": items[1]["gain"]["mean"],
            "rep0_median_gain": items[0]["gain"]["median"],
            "rep1_median_gain": items[1]["gain"]["median"],
            "both_pass": all(
                item["gain"]["mean"] >= 1.7
                and item["gain"]["median"] >= 1.5
                and item["gain"]["q10"] >= 1.0
                for item in items
            ),
        })
    passing = [item for item in replication if item["both_pass"]]
    verdict = "PASS: active-subspace reflection survives both fresh-reference replications" if complete and checksums and passing else "FAIL: no replicated active-subspace reflection plan met the gain gate"
    result = {
        "complete": complete,
        "checksum_ok": checksums,
        "n_mlps": len(rows),
        "verdict": verdict,
        "summaries": summaries,
        "replication": replication,
        "passing": passing,
        "thresholds": {"pooled_gain": 1.7, "median_gain": 1.5, "q10_gain": 1.0},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Active-subspace reflection covariance gate",
        "",
        f"Successful shards: `{len(rows)}/100`; checksums: `{'PASS' if checksums else 'FAIL'}`.",
        "",
        f"**{verdict}**",
        "",
        "Thresholds: pooled gain >=1.7x, per-MLP median >=1.5x, q10 >=1.0x, and the same plan must pass in both fresh-reference replications.",
        "",
        "| family | rank | law | ref | gain mean | gain median | gain q10 | rho mean | sanity mean |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['family']} | {item['rank']} | {item['law']} | {item['reference_rep']} | "
            f"{item['gain']['mean']:.4f} | {item['gain']['median']:.4f} | {item['gain']['q10']:.4f} | "
            f"{item['rho']['mean']:.4f} | {item['sanity']['mean']:.4f} |"
        )
    lines.extend(["", "Replication decisions:", ""])
    for item in replication:
        lines.append(
            f"- `{item['family']}` rank {item['rank']} `{item['law']}`: "
            f"rep0 mean/median {item['rep0_mean_gain']:.4f}/{item['rep0_median_gain']:.4f}, "
            f"rep1 mean/median {item['rep1_mean_gain']:.4f}/{item['rep1_median_gain']:.4f}; "
            f"{'PASS' if item['both_pass'] else 'fail'}."
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_mlps": len(rows), "verdict": verdict, "passing": passing}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

