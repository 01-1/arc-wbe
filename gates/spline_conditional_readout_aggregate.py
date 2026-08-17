#!/usr/bin/env python3
"""Aggregate cross-fitted smooth conditional readout Fly payload rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("ok") is True and obj.get("script") == "spline_conditional_readout_payload_v1":
            rows.append(obj)
    return rows


def aggregate(rows: list[dict]) -> dict[str, object]:
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for rec in row["records"]:
            key = (rec["layer"], rec["rank"], rec["family"], rec["ridge"])
            by_cfg[key].append({**rec, "mlp_index": row["mlp_index"]})

    configs = []
    for key, recs in by_cfg.items():
        layer, rank, family, ridge = key
        per_mlp = []
        for mlp_index in sorted({r["mlp_index"] for r in recs}):
            vals = [r for r in recs if r["mlp_index"] == mlp_index]
            equal = float(np.mean([r["equal_mse"] for r in vals]))
            cond = float(np.mean([r["conditional_mse"] for r in vals]))
            per_mlp.append(equal / cond if cond > 0.0 else 0.0)
        ratios = np.asarray(per_mlp, dtype=np.float64)
        mean_equal = float(np.mean([r["equal_mse"] for r in recs]))
        mean_cond = float(np.mean([r["conditional_mse"] for r in recs]))
        configs.append(
            {
                "layer": layer,
                "rank": rank,
                "family": family,
                "ridge": ridge,
                "n_mlps": int(len(per_mlp)),
                "mean_equal_mse": mean_equal,
                "mean_conditional_mse": mean_cond,
                "mean_ratio": mean_equal / mean_cond if mean_cond > 0.0 else 0.0,
                "median_ratio": float(np.median(ratios)),
                "q10_ratio": float(np.quantile(ratios, 0.10)),
                "q90_ratio": float(np.quantile(ratios, 0.90)),
                "tail_blowup_count": int(np.sum(ratios < 0.80)),
                "mean_bias_norm": float(np.mean([r["bias_norm"] for r in recs])),
            }
        )
    configs.sort(key=lambda item: (item["mean_ratio"], item["median_ratio"], item["q10_ratio"]), reverse=True)
    best = configs[0] if configs else None
    passed = bool(
        best
        and best["mean_ratio"] >= 1.35
        and best["median_ratio"] >= 1.20
        and best["q10_ratio"] >= 0.90
        and best["tail_blowup_count"] == 0
    )
    return {
        "n_payloads": len(rows),
        "pass": passed,
        "pass_thresholds": {
            "mean_ratio_min": 1.35,
            "median_ratio_min": 1.20,
            "q10_ratio_min": 0.90,
            "tail_blowup_count_max": 0,
        },
        "best": best,
        "configs": configs[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(_read(args.jsonl))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
