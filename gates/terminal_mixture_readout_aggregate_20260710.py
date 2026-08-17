#!/usr/bin/env python3
"""Aggregate the frozen terminal mixture readout truth-bank payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCRIPT_VERSION = "terminal-mixture-readout-v1"


def _read(path: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("ok") is True and obj.get("script_version") == SCRIPT_VERSION:
            rows.append(obj)
        else:
            failures.append(obj)
    return rows, failures


def _summary(rows: list[dict], method: str) -> dict[str, float]:
    mean_mse = np.asarray([row[f"{method}_mean_mse"] for row in rows], dtype=np.float64)
    three_mse = np.asarray([row[f"{method}_three_rep_mean_mse"] for row in rows], dtype=np.float64)
    pair_var = np.asarray([row[f"{method}_pair_variance"] for row in rows], dtype=np.float64)
    return {
        "mean_replicate_mse": float(np.mean(mean_mse)),
        "median_replicate_mse": float(np.median(mean_mse)),
        "mean_three_rep_mean_mse": float(np.mean(three_mse)),
        "mean_pair_variance": float(np.mean(pair_var)),
    }


def aggregate(rows: list[dict], failures: list[dict]) -> dict[str, object]:
    baseline = np.asarray([row["baseline_mean_mse"] for row in rows], dtype=np.float64)
    mixture = np.asarray([row["mixture_mean_mse"] for row in rows], dtype=np.float64)
    gaussian = np.asarray([row["gaussian_mean_mse"] for row in rows], dtype=np.float64)
    mixture_ratio = baseline / np.maximum(mixture, 1e-300)
    gaussian_ratio = baseline / np.maximum(gaussian, 1e-300)
    mixture_bias = np.asarray([row["mixture_squared_bias_proxy"] for row in rows], dtype=np.float64)
    passed = bool(
        len(rows) == 100
        and np.mean(baseline) / max(np.mean(mixture), 1e-300) >= 1.35
        and np.median(mixture_ratio) >= 1.20
        and np.quantile(mixture_ratio, 0.10) >= 0.90
        and np.min(mixture_ratio) >= 0.70
        and np.mean(mixture_bias) <= 1.0e-6
    )
    return {
        "script_version": SCRIPT_VERSION,
        "n_payloads": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "pass_thresholds": {
            "payloads": 100,
            "mean_ratio_min": 1.35,
            "median_ratio_min": 1.20,
            "q10_ratio_min": 0.90,
            "minimum_ratio_min": 0.70,
            "mean_squared_bias_proxy_max": 1.0e-6,
        },
        "mixture": {
            "mean_baseline_mse": float(np.mean(baseline)) if len(rows) else None,
            "mean_candidate_mse": float(np.mean(mixture)) if len(rows) else None,
            "mean_ratio": float(np.mean(baseline) / max(np.mean(mixture), 1e-300)) if len(rows) else None,
            "median_ratio": float(np.median(mixture_ratio)) if len(rows) else None,
            "q10_ratio": float(np.quantile(mixture_ratio, 0.10)) if len(rows) else None,
            "q90_ratio": float(np.quantile(mixture_ratio, 0.90)) if len(rows) else None,
            "minimum_ratio": float(np.min(mixture_ratio)) if len(rows) else None,
            "tail_count_ratio_below_0.70": int(np.sum(mixture_ratio < 0.70)),
            "mean_squared_bias_proxy": float(np.mean(mixture_bias)) if len(rows) else None,
            "mean_pair_variance": float(np.mean([row["mixture_pair_variance"] for row in rows])) if len(rows) else None,
            "mean_three_rep_mean_mse": float(np.mean([row["mixture_three_rep_mean_mse"] for row in rows])) if len(rows) else None,
        },
        "gaussian_control": {
            "mean_baseline_mse": float(np.mean(baseline)) if len(rows) else None,
            "mean_control_mse": float(np.mean(gaussian)) if len(rows) else None,
            "mean_ratio": float(np.mean(baseline) / max(np.mean(gaussian), 1e-300)) if len(rows) else None,
            "median_ratio": float(np.median(gaussian_ratio)) if len(rows) else None,
            "q10_ratio": float(np.quantile(gaussian_ratio, 0.10)) if len(rows) else None,
            "q90_ratio": float(np.quantile(gaussian_ratio, 0.90)) if len(rows) else None,
        },
        "baseline_replicate_summary": _summary(rows, "baseline"),
        "mixture_replicate_summary": _summary(rows, "mixture"),
        "gaussian_replicate_summary": _summary(rows, "gaussian"),
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, failures = _read(args.jsonl)
    result = aggregate(rows, failures)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
