#!/usr/bin/env python3
"""Run paired WhestBench residual-time comparisons for two estimators."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCI_RE = r"([0-9]+(?:\.[0-9]+)?)e([+-]?[0-9]+)"
FLOAT_RE = r"([0-9]+(?:\.[0-9]+)?)"


def _sci_value(match: tuple[str, str]) -> float:
    return float(match[0]) * (10.0 ** int(match[1]))


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        return _sci_value(value)
    return float(value)


def _parse_report(text: str) -> dict[str, float | None]:
    return {
        "residual": _last_float(
            rf"Residual Wall Time \[residual_wall_time_s\]\s+{FLOAT_RE}s", text
        ),
        "effective": _last_float(
            rf"Effective Compute \[effective_compute\]\s+{SCI_RE}", text
        ),
        "flops": _last_float(rf"Total FLOPs \[flops_used\]\s+{SCI_RE}", text),
        "backend": _last_float(
            rf"Flopscope Backend \[flopscope_backend_time_s\]\s+{FLOAT_RE}s", text
        ),
        "overhead": _last_float(
            rf"Flopscope Overhead \[flopscope_overhead_time_s\]\s+{FLOAT_RE}s", text
        ),
        "score": _last_float(rf"Adjusted Final-Layer Score\s+{SCI_RE}", text),
        "raw_mse": _last_float(rf"Raw Final-Layer MSE \[final_layer_mse\]\s+{SCI_RE}", text),
        "multiplier": _last_float(rf"Mean Score Multiplier\s+{FLOAT_RE}", text),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "stdev": _stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sign_test_two_sided(wins: int, n: int) -> float:
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def _build_whest_command(args: argparse.Namespace, estimator: str) -> list[str]:
    return [
        sys.executable,
        "scripts/whest_with_residual_multiplier.py",
        "--residual-wall-time-multiplier",
        str(args.residual_wall_time_multiplier),
        "--",
        "run",
        "--estimator",
        estimator,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--runner",
        args.runner,
        "--fail-fast",
        "--n-mlps",
        str(args.n_mlps),
        "--flop-budget",
        str(args.flop_budget),
        "--wall-time-limit",
        str(args.wall_time_limit),
    ]


def _run_one(args: argparse.Namespace, label: str, estimator: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = args.uv_cache_dir
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env values must be KEY=VALUE, got {item!r}")
        env[key] = value

    started = time.monotonic()
    proc = subprocess.run(
        _build_whest_command(args, estimator),
        cwd=args.cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall = time.monotonic() - started
    if proc.returncode != 0:
        print(proc.stdout[-8000:], file=sys.stderr)
        raise SystemExit(f"{label} failed with return code {proc.returncode}")

    row: dict[str, Any] = {
        "label": label,
        "estimator": estimator,
        "wall": wall,
    }
    row.update(_parse_report(proc.stdout))
    missing = [key for key in ("residual", "effective", "flops") if row[key] is None]
    if missing:
        raise SystemExit(f"{label} report was missing fields: {', '.join(missing)}")
    return row


def _write_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _print_metric_summary(name: str, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    print(f"\n{name}")
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        summary = _summary(values)
        print(
            f"  {metric:10s} mean={summary['mean']:.9g} "
            f"stdev={summary['stdev']:.9g} min={summary['min']:.9g} max={summary['max']:.9g}"
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="First estimator path.")
    parser.add_argument("--new", required=True, help="Second estimator path.")
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="new")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--n-mlps", type=int, default=1)
    parser.add_argument("--dataset", default="hf://aicrowd/arc-whestbench-public-2026")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--runner", default="subprocess")
    parser.add_argument("--flop-budget", type=int, default=68_000_000_000)
    parser.add_argument("--wall-time-limit", type=int, default=240)
    parser.add_argument("--residual-wall-time-multiplier", type=float, default=2.0)
    parser.add_argument("--uv-cache-dir", default="/i/e/.uv-cache")
    parser.add_argument("--cwd", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment assignment for estimator runs, e.g. --env WHEST_K3_MODE=r1.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Optional path to append per-run JSON rows.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    args = parser.parse_args(argv)
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if args.n_mlps <= 0:
        raise SystemExit("--n-mlps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    deltas: dict[str, list[float]] = {
        "residual": [],
        "effective": [],
        "flops": [],
        "score": [],
        "backend": [],
        "overhead": [],
        "wall": [],
    }

    started = time.monotonic()
    for pair in range(1, args.pairs + 1):
        order = [
            (args.old_label, args.old, old_rows),
            (args.new_label, args.new, new_rows),
        ]
        if pair % 2 == 0:
            order.reverse()

        got: dict[str, dict[str, Any]] = {}
        for label, estimator, rows in order:
            row = _run_one(args, label, estimator)
            row["pair"] = pair
            rows.append(row)
            got[label] = row
            _write_jsonl(args.jsonl, row)

        for metric in deltas:
            old_value = got[args.old_label].get(metric)
            new_value = got[args.new_label].get(metric)
            if old_value is not None and new_value is not None:
                deltas[metric].append(float(new_value) - float(old_value))

        if not args.quiet:
            residual_delta = deltas["residual"][-1]
            wins = sum(delta < 0.0 for delta in deltas["residual"])
            print(
                f"pair {pair:03d}: "
                f"{args.old_label}={got[args.old_label]['residual']:.6f}s "
                f"{args.new_label}={got[args.new_label]['residual']:.6f}s "
                f"delta={residual_delta:+.6f}s "
                f"{args.new_label}_wins={wins}/{pair}",
                flush=True,
            )

    metrics = ["residual", "effective", "flops", "score", "raw_mse", "multiplier", "backend", "overhead", "wall"]
    _print_metric_summary(args.old_label, old_rows, metrics)
    _print_metric_summary(args.new_label, new_rows, metrics)

    print(f"\npaired deltas ({args.new_label} - {args.old_label})")
    for metric, values in deltas.items():
        if not values:
            continue
        summary = _summary(values)
        se = summary["stdev"] / math.sqrt(len(values))
        z = summary["mean"] / se if se > 0.0 else float("inf")
        normal_p = 2.0 * (1.0 - _normal_cdf(abs(z))) if math.isfinite(z) else 0.0
        wins = sum(delta < 0.0 for delta in values)
        sign_p = _sign_test_two_sided(wins, len(values))
        print(
            f"  {metric:10s} mean={summary['mean']:+.9g} "
            f"stdev={summary['stdev']:.9g} se={se:.9g} "
            f"wins={wins}/{len(values)} sign_p={sign_p:.4g} normal_p={normal_p:.4g}"
        )

    print(f"\nelapsed_s={time.monotonic() - started:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
