#!/usr/bin/env python3
"""Extract per-MLP final-layer MSE from the four block-ladder Fly logs into a CSV.

Run once to regenerate `ladder_per_mlp_mse.csv` from the raw run logs. The logs
themselves are not published: they carry presigned object-store URLs and machine
identifiers that are irrelevant to the measurement. This script keeps only the
per-MLP final-layer MSE, which is the entire evidentiary basis for the fit.
"""
import csv
import json
import re
import sys
from pathlib import Path

RUNS = [
    ("b8", 8, 4096, "b8_full_json.log"),
    ("b16", 16, 8192, "default_a_full_json.log"),
    ("b32", 32, 16384, "b32_full_json.log"),
    ("b64", 64, 32768, "b64_full_json.log"),
]


def parse_log(path):
    """Map MLP index -> final-layer MSE for clean rows only."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(errors="ignore"))
    mlp_of_machine = {}
    blocks = re.split(r"===== Fly MLP (\d+) returncode=(-?\d+) =====", txt)
    for i in range(1, len(blocks) - 2, 3):
        idx, body = int(blocks[i]), blocks[i + 2]
        m = re.search(r"Machine ID: (\w+)", body)
        if m:
            mlp_of_machine[m.group(1)] = idx
    rows = {}
    for m in re.finditer(r"app\[(\w+)\].*?WHEST_RESULT_JSON (\{.*)", txt):
        machine = m.group(1)
        if machine not in mlp_of_machine:
            continue
        try:
            d, _ = json.JSONDecoder().raw_decode(m.group(2))
        except json.JSONDecodeError:
            continue
        failed = bool(d.get("n_failed_mlps", 0)) or any((d.get("failure_breakdown") or {}).values())
        if not failed:
            rows[mlp_of_machine[machine]] = d["final_layer_mse"]
    return rows


def main():
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "paired_fly_logs")
    data = {}
    for tag, blocks, samples, fname in RUNS:
        rows = parse_log(log_dir / fname)
        data[tag] = rows
        print(f"{tag:>4} ({blocks:>2} blocks, {samples:>5} samples): {len(rows)} clean rows", file=sys.stderr)
    out = Path(__file__).with_name("ladder_per_mlp_mse.csv")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mlp_index", "mse_b8", "mse_b16", "mse_b32", "mse_b64"])
        for i in range(100):
            w.writerow([i] + ["" if i not in data[t] else repr(data[t][i]) for t in ("b8", "b16", "b32", "b64")])
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
