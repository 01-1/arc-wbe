#!/usr/bin/env python3
"""Extract per-MLP final-layer MSE rows from full-JSON Fly logs and pair runs."""
import json
import re
import sys


def parse_log(path):
    txt = open(path).read()
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    # Map "===== Fly MLP NNNNNN" header -> following "Machine ID: xxxx"
    mlp_of_machine = {}
    blocks = re.split(r"===== Fly MLP (\d+) returncode=(-?\d+) =====", txt)
    # blocks: [pre, idx, rc, body, idx, rc, body, ...]
    for i in range(1, len(blocks) - 2, 3):
        idx = int(blocks[i])
        rc = int(blocks[i + 1])
        body = blocks[i + 2]
        m = re.search(r"Machine ID: (\w+)", body)
        if m:
            mlp_of_machine[m.group(1)] = (idx, rc)
    rows = {}
    for m in re.finditer(r"app\[(\w+)\].*?WHEST_RESULT_JSON (\{.*)", txt):
        machine = m.group(1)
        try:
            d, _ = json.JSONDecoder().raw_decode(m.group(2))
        except json.JSONDecodeError:
            continue
        if machine not in mlp_of_machine:
            continue
        idx, rc = mlp_of_machine[machine]
        fails = d.get("failure_breakdown") or {}
        n_failed = d.get("n_failed_mlps", 0)
        rows[idx] = {
            "mse": d["final_layer_mse"],
            "flops": d.get("mlp_flops_used"),
            "eff": d.get("mlp_effective_compute"),
            "failed": bool(n_failed) or any(fails.values()),
            "fail_kinds": fails,
        }
    return rows


def summarize(rows, name):
    clean = {k: v for k, v in rows.items() if not v["failed"]}
    mses = [v["mse"] for v in clean.values()]
    print(f"{name}: {len(rows)} rows, {len(clean)} clean, "
          f"mean final-layer MSE {sum(mses)/len(mses):.9e}")
    bad = {k: v["fail_kinds"] for k, v in rows.items() if v["failed"]}
    if bad:
        print(f"  failed rows: {bad}")
    return clean


def main():
    if len(sys.argv) == 2:
        summarize(parse_log(sys.argv[1]), sys.argv[1])
        return
    base_path, probe_path = sys.argv[1], sys.argv[2]
    base = summarize(parse_log(base_path), "baseline")
    probe = summarize(parse_log(probe_path), "probe")
    common = sorted(set(base) & set(probe))
    print(f"paired clean rows: {len(common)}")
    if not common:
        return
    b = [base[i]["mse"] for i in common]
    p = [probe[i]["mse"] for i in common]
    mb = sum(b) / len(b)
    mp = sum(p) / len(p)
    # Truth floor + route bias estimate: F+B = 2*MSE_probe - MSE_base
    # (valid when probe uses 2x the blocks of baseline and MSE = V/blocks + F + B)
    d = [2.0 * pi - bi for pi, bi in zip(p, b)]
    md = sum(d) / len(d)
    var = sum((x - md) ** 2 for x in d) / (len(d) - 1)
    se = (var / len(d)) ** 0.5
    print(f"baseline mean MSE: {mb:.6e}")
    print(f"probe    mean MSE: {mp:.6e}   ratio {mp/mb:.4f} (pure-variance-no-floor predicts 0.5)")
    print(f"F+B = 2*probe - base (paired): {md:.6e} +/- {se:.6e} (1 s.e.)")
    print(f"implied variance part of baseline MSE: {mb - md:.6e}")


if __name__ == "__main__":
    main()
