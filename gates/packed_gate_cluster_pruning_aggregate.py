#!/usr/bin/env python3
"""Aggregate the packed gate-clustered pruning Fly payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MLPS = 100
RAW_GATE = 2.4e10
PRODUCT_GATE = 0.31
STRONG_PRODUCT_GATE = 0.29
MAX_BUCKET_MEAN = 8.0
MAX_BUCKET_ANY = 16
MIN_GROUPS_PER_BUCKET = 4.0
MAX_PEAK_BYTES = 512 * 1024 * 1024
MAX_ACTIVATION_TRAFFIC = 2.5


def qstats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("ok") is True and "combos" in obj:
            rows.append(obj)
        elif isinstance(obj.get("result"), dict):
            result = obj["result"]
            if result.get("ok") is True and "combos" in result:
                rows.append(result)
    dedup = {int(row["mlp_index"]): row for row in rows}
    return [dedup[index] for index in sorted(dedup)]


def aggregate(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    indices = [int(row["mlp_index"]) for row in rows]
    complete = len(rows) == EXPECTED_MLPS and indices == list(range(EXPECTED_MLPS))
    checksum_ok = all(bool(row.get("checksum_ok")) for row in rows)
    if not rows:
        return {
            "complete": False,
            "n_mlps": 0,
            "verdict": "FAIL: no successful payload rows",
            "plans": [],
        }

    combo_keys = sorted(set.intersection(*(set(row["combos"]) for row in rows)))
    combo_summaries: dict[str, Any] = {}
    plans: list[dict[str, Any]] = []

    for combo_key in combo_keys:
        sample = rows[0]["combos"][combo_key]
        layers = [int(value) for value in sample["layers"]]
        layer_summary: dict[str, Any] = {
            "layers": layers,
            "live_input_mean": [],
            "true_output_dead_mean": [],
            "box": {},
            "combined": {},
        }
        for field in ("live_input_mean", "true_output_dead_mean"):
            matrix = np.asarray([row["combos"][combo_key][field] for row in rows], dtype=np.float64)
            layer_summary[field] = {
                "mean": np.mean(matrix, axis=0).tolist(),
                "q10": np.quantile(matrix, 0.10, axis=0).tolist(),
                "q90": np.quantile(matrix, 0.90, axis=0).tolist(),
            }
        decision_mask = np.asarray(layers, dtype=np.int64) >= 3
        per_mlp_live = np.asarray([
            np.mean(np.asarray(row["combos"][combo_key]["live_input_mean"], dtype=np.float64)[decision_mask])
            for row in rows
        ])
        per_mlp_true_dead = np.asarray([
            np.mean(np.asarray(row["combos"][combo_key]["true_output_dead_mean"], dtype=np.float64)[decision_mask])
            for row in rows
        ])
        # For group live fraction L and true-dead fraction D,
        # L(1-D) >= L-D. This lower-bounds even an oracle certificate.
        oracle_lower_bound = per_mlp_live - per_mlp_true_dead
        layer_summary["decision_live_input"] = qstats(per_mlp_live)
        layer_summary["decision_true_output_dead"] = qstats(per_mlp_true_dead)
        layer_summary["oracle_product_lower_bound"] = qstats(oracle_lower_bound)

        for cert_name in ("box", "combined"):
            layer_summary[cert_name] = {}
            for field in (
                "certified_output_dead_mean",
                "certificate_recall",
                "product_mean",
            ):
                matrix = np.asarray(
                    [row["combos"][combo_key][cert_name][field] for row in rows],
                    dtype=np.float64,
                )
                layer_summary[cert_name][field] = {
                    "mean": np.mean(matrix, axis=0).tolist(),
                    "q10": np.quantile(matrix, 0.10, axis=0).tolist(),
                    "q90": np.quantile(matrix, 0.90, axis=0).tolist(),
                }

            decision_products = [
                float(row["combos"][combo_key][cert_name]["decision_product_mean"])
                for row in rows
            ]
            recalls = [
                float(row["combos"][combo_key][cert_name]["decision_certificate_recall"])
                for row in rows
            ]
            violations = sum(
                int(row["combos"][combo_key][cert_name]["total_violations"])
                for row in rows
            )
            low_rank_only_violations = sum(
                int(row["combos"][combo_key].get("low_rank_only_violations_total", 0))
                for row in rows
            )

            for quantum in (16, 32, 64):
                qkey = str(quantum)
                projections = [
                    row["combos"][combo_key][cert_name]["projection"][qkey]
                    for row in rows
                ]
                projected_raw = [float(value["projected_raw_b28"]) for value in projections]
                bucket_mean = [float(value["bucket_count_mean"]) for value in projections]
                bucket_max = max(int(value["bucket_count_max"]) for value in projections)
                groups_median = [float(value["groups_per_bucket_median"]) for value in projections]
                peak_bytes = max(int(value["projected_peak_bucket_bytes_b28"]) for value in projections)
                activation_traffic = [float(value["activation_traffic_ratio_max"]) for value in projections]
                total_traffic = [float(value["total_traffic_ratio_max"]) for value in projections]
                product_stats = qstats(decision_products)
                raw_stats = qstats(projected_raw)
                packing_pass = bool(
                    np.mean(bucket_mean) <= MAX_BUCKET_MEAN
                    and bucket_max <= MAX_BUCKET_ANY
                    and np.median(groups_median) >= MIN_GROUPS_PER_BUCKET
                    and peak_bytes <= MAX_PEAK_BYTES
                    and max(activation_traffic) <= MAX_ACTIVATION_TRAFFIC
                    and quantum in (32, 64)
                )
                arithmetic_pass = bool(
                    product_stats["mean"] <= PRODUCT_GATE
                    and raw_stats["q90"] <= RAW_GATE
                    and violations == 0
                    and (cert_name != "combined" or low_rank_only_violations == 0)
                )
                plan = {
                    "combo": combo_key,
                    "strategy": sample["strategy"],
                    "group_size": int(sample["group_size"]),
                    "certificate": cert_name,
                    "padding": quantum,
                    "decision_product": product_stats,
                    "oracle_product_lower_bound": qstats(oracle_lower_bound),
                    "decision_certificate_recall": qstats(recalls),
                    "projected_raw_b28": raw_stats,
                    "violations": int(violations),
                    "low_rank_only_violations": int(low_rank_only_violations),
                    "bucket_count_mean": qstats(bucket_mean),
                    "bucket_count_max": bucket_max,
                    "groups_per_bucket_median": qstats(groups_median),
                    "peak_bucket_bytes_b28": peak_bytes,
                    "activation_traffic_ratio_max": max(activation_traffic),
                    "total_traffic_ratio": qstats(total_traffic),
                    "arithmetic_pass": arithmetic_pass,
                    "packing_pass": packing_pass,
                    "strong_product_pass": product_stats["mean"] <= STRONG_PRODUCT_GATE,
                    "gate_pass": bool(complete and checksum_ok and arithmetic_pass and packing_pass),
                }
                plans.append(plan)

        combo_summaries[combo_key] = layer_summary

    plans.sort(key=lambda plan: (plan["projected_raw_b28"]["q90"], plan["decision_product"]["mean"]))
    passing = [plan for plan in plans if plan["gate_pass"]]
    selected = passing[0] if passing else None
    if not complete:
        verdict = f"FAIL: incomplete Fly payload ({len(rows)}/{EXPECTED_MLPS} unique MLPs)"
    elif not checksum_ok:
        verdict = "FAIL: one or more truth-bank MLP checksum mismatches"
    elif selected is None:
        arithmetic = [plan for plan in plans if plan["arithmetic_pass"]]
        if arithmetic:
            verdict = "FAIL: arithmetic gate passed for at least one plan, but packed-kernel residual/memory proxy failed"
        else:
            verdict = "FAIL: no plan passed the frozen product/raw/certificate gate; close two-sided pruning"
    else:
        tier = "strong" if selected["strong_product_pass"] else "borderline"
        verdict = f"PASS ({tier}): design one exact mode-gated packed implementation from the selected plan"

    implementation_design = None
    if selected is not None:
        implementation_design = {
            "mode_only": True,
            "blocks": 28,
            "strategy": selected["strategy"],
            "group_size": selected["group_size"],
            "certificate": selected["certificate"],
            "padding": selected["padding"],
            "steps": [
                "Run the unchanged current first layer, exact recolor, and 1.5x first-successor transform.",
                "Before each weight layer 2..31, derive the selected stable label-free row ordering and fixed-size groups.",
                "For each group, gather the exact nonzero input union and compute the registered rigorous dead-output certificate with fp32 slack.",
                "Round live-input and uncertified-output widths to the selected padding, bucket groups by the padded pair, and execute one batched rectangular kernel per bucket.",
                "Scatter uncertified outputs, set certified post-ReLU outputs to exact zero, preserve row identity/permutation, and continue without approximation.",
                "Require py_compile plus full-100 paired Fly proof with zero prediction/MSE delta before any default consideration.",
            ],
        }

    return {
        "complete": complete,
        "checksum_ok": checksum_ok,
        "n_mlps": len(rows),
        "mlp_indices": indices,
        "script_versions": sorted(set(str(row.get("script_version")) for row in rows)),
        "wall_time_s": qstats([float(row["wall_time_s"]) for row in rows]),
        "thresholds": {
            "decision_product": PRODUCT_GATE,
            "strong_decision_product": STRONG_PRODUCT_GATE,
            "projected_raw_b28_q90": RAW_GATE,
            "certificate_violations": 0,
            "bucket_count_mean": MAX_BUCKET_MEAN,
            "bucket_count_max": MAX_BUCKET_ANY,
            "groups_per_bucket_median": MIN_GROUPS_PER_BUCKET,
            "peak_bytes": MAX_PEAK_BYTES,
            "activation_traffic_ratio": MAX_ACTIVATION_TRAFFIC,
        },
        "verdict": verdict,
        "selected_plan": selected,
        "implementation_design": implementation_design,
        "plans": plans,
        "combo_layer_summaries": combo_summaries,
    }


def render_markdown(result: dict[str, Any], source: Path) -> str:
    lines = [
        "# Packed gate-clustered two-sided pruning gate",
        "",
        f"Source: `{source}`. Successful unique shards: `{result['n_mlps']}/100`; checksums: "
        f"`{'PASS' if result.get('checksum_ok') else 'FAIL'}`.",
        "",
        "The payload reproduced the real current 16-block route machine-side and used full dense preactivations only to measure true output-dead coordinates and certificate violations. Truth-bank means were not read. All candidate sorting and certificates were label-free functions of the MLP and its own route activations.",
        "",
        "## Verdict",
        "",
        f"**{result['verdict']}**",
        "",
        "Frozen gates: decision-layer mean `live_input x uncertified_output <= 0.31` (`<=0.29` strong), per-MLP q90 projected 28-block raw `<=2.4e10`, zero certificate violations, and a padding-32/64 plan with mean/max buckets `<=8/16`, median groups per bucket `>=4`, peak packed memory `<=512 MiB`, and activation gather/scatter ratio `<=2.5x`.",
        "",
    ]
    oracle_plans = sorted(
        result.get("plans", []),
        key=lambda plan: plan["oracle_product_lower_bound"]["mean"],
    )
    if oracle_plans:
        oracle = oracle_plans[0]
        lines.extend([
            "Even an oracle output-dead certificate cannot reach the product gate. "
            f"The best grouping (`{oracle['strategy']}`, G={oracle['group_size']}) obeys "
            f"`E[L(1-D)] >= E[L]-E[D] = {oracle['oracle_product_lower_bound']['mean']:.4f}` "
            f"(per-MLP q10/q90 `{oracle['oracle_product_lower_bound']['q10']:.4f}`/"
            f"`{oracle['oracle_product_lower_bound']['q90']:.4f}`), already above `0.31` before screening or padding costs.",
            "",
        ])
    lines.extend([
        "## Best projected plans",
        "",
        "| rank | strategy | G | cert | pad | product mean | cert recall med | raw28 q90 | violations | buckets mean/max | groups/bucket med | act traffic max | arithmetic | packing | gate |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for rank, plan in enumerate(result.get("plans", [])[:24], start=1):
        lines.append(
            f"| {rank} | {plan['strategy']} | {plan['group_size']} | {plan['certificate']} | {plan['padding']} | "
            f"{plan['decision_product']['mean']:.4f} | {plan['decision_certificate_recall']['median']:.3f} | "
            f"{plan['projected_raw_b28']['q90']:.3e} | {plan['violations']} | "
            f"{plan['bucket_count_mean']['mean']:.1f}/{plan['bucket_count_max']} | "
            f"{plan['groups_per_bucket_median']['median']:.1f} | {plan['activation_traffic_ratio_max']:.2f} | "
            f"{'PASS' if plan['arithmetic_pass'] else 'fail'} | {'PASS' if plan['packing_pass'] else 'fail'} | "
            f"{'PASS' if plan['gate_pass'] else 'fail'} |"
        )

    controls = [
        plan for plan in result.get("plans", [])
        if plan["strategy"] == "contiguous" and plan["padding"] == 32
    ]
    lines.extend([
        "",
        "## Unsorted contiguous controls (padding 32)",
        "",
        "| G | cert | product mean [q10,q90] | raw28 q90 | recall median | buckets mean/max |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for plan in controls:
        lines.append(
            f"| {plan['group_size']} | {plan['certificate']} | {plan['decision_product']['mean']:.4f} "
            f"[{plan['decision_product']['q10']:.4f},{plan['decision_product']['q90']:.4f}] | "
            f"{plan['projected_raw_b28']['q90']:.3e} | {plan['decision_certificate_recall']['median']:.3f} | "
            f"{plan['bucket_count_mean']['mean']:.1f}/{plan['bucket_count_max']} |"
        )

    if result.get("selected_plan") is not None:
        plan = result["selected_plan"]
        lines.extend([
            "",
            "## Exact mode design (gate passed; not implemented here)",
            "",
            f"Use 28 blocks, `{plan['strategy']}` sorting, groups of {plan['group_size']}, the `{plan['certificate']}` certificate, and padding {plan['padding']}. Bucket by padded `(live input, uncertified output)` and issue one batched rectangular kernel per nonempty bucket. Certified outputs become exact post-ReLU zeros. Keep the default untouched until a mode-gated implementation shows zero paired prediction/MSE delta on all 100 fixed Fly MLPs.",
        ])
    else:
        lines.extend([
            "",
            "## Closeout",
            "",
            "No estimator-mode design is authorized by this gate. The negative result closes packed two-sided pruning as formulated; do not edit `estimator.py` from this lane.",
        ])

    lines.extend([
        "",
        "Full per-plan statistics and per-layer mean/q10/q90 curves are in the adjacent results JSON.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.jsonl)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(render_markdown(result, args.jsonl), encoding="utf-8")
    print(json.dumps({
        "n_mlps": result["n_mlps"],
        "verdict": result["verdict"],
        "selected_plan": result["selected_plan"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
