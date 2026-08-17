#!/usr/bin/env python3
"""Aggregate the positive-homogeneity angular-importance Fly gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("ok") is True and obj.get("script") == "angular-importance-gate-v1":
            rows.append(obj)
    return sorted(rows, key=lambda row: int(row["mlp_index"]))


def _spread(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def aggregate(rows: list[dict]) -> dict[str, object]:
    if not rows:
        raise ValueError("no successful angular-importance rows")
    primary_ratio = np.asarray([row["primary"]["proposal_ratio"] for row in rows])
    primary_vp = np.asarray([row["primary"]["baseline_variance"] for row in rows])
    primary_vq = np.asarray([row["primary"]["proposal_variance"] for row in rows])
    raw_vq = np.asarray([row["raw"]["proposal_variance"] for row in rows])
    oracle_vq = np.asarray([row["primary"]["oracle_variance"] for row in rows])
    direct_vq = np.asarray([row["direct"]["proposal_variance"] for row in rows])
    identity_second = np.asarray(
        [row["primary"]["proposal_second_moment"] for row in rows]
    )
    direct_second = np.asarray([row["direct"]["proposal_second_moment"] for row in rows])
    ess = np.asarray([row["direct"]["ess_fraction"] for row in rows])
    max_weight = np.asarray([row["direct"]["weight_max"] for row in rows])
    mean_stat = np.asarray([row["direct"]["mean_difference_stat"] for row in rows])
    direct_ratio = np.asarray([row["direct"]["proposal_ratio"] for row in rows])
    holdout_corr = np.asarray([row["fit"]["holdout_pearson"] for row in rows])
    holdout_rank_corr = np.asarray([row["fit"]["holdout_spearman"] for row in rows])
    train_r2 = np.asarray([row["fit"]["train_r2"] for row in rows])
    active = np.asarray([row["fit"]["active_slopes"] for row in rows])
    uniform_mass = np.asarray([row["mixture"]["uniform_mass"] for row in rows])
    safe_ratio_mean = np.asarray(
        [row["normalizer"]["heldout_safe_ratio_mean"] for row in rows]
    )

    pooled_ratio = float(np.sum(primary_vp) / np.sum(primary_vq))
    pooled_fraction = float(0.125 + 0.875 / pooled_ratio)
    pooled_raw_ratio = float(np.sum(primary_vp) / np.sum(raw_vq))
    pooled_oracle_ratio = float(np.sum(primary_vp) / np.sum(oracle_vq))
    direct_second_agreement = float(np.sum(direct_second) / np.sum(identity_second))
    direct_variance_agreement = float(np.sum(direct_vq) / np.sum(primary_vq))
    pooled_mean_stat = float(np.mean(mean_stat))

    performance_pass = bool(
        pooled_ratio >= 2.1
        and np.median(primary_ratio) >= 1.8
        and np.quantile(primary_ratio, 0.10) >= 1.05
        and pooled_fraction <= 0.58
    )
    tail_pass = bool(
        np.max(max_weight) <= 10.000001
        and np.median(ess) >= 0.50
        and np.quantile(ess, 0.10) >= 0.25
    )
    direct_pass = bool(
        0.80 <= direct_second_agreement <= 1.20
        and 0.80 <= direct_variance_agreement <= 1.20
        and 0.50 <= pooled_mean_stat <= 1.50
    )

    return {
        "gate": "positive-homogeneity-angular-importance-v1",
        "n_mlps": len(rows),
        "all_checksums_ok": bool(all(row.get("checksum_ok") is True for row in rows)),
        "counts": {
            "pilot_pairs_per_mlp": int(rows[0]["pilot_pairs"]),
            "holdout_pairs_per_mlp": int(rows[0]["holdout_pairs"]),
            "direct_pairs_per_mlp": int(rows[0]["direct_pairs"]),
            "pilot_fraction": float(rows[0]["pilot_fraction"]),
        },
        "primary_safe10": {
            "pooled_proposal_ratio": pooled_ratio,
            "per_mlp_proposal_ratio": _spread(primary_ratio),
            "pooled_projected_total_fraction": pooled_fraction,
            "pooled_projected_total_gain": 1.0 / pooled_fraction,
            "direct_proposal_ratio": _spread(direct_ratio),
        },
        "oracle_ceiling": {
            "pooled_proposal_ratio": pooled_oracle_ratio,
            "pooled_projected_total_fraction": float(0.125 + 0.875 / pooled_oracle_ratio),
            "per_mlp_proposal_ratio": _spread(primary_vp / oracle_vq),
        },
        "raw_fitted_diagnostic": {
            "pooled_proposal_ratio": pooled_raw_ratio,
            "per_mlp_proposal_ratio": _spread(primary_vp / raw_vq),
        },
        "surrogate": {
            "pilot_train_r2": _spread(train_r2),
            "holdout_pearson": _spread(holdout_corr),
            "holdout_spearman": _spread(holdout_rank_corr),
            "active_slopes": _spread(active),
            "safe_uniform_component_mass": _spread(uniform_mass),
            "heldout_likelihood_ratio_mean": _spread(safe_ratio_mean),
        },
        "importance_tail": {
            "ess_fraction": _spread(ess),
            "weight_max_per_mlp": _spread(max_weight),
            "global_weight_max": float(np.max(max_weight)),
        },
        "direct_validation": {
            "pooled_second_moment_direct_over_identity": direct_second_agreement,
            "pooled_variance_direct_over_identity": direct_variance_agreement,
            "mean_difference_stat": _spread(mean_stat),
            "pooled_mean_difference_stat": pooled_mean_stat,
        },
        "decision": {
            "performance_pass": performance_pass,
            "tail_pass": tail_pass,
            "direct_validation_pass": direct_pass,
            "overall_pass": bool(
                len(rows) == 100
                and all(row.get("checksum_ok") is True for row in rows)
                and performance_pass
                and tail_pass
                and direct_pass
            ),
        },
        "per_mlp": [
            {
                "mlp_index": int(row["mlp_index"]),
                "proposal_ratio": float(row["primary"]["proposal_ratio"]),
                "projected_total_fraction": float(
                    row["primary"]["projected_total_fraction"]
                ),
                "oracle_ratio": float(row["primary"]["oracle_ratio"]),
                "raw_ratio": float(row["raw"]["proposal_ratio"]),
                "direct_ratio": float(row["direct"]["proposal_ratio"]),
                "ess_fraction": float(row["direct"]["ess_fraction"]),
                "weight_max": float(row["direct"]["weight_max"]),
                "holdout_pearson": float(row["fit"]["holdout_pearson"]),
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(_read_rows(args.jsonl))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
