from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import flopscope as flops
from local_engine import build_mlp
import estimator


WIDTH = 256
DEPTH = 32
SEEDS = (11, 22, 33)
OUTDIR = REPO / "paired_fly_logs" / "fingerprint_theory"
JSON_PATH = OUTDIR / "r1_bias_measurement_20260705.json"
MD_PATH = OUTDIR / "r1_bias_measurement_20260705.md"


def to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def mlp_weights_np(mlp: Any) -> list[np.ndarray]:
    return [to_numpy(w) for w in mlp.weights]


def antithetic_truth(
    weights: list[np.ndarray],
    *,
    n_samples: int,
    seed: int,
    batch_pairs: int,
) -> dict[str, Any]:
    if n_samples % 2:
        raise ValueError("n_samples must be even for antithetic truth")
    n_pairs = n_samples // 2
    rng = np.random.default_rng(seed)
    depth = len(weights)
    width = weights[0].shape[0]
    sum_pair = np.zeros((depth, width), dtype=np.float64)
    sumsq_pair = np.zeros((depth, width), dtype=np.float64)
    done = 0
    started = time.time()

    while done < n_pairs:
        b = min(batch_pairs, n_pairs - done)
        x0 = rng.standard_normal((b, width), dtype=np.float32)
        x = np.concatenate((x0, -x0), axis=0)
        pair_rows = []
        for w in weights:
            x = np.maximum(x @ w, 0.0)
            pair_rows.append((x[:b] + x[b:]) * 0.5)
        pair = np.stack(pair_rows, axis=0).astype(np.float64, copy=False)
        sum_pair += pair.sum(axis=1)
        sumsq_pair += (pair * pair).sum(axis=1)
        done += b

    mean = sum_pair / n_pairs
    var_pair = np.maximum(sumsq_pair / n_pairs - mean * mean, 0.0)
    noise_by_layer = var_pair.mean(axis=1) / n_pairs
    return {
        "mean": mean.astype(np.float32),
        "truth_noise_mse_by_layer": noise_by_layer,
        "truth_noise_final_mse": float(noise_by_layer[-1]),
        "truth_noise_all_layer_mse": float(noise_by_layer.mean()),
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "seed": seed,
        "batch_pairs": batch_pairs,
        "wall_time_s": time.time() - started,
    }


def run_r1(mlp: Any, *, budget: int) -> dict[str, Any]:
    gc.collect()
    started = time.time()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as ctx:
        pred = estimator._factorized_k3_propagation(mlp)
    wall = time.time() - started
    return {
        "prediction": to_numpy(pred),
        "flops_used": int(ctx.flops_used),
        "wall_time_s": wall,
        "summary": ctx.summary(),
        "summary_dict": ctx.summary_dict(),
    }


def run_k2_if_available(mlp: Any, *, budget: int) -> dict[str, Any] | None:
    path = REPO / "estimator_covariance.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("estimator_covariance_probe", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "_covariance_plus_sampling"):
        return None
    old_mode = os.environ.get("WHEST_EXPERIMENT_MODE")
    old_frac = os.environ.get("WHEST_EXPERIMENT_BUDGET_FRACTION")
    os.environ["WHEST_EXPERIMENT_BUDGET_FRACTION"] = "0"
    try:
        gc.collect()
        started = time.time()
        with flops.BudgetContext(flop_budget=budget, quiet=True) as ctx:
            pred = mod._covariance_plus_sampling(mlp, budget)
        return {
            "route": "_covariance_plus_sampling with sample fraction forced to 0",
            "prediction": to_numpy(pred),
            "flops_used": int(ctx.flops_used),
            "wall_time_s": time.time() - started,
            "summary": ctx.summary(),
            "summary_dict": ctx.summary_dict(),
        }
    finally:
        if old_mode is None:
            os.environ.pop("WHEST_EXPERIMENT_MODE", None)
        else:
            os.environ["WHEST_EXPERIMENT_MODE"] = old_mode
        if old_frac is None:
            os.environ.pop("WHEST_EXPERIMENT_BUDGET_FRACTION", None)
        else:
            os.environ["WHEST_EXPERIMENT_BUDGET_FRACTION"] = old_frac


def op_rows(summary_dict: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    ops = summary_dict.get("operations") or summary_dict.get("by_operation") or {}
    rows = []
    if isinstance(ops, dict):
        for name, data in ops.items():
            if isinstance(data, dict):
                flp = (
                    data.get("flop_cost")
                    or data.get("flops")
                    or data.get("flops_used")
                    or data.get("total_flops")
                    or 0
                )
                calls = data.get("calls") or data.get("count") or 0
            else:
                flp = data
                calls = None
            rows.append({"op": name, "flops": int(flp), "calls": calls})
    return sorted(rows, key=lambda r: r["flops"], reverse=True)[:limit]


def metrics(pred: np.ndarray, truth: np.ndarray, noise_by_layer: np.ndarray) -> dict[str, Any]:
    layer_mse = ((pred.astype(np.float64) - truth.astype(np.float64)) ** 2).mean(axis=1)
    return {
        "final_layer_mse": float(layer_mse[-1]),
        "all_layer_mse": float(layer_mse.mean()),
        "layer_mse": layer_mse.tolist(),
        "net_bias_final_mse": float(layer_mse[-1] - noise_by_layer[-1]),
        "net_bias_all_layer_mse": float(layer_mse.mean() - noise_by_layer.mean()),
    }


def serializable_run(run: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run.items() if k != "prediction"}


def write_outputs(results: dict[str, Any]) -> None:
    JSON_PATH.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    r1 = results["r1"]
    pooled = r1["pooled"]
    verdict = results["verdict"]
    lines = [
        "# r1 depth-32 bias measurement (2026-07-05)",
        "",
        "Offline local MLPs from `local_engine.build_mlp(width=256, depth=32)`; MC truth uses fresh antithetic N(0,I) samples generated by this script. No Fly, network, pytest, tracked-file edits, public suites, or grader internals.",
        "",
        "## Verdict",
        "",
        f"- Pooled r1 final-layer MSE vs MC truth: `{pooled['final_layer_mse']:.9e}`",
        f"- Pooled estimated truth-noise MSE: `{pooled['truth_noise_final_mse']:.9e}`",
        f"- Pooled net final-layer bias MSE: `{pooled['net_bias_final_mse']:.9e}`",
        f"- Verdict: **{verdict}**",
        "",
        "## Per-MLP r1 results",
        "",
        "| seed | samples | r1 final MSE | truth noise | net bias MSE | all-layer MSE | raw FLOPs | wall s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r1["per_mlp"]:
        lines.append(
            f"| {row['seed']} | {row['truth']['n_samples']} | "
            f"{row['metrics']['final_layer_mse']:.9e} | "
            f"{row['truth']['truth_noise_final_mse']:.9e} | "
            f"{row['metrics']['net_bias_final_mse']:.9e} | "
            f"{row['metrics']['all_layer_mse']:.9e} | "
            f"{row['r1']['flops_used']} | {row['r1']['wall_time_s']:.1f} |"
        )
    lines += [
        "",
        "## All-layer profile",
        "",
        "`pooled_layer_mse` and `pooled_truth_noise_by_layer` are in the JSON result.",
        "",
        "## r1 cost profile",
        "",
        "| op | FLOPs | calls |",
        "|---|---:|---:|",
    ]
    for op in r1["top_ops"]:
        lines.append(f"| {op['op']} | {op['flops']} | {op.get('calls')} |")
    if results.get("k2"):
        lines += ["", "## K=2 covariance route", ""]
        for row in results["k2"]["per_mlp"]:
            lines.append(
                f"- seed {row['seed']}: final MSE `{row['metrics']['final_layer_mse']:.9e}`, "
                f"net bias `{row['metrics']['net_bias_final_mse']:.9e}`, raw FLOPs `{row['k2']['flops_used']}`"
            )
    lines += [
        "",
        "## Recommended next action",
        "",
        results["recommended_next_action"],
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--escalation-samples", type=int, default=4_000_000)
    ap.add_argument("--batch-pairs", type=int, default=4096)
    ap.add_argument("--budget", type=int, default=10**15)
    ap.add_argument("--skip-k2", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "seeds": list(SEEDS),
            "samples": args.samples,
            "batch_pairs": args.batch_pairs,
            "budget": args.budget,
        },
        "r1": {"per_mlp": []},
    }

    preds = []
    truths = []
    noises = []
    for seed in SEEDS:
        print(f"[r1] seed {seed}: building MLP and MC truth ({args.samples} samples)", flush=True)
        mlp = build_mlp(WIDTH, DEPTH, seed=seed)
        weights = mlp_weights_np(mlp)
        truth = antithetic_truth(weights, n_samples=args.samples, seed=10_000 + seed, batch_pairs=args.batch_pairs)
        print(f"[r1] seed {seed}: running real _factorized_k3_propagation", flush=True)
        r1_run = run_r1(mlp, budget=args.budget)
        m = metrics(r1_run["prediction"], truth["mean"], truth["truth_noise_mse_by_layer"])
        preds.append(r1_run["prediction"].astype(np.float64))
        truths.append(truth["mean"].astype(np.float64))
        noises.append(truth["truth_noise_mse_by_layer"])
        results["r1"]["per_mlp"].append(
            {
                "seed": seed,
                "truth": {k: v for k, v in truth.items() if k != "mean" and k != "truth_noise_mse_by_layer"},
                "truth_noise_mse_by_layer": truth["truth_noise_mse_by_layer"].tolist(),
                "r1": serializable_run(r1_run),
                "metrics": m,
            }
        )
        print(
            f"[r1] seed {seed}: final_mse={m['final_layer_mse']:.9e} "
            f"noise={truth['truth_noise_final_mse']:.9e} flops={r1_run['flops_used']}",
            flush=True,
        )

    pooled_layer = ((np.stack(preds) - np.stack(truths)) ** 2).mean(axis=(0, 2))
    pooled_noise = np.stack(noises).mean(axis=0)
    pooled_final = float(pooled_layer[-1])
    pooled_noise_final = float(pooled_noise[-1])
    net_final = pooled_final - pooled_noise_final
    results["r1"]["pooled"] = {
        "final_layer_mse": pooled_final,
        "truth_noise_final_mse": pooled_noise_final,
        "net_bias_final_mse": net_final,
        "all_layer_mse": float(pooled_layer.mean()),
        "truth_noise_all_layer_mse": float(pooled_noise.mean()),
        "net_bias_all_layer_mse": float(pooled_layer.mean() - pooled_noise.mean()),
        "pooled_layer_mse": pooled_layer.tolist(),
        "pooled_truth_noise_by_layer": pooled_noise.tolist(),
        "mean_flops_used": float(np.mean([r["r1"]["flops_used"] for r in results["r1"]["per_mlp"]])),
    }
    results["r1"]["top_ops"] = op_rows(results["r1"]["per_mlp"][0]["r1"]["summary_dict"])

    if pooled_final < 0.12e-6:
        seed = SEEDS[0]
        print(f"[escalation] pooled r1 MSE {pooled_final:.9e}; rerunning seed {seed} at {args.escalation_samples}", flush=True)
        mlp = build_mlp(WIDTH, DEPTH, seed=seed)
        truth = antithetic_truth(mlp_weights_np(mlp), n_samples=args.escalation_samples, seed=20_000 + seed, batch_pairs=args.batch_pairs)
        r1_run = run_r1(mlp, budget=args.budget)
        results["r1"]["escalation"] = {
            "seed": seed,
            "truth": {k: v for k, v in truth.items() if k != "mean" and k != "truth_noise_mse_by_layer"},
            "truth_noise_mse_by_layer": truth["truth_noise_mse_by_layer"].tolist(),
            "r1": serializable_run(r1_run),
            "metrics": metrics(r1_run["prediction"], truth["mean"], truth["truth_noise_mse_by_layer"]),
        }

    if not args.skip_k2:
        print("[k2] attempting covariance route from estimator_covariance.py", flush=True)
        k2_rows = []
        try:
            for seed, truth_mean, noise in zip(SEEDS, truths, noises):
                mlp = build_mlp(WIDTH, DEPTH, seed=seed)
                k2_run = run_k2_if_available(mlp, budget=args.budget)
                if k2_run is None:
                    break
                k2_rows.append(
                    {
                        "seed": seed,
                        "k2": serializable_run(k2_run),
                        "metrics": metrics(k2_run["prediction"], truth_mean.astype(np.float32), noise),
                    }
                )
            if k2_rows:
                results["k2"] = {"per_mlp": k2_rows}
        except Exception as exc:
            results["k2_error"] = f"{type(exc).__name__}: {exc}"

    if net_final <= 0.1e-6:
        results["verdict"] = "ANALYTIC LANE CONFIRMED: r1 net bias <= 0.1e-6"
        results["recommended_next_action"] = (
            "Engineer the exact r1 route down from ~2.31e11 raw FLOPs toward the <=1.35e11 target. "
            "Start with the dominant ops in the cost profile: reduce/avoid the largest einsum contractions, "
            "exploit symmetry/diagonal shortcuts in repeated degree-3/degree-4 updates, and consider layer-suffix or "
            "structured low-rank cuts only if they preserve this measured bias."
        )
    elif net_final <= 0.5e-6:
        results["verdict"] = "PARTIALLY VIABLE: K=3 r1 bias is between 0.1e-6 and 0.5e-6"
        results["recommended_next_action"] = "Treat K=3 as relevant but insufficient for entry-1 accuracy; inspect higher-order or correction terms before heavy FLOP engineering."
    else:
        results["verdict"] = "NEGATIVE: K=3-class r1 analytics do not explain entry-1 accuracy"
        results["recommended_next_action"] = "Do not spend primary effort compressing exact r1 for the top cluster; redirect to the Hadamard/variance route or another mechanism."

    write_outputs(results)
    print(json.dumps({"pooled": results["r1"]["pooled"], "verdict": results["verdict"], "json": str(JSON_PATH), "md": str(MD_PATH)}, indent=2))


if __name__ == "__main__":
    main()
