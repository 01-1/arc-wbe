from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_engine import build_mlp


WIDTH = 256
DEPTH = 32
SEEDS = (11, 22)
MS = (1, 2, 4, 8, 16)
OUTDIR = REPO / "paired_fly_logs" / "fingerprint_theory"
JSON_PATH = OUTDIR / "gaussian_sum_pretest_results.json"
MD_PATH = OUTDIR / "gaussian_sum_pretest_20260705.md"

MIN_VARIANCE = 1e-12
SPLIT_C = 0.70

GL16_NODES = np.array(
    [
        -0.9894009349916499,
        -0.9445750230732326,
        -0.8656312023878318,
        -0.7554044083550030,
        -0.6178762444026438,
        -0.4580167776572274,
        -0.2816035507792589,
        -0.0950125098376374,
        0.0950125098376374,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    ],
    dtype=np.float64,
)
GL16_WEIGHTS = np.array(
    [
        0.0271524594117541,
        0.0622535239386479,
        0.0951585116824928,
        0.1246289712555339,
        0.1495959888165767,
        0.1691565193950025,
        0.1826034150449236,
        0.1894506104550685,
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    ],
    dtype=np.float64,
)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def mlp_weights_np(seed: int) -> list[np.ndarray]:
    mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    return [np.asarray(w, dtype=np.float64) for w in mlp.weights]


def antithetic_truth(
    weights: list[np.ndarray],
    *,
    n_samples: int,
    seed: int,
    batch_pairs: int,
) -> dict[str, Any]:
    if n_samples % 2:
        raise ValueError("n_samples must be even")
    n_pairs = n_samples // 2
    rng = np.random.default_rng(seed)
    total = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    total2 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    done = 0
    started = time.time()
    while done < n_pairs:
        b = min(batch_pairs, n_pairs - done)
        x0 = rng.standard_normal((b, WIDTH)).astype(np.float32)
        x = np.concatenate((x0, -x0), axis=0).astype(np.float64)
        pair_rows = []
        for w in weights:
            x = np.maximum(x @ w, 0.0)
            pair_rows.append(0.5 * (x[:b] + x[b:]))
        rows = np.stack(pair_rows, axis=0)
        total += rows.sum(axis=1)
        total2 += (rows * rows).sum(axis=1)
        done += b
    mean = total / n_pairs
    var_pair = np.maximum(total2 / n_pairs - mean * mean, 0.0)
    noise = var_pair.mean(axis=1) / n_pairs
    return {
        "mean": mean,
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "truth_noise_mse_by_layer": noise.tolist(),
        "truth_noise_final_mse": float(noise[-1]),
        "wall_time_s": time.time() - started,
    }


def relu_mean_cov(mean_pre: np.ndarray, cov_pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    var = np.maximum(np.diag(cov_pre), MIN_VARIANCE)
    sigma = np.sqrt(var)
    beta = mean_pre / sigma
    phi = norm_pdf(beta)
    Phi = norm_cdf(beta)
    relu_mean = sigma * phi + mean_pre * Phi
    marginal_second = (mean_pre * mean_pre + var) * Phi + mean_pre * sigma * phi

    denom = sigma[:, None] * sigma[None, :]
    rho = np.clip(cov_pre / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    alpha_i = -beta[:, None]
    alpha_j = -beta[None, :]
    tail_i = Phi[:, None]
    tail_j = Phi[None, :]
    rho_int = np.zeros_like(rho)
    for node, weight in zip(GL16_NODES, GL16_WEIGHTS):
        r = 0.5 * rho * (node + 1.0)
        one_minus = np.maximum(1.0 - r * r, MIN_VARIANCE)
        exponent = -(
            alpha_i * alpha_i - 2.0 * r * alpha_i * alpha_j + alpha_j * alpha_j
        ) / (2.0 * one_minus)
        phi2 = np.exp(exponent) / (2.0 * math.pi * np.sqrt(one_minus))
        rho_int += weight * (rho - r) * phi2
    second = relu_mean[:, None] * relu_mean[None, :] + denom * (
        rho * tail_i * tail_j + 0.5 * rho * rho_int
    )
    np.fill_diagonal(second, marginal_second)
    cov = second - relu_mean[:, None] * relu_mean[None, :]
    cov = 0.5 * (cov + cov.T)
    np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE))
    return relu_mean, cov


def split_components(target_m: int) -> list[tuple[float, np.ndarray, np.ndarray]]:
    comps: list[tuple[float, np.ndarray, np.ndarray]] = [
        (1.0, np.zeros(WIDTH, dtype=np.float64), np.eye(WIDTH, dtype=np.float64))
    ]
    while len(comps) < target_m:
        scores = []
        for weight, _mean, cov in comps:
            vals = np.linalg.eigvalsh(cov)
            scores.append(weight * float(vals[-1]))
        idx = int(np.argmax(scores))
        weight, mean, cov = comps.pop(idx)
        vals, vecs = np.linalg.eigh(cov)
        lam = max(float(vals[-1]), MIN_VARIANCE)
        v = vecs[:, -1]
        offset = SPLIT_C * math.sqrt(lam) * v
        child_cov = cov - (SPLIT_C * SPLIT_C * lam) * np.outer(v, v)
        child_cov = 0.5 * (child_cov + child_cov.T)
        comps.append((0.5 * weight, mean + offset, child_cov.copy()))
        comps.append((0.5 * weight, mean - offset, child_cov.copy()))
    return comps


def split_component_list(
    comps: list[tuple[float, np.ndarray, np.ndarray]],
    target_m: int,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    comps = [(w, m.copy(), c.copy()) for w, m, c in comps]
    while len(comps) < target_m:
        scores = []
        for weight, _mean, cov in comps:
            vals = np.linalg.eigvalsh(cov)
            scores.append(weight * float(vals[-1]))
        idx = int(np.argmax(scores))
        weight, mean, cov = comps.pop(idx)
        vals, vecs = np.linalg.eigh(cov)
        lam = max(float(vals[-1]), MIN_VARIANCE)
        v = vecs[:, -1]
        offset = SPLIT_C * math.sqrt(lam) * v
        child_cov = cov - (SPLIT_C * SPLIT_C * lam) * np.outer(v, v)
        child_cov = 0.5 * (child_cov + child_cov.T)
        comps.append((0.5 * weight, mean + offset, child_cov.copy()))
        comps.append((0.5 * weight, mean - offset, child_cov.copy()))
    return comps


def gaussian_sum_propagate(weights: list[np.ndarray], target_m: int) -> dict[str, Any]:
    comps = split_components(target_m)
    layer_means = []
    started = time.time()
    for layer_idx, w in enumerate(weights):
        next_comps = []
        mix_mean = np.zeros(WIDTH, dtype=np.float64)
        layer_started = time.time()
        for weight, mean, cov in comps:
            pre_mean = mean @ w
            pre_cov = w.T @ cov @ w
            relu_mean, relu_cov = relu_mean_cov(pre_mean, pre_cov)
            next_comps.append((weight, relu_mean, relu_cov))
            mix_mean += weight * relu_mean
        comps = next_comps
        layer_means.append(mix_mean)
        print(
            f"  M={target_m} layer {layer_idx + 1:02d}/{DEPTH} "
            f"wall={time.time() - layer_started:.2f}s",
            flush=True,
        )
    return {
        "prediction": np.stack(layer_means, axis=0),
        "wall_time_s": time.time() - started,
        "split": {
            "strategy": "input-space recursive top-eigen split",
            "child_weight": 0.5,
            "mean_offset_c": SPLIT_C,
            "child_variance_fraction_along_split_axis": 1.0 - SPLIT_C * SPLIT_C,
        },
    }


def gaussian_sum_progressive_prerelu(weights: list[np.ndarray], target_m: int) -> dict[str, Any]:
    comps: list[tuple[float, np.ndarray, np.ndarray]] = [
        (1.0, np.zeros(WIDTH, dtype=np.float64), np.eye(WIDTH, dtype=np.float64))
    ]
    layer_means = []
    started = time.time()
    for layer_idx, w in enumerate(weights):
        pre_comps = []
        for weight, mean, cov in comps:
            pre_comps.append((weight, mean @ w, w.T @ cov @ w))
        layer_target = min(target_m, 2 ** (layer_idx + 1))
        pre_comps = split_component_list(pre_comps, layer_target)
        next_comps = []
        mix_mean = np.zeros(WIDTH, dtype=np.float64)
        layer_started = time.time()
        for weight, pre_mean, pre_cov in pre_comps:
            relu_mean, relu_cov = relu_mean_cov(pre_mean, pre_cov)
            next_comps.append((weight, relu_mean, relu_cov))
            mix_mean += weight * relu_mean
        comps = next_comps
        layer_means.append(mix_mean)
        print(
            f"  adaptive M={target_m} layer {layer_idx + 1:02d}/{DEPTH} "
            f"components={len(comps)} wall={time.time() - layer_started:.2f}s",
            flush=True,
        )
    return {
        "prediction": np.stack(layer_means, axis=0),
        "wall_time_s": time.time() - started,
        "split": {
            "strategy": "progressive pre-ReLU top-eigen split",
            "child_weight": 0.5,
            "mean_offset_c": SPLIT_C,
            "child_variance_fraction_along_split_axis": 1.0 - SPLIT_C * SPLIT_C,
        },
    }


def mse_metrics(pred: np.ndarray, truth: np.ndarray, noise: list[float]) -> dict[str, Any]:
    layer_mse = ((pred - truth) ** 2).mean(axis=1)
    final = float(layer_mse[-1])
    return {
        "final_layer_mse": final,
        "all_layer_mse": float(layer_mse.mean()),
        "net_bias_final_mse": float(final - noise[-1]),
        "layer_mse": layer_mse.tolist(),
    }


def fit_alpha(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_m: dict[int, list[float]] = {}
    for row in rows:
        if row["net_bias_final_mse"] > 0:
            by_m.setdefault(int(row["M"]), []).append(float(row["net_bias_final_mse"]))
    if len(by_m) < 2:
        return {"alpha": None, "intercept": None}
    xs = np.array(sorted(by_m), dtype=np.float64)
    ys = np.array([np.mean(by_m[int(m)]) for m in xs], dtype=np.float64)
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    alpha = -float(slope)
    return {"alpha": alpha, "intercept": float(intercept)}


def extrapolate(fit: dict[str, Any], targets: tuple[float, ...]) -> dict[str, Any]:
    if fit.get("alpha") is None:
        return {}
    alpha = fit["alpha"]
    if alpha <= 0:
        return {
            f"{target:.1e}": {"M": None, "raw_flops": None, "note": "non-decreasing fitted bias"}
            for target in targets
        }
    intercept = fit["intercept"]
    out = {}
    for target in targets:
        log_m = (intercept - math.log(target)) / alpha
        if log_m > 700:
            out[f"{target:.1e}"] = {
                "M": None,
                "raw_flops": None,
                "note": f"extrapolated log(M)={log_m:.1f} overflows double precision",
            }
        else:
            m = math.exp(log_m)
            out[f"{target:.1e}"] = {
                "M": m,
                "raw_flops": raw_flops(m),
            }
    return out


def raw_flops(m: float) -> float:
    width = WIDTH
    depth = DEPTH
    linear = 4.0 * width**3
    quadrature = 16.0 * 18.0 * width**2
    marginal = 50.0 * width
    return m * depth * (linear + quadrature + marginal)


def load_results() -> dict[str, Any]:
    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        for mlp in data.get("mlps", []):
            if "truth" in mlp and "mean" in mlp["truth"]:
                mlp["truth"]["mean"] = np.array(mlp["truth"]["mean"], dtype=np.float64)
            for run in mlp.get("runs", []):
                run.pop("prediction", None)
        return data
    return {
        "metadata": {
            "width": WIDTH,
            "depth": DEPTH,
            "seeds": list(SEEDS),
            "split_c": SPLIT_C,
            "notes": "Offline self-generated MLPs and MC truth; no Fly, network, pytest, or tracked edits.",
        },
        "mlps": [],
    }


def dump_json(data: dict[str, Any]) -> None:
    def clean(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    JSON_PATH.write_text(json.dumps(clean(data), indent=2, sort_keys=True), encoding="utf-8")


def write_report(data: dict[str, Any]) -> None:
    lines = [
        "# Gaussian-sum propagation pretest (2026-07-05)",
        "",
        "Offline local MLPs from `local_engine.build_mlp(width=256, depth=32)`, seeds 11 and 22. MC truth is fresh antithetic N(0,I), generated by this script. No Fly, network, pytest, tracked-file edits, public suites, or grader internals.",
        "",
        "## Split and closure",
        "",
        f"- Mixture starts by recursively splitting the input Gaussian into M components along the current largest `w * lambda_max(Sigma)` axis.",
        f"- Split choice: two equal-weight children, means offset `+/- {SPLIT_C:.2f} sigma` along the top eigenvector, child variance along that axis reduced to `{1.0 - SPLIT_C * SPLIT_C:.2f}` of the parent.",
        "- Each layer applies exact linear mean/covariance propagation and the nonzero-mean Gaussian ReLU moment closure ported from `estimator.py`, including the GL16 Price-identity bivariate covariance integral.",
        "",
        "## Bias vs M",
        "",
    ]
    pooled_rows = []
    for mlp in data["mlps"]:
        lines += [
            f"### seed {mlp['seed']}",
            "",
            "| M | final MSE | truth noise | net bias MSE | all-layer MSE | wall s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for run in sorted(mlp.get("runs", []), key=lambda r: r["M"]):
            met = run["metrics"]
            pooled_rows.append({"M": run["M"], "net_bias_final_mse": met["net_bias_final_mse"]})
            lines.append(
                f"| {run['M']} | {met['final_layer_mse']:.9e} | "
                f"{mlp['truth']['truth_noise_final_mse']:.9e} | "
                f"{met['net_bias_final_mse']:.9e} | {met['all_layer_mse']:.9e} | "
                f"{run['wall_time_s']:.1f} |"
            )
        fit = mlp.get("fit", {})
        if fit.get("alpha") is not None:
            lines += ["", f"Fit alpha: `{fit['alpha']:.3f}`", ""]
        adaptive_runs = sorted(mlp.get("adaptive_runs", []), key=lambda r: r["M"])
        if adaptive_runs:
            lines += [
                "",
                "Adaptive pre-ReLU split check:",
                "",
                "| M | final MSE | net bias MSE | all-layer MSE | wall s |",
                "|---:|---:|---:|---:|---:|",
            ]
            for run in adaptive_runs:
                met = run["metrics"]
                lines.append(
                    f"| {run['M']} | {met['final_layer_mse']:.9e} | "
                    f"{met['net_bias_final_mse']:.9e} | {met['all_layer_mse']:.9e} | "
                    f"{run['wall_time_s']:.1f} |"
                )
            lines.append("")
    pooled_fit = data.get("pooled_fit", {})
    lines += [
        "## Pooled fit and extrapolation",
        "",
        f"- Pooled alpha: `{pooled_fit.get('alpha'):.3f}`" if pooled_fit.get("alpha") is not None else "- Pooled alpha: unavailable",
    ]
    for target, row in data.get("pooled_extrapolation", {}).items():
        if row.get("M") is None:
            lines.append(f"- Target `{target}` net bias MSE: not reachable by extrapolation ({row.get('note')}).")
        else:
            lines.append(f"- Target `{target}` net bias MSE: M `{row['M']:.1f}`, raw FLOPs `{row['raw_flops']:.3e}`")
    lines += [
        "",
        "## Cost model",
        "",
        f"- Per component per layer: two dense 256^3-class matmuls plus GL16 bivariate covariance closure, estimated as `{raw_flops(1) / DEPTH:.3e}` raw FLOPs.",
        f"- Per component for all 32 layers: `{raw_flops(1):.3e}` raw FLOPs.",
        "- Cluster comparison points: floor cluster near `2.5e10` raw FLOPs; 47%-budget entry near `1.25e11` raw FLOPs.",
        "",
        "## Verdict",
        "",
        data.get("verdict", "Incomplete: still running or insufficient successful M values."),
        "",
        "## Recommended next action",
        "",
        data.get("recommended_next_action", "Complete the remaining M sweep, then decide whether adaptive split placement is warranted."),
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def summarize(data: dict[str, Any]) -> None:
    write_report(data)
    print(MD_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400_000)
    parser.add_argument("--batch-pairs", type=int, default=20_000)
    parser.add_argument("--max-m", type=int, default=16)
    parser.add_argument("--adaptive-check", action="store_true")
    args = parser.parse_args()

    data = load_results()
    for seed in SEEDS:
        mlp = next((m for m in data["mlps"] if m["seed"] == seed), None)
        weights = mlp_weights_np(seed)
        if mlp is None:
            print(f"truth seed={seed}", flush=True)
            truth = antithetic_truth(
                weights,
                n_samples=args.samples,
                seed=1000 + seed,
                batch_pairs=args.batch_pairs,
            )
            mlp = {"seed": seed, "truth": truth, "runs": []}
            data["mlps"].append(mlp)
            dump_json(data)
        else:
            truth = mlp["truth"]
        truth_mean = np.array(truth["mean"], dtype=np.float64)
        have = {run["M"] for run in mlp.get("runs", [])}
        for m in [x for x in MS if x <= args.max_m]:
            if m in have:
                continue
            print(f"propagate seed={seed} M={m}", flush=True)
            run = gaussian_sum_propagate(weights, m)
            metrics = mse_metrics(run["prediction"], truth_mean, truth["truth_noise_mse_by_layer"])
            mlp["runs"].append(
                {
                    "M": m,
                    "wall_time_s": run["wall_time_s"],
                    "split": run["split"],
                    "metrics": metrics,
                }
            )
            mlp["fit"] = fit_alpha(
                [
                    {"M": r["M"], "net_bias_final_mse": r["metrics"]["net_bias_final_mse"]}
                    for r in mlp["runs"]
                ]
            )
            dump_json(data)
            write_report(data)
        if args.adaptive_check and args.max_m >= 16:
            adaptive = mlp.setdefault("adaptive_runs", [])
            if not any(run["M"] == 16 for run in adaptive):
                print(f"adaptive propagate seed={seed} M=16", flush=True)
                run = gaussian_sum_progressive_prerelu(weights, 16)
                metrics = mse_metrics(run["prediction"], truth_mean, truth["truth_noise_mse_by_layer"])
                adaptive.append(
                    {
                        "M": 16,
                        "wall_time_s": run["wall_time_s"],
                        "split": run["split"],
                        "metrics": metrics,
                    }
                )
                dump_json(data)
                write_report(data)
    pooled_rows = [
        {"M": r["M"], "net_bias_final_mse": r["metrics"]["net_bias_final_mse"]}
        for mlp in data["mlps"]
        for r in mlp.get("runs", [])
    ]
    data["pooled_fit"] = fit_alpha(pooled_rows)
    data["pooled_extrapolation"] = extrapolate(data["pooled_fit"], (0.5e-6, 0.1e-6, 0.05e-6))
    if data["pooled_fit"].get("alpha") is None:
        data["verdict"] = "Incomplete: not enough positive-bias runs for a scaling fit."
    else:
        alpha = data["pooled_fit"]["alpha"]
        m16 = [
            r["metrics"]["net_bias_final_mse"]
            for mlp in data["mlps"]
            for r in mlp.get("runs", [])
            if r["M"] == 16
        ]
        if alpha >= 1.2 and m16 and float(np.mean(m16)) <= 3e-6:
            data["verdict"] = "MECHANISM PLAUSIBLE under the pre-registered gate."
            data["recommended_next_action"] = "Engineer/adapt the split policy and investigate cheaper covariance closures, because raw GL16 full-covariance cost is high."
        elif alpha < 0.8 or (m16 and float(np.mean(m16)) > 1e-5):
            data["verdict"] = "NEGATIVE: mechanism as formulated does not explain the cluster."
            data["recommended_next_action"] = "Treat input-space Gaussian splitting as failed; only a materially smarter adaptive split would be worth a separate gate."
        else:
            data["verdict"] = "BORDERLINE/INCONCLUSIVE under the pre-registered gate."
            data["recommended_next_action"] = "Run adaptive pre-ReLU splitting on the same truth rows before spending estimator-engineering effort."
    dump_json(data)
    summarize(data)


if __name__ == "__main__":
    main()
