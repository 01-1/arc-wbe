"""Fit simple CPU/GPU timing predictors for the local flopscope GPU bridge."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _collect_rows(repeats: int) -> list[dict[str, str]]:
    cmd = [
        sys.executable,
        "scripts/benchmark_flopscope_gpu_ops.py",
        "--repeats",
        str(repeats),
        "--threshold",
        "0",
        "--intensity",
        "0",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [
        line
        for line in proc.stdout.splitlines()
        if line and not line.startswith("gpu_status")
    ]
    return list(csv.DictReader(StringIO("\n".join(lines))))


def _family(name: str) -> str:
    return name.rsplit("_", 1)[0]


def _linear_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    families = sorted({_family(row["name"]) for row in rows})
    names = [
        "bias",
        "input_gb",
        "flops_g",
        *[f"family={family}" for family in families],
        *[f"input_gb:family={family}" for family in families],
        *[f"flops_g:family={family}" for family in families],
    ]
    x = []
    for row in rows:
        input_bytes = float(row["input_bytes"])
        flops = float(row["est_flops"])
        family = _family(row["name"])
        input_gb = input_bytes / 1e9
        flops_g = flops / 1e9
        family_bits = [1.0 if family == candidate else 0.0 for candidate in families]
        x.append(
            [
                1.0,
                input_gb,
                flops_g,
                *family_bits,
                *[input_gb * bit for bit in family_bits],
                *[flops_g * bit for bit in family_bits],
            ]
        )
    return np.asarray(x, dtype=float), names


def _per_family_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    families = sorted({_family(row["name"]) for row in rows})
    names = [
        "bias",
        *[f"family={family}" for family in families],
        *[f"input_gb:family={family}" for family in families],
        *[f"flops_g:family={family}" for family in families],
    ]
    x = []
    for row in rows:
        input_gb = float(row["input_bytes"]) / 1e9
        flops_g = float(row["est_flops"]) / 1e9
        family = _family(row["name"])
        family_bits = [1.0 if family == candidate else 0.0 for candidate in families]
        x.append(
            [
                1.0,
                *family_bits,
                *[input_gb * bit for bit in family_bits],
                *[flops_g * bit for bit in family_bits],
            ]
        )
    return np.asarray(x, dtype=float), names


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
    }


def _standardized_importance(
    x: np.ndarray, beta: np.ndarray, feature_names: list[str]
) -> list[tuple[str, float]]:
    std = x.std(axis=0)
    scores = np.abs(beta) * std
    return sorted(zip(feature_names, scores, strict=True), key=lambda item: item[1], reverse=True)


def _feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "input_gb": [],
        "flops_g": [],
        "family_intercept": [],
        "input_gb_by_family": [],
        "flops_g_by_family": [],
    }
    for i, name in enumerate(feature_names):
        if name == "input_gb":
            groups["input_gb"].append(i)
        elif name == "flops_g":
            groups["flops_g"].append(i)
        elif name.startswith("family="):
            groups["family_intercept"].append(i)
        elif name.startswith("input_gb:family="):
            groups["input_gb_by_family"].append(i)
        elif name.startswith("flops_g:family="):
            groups["flops_g_by_family"].append(i)
    return groups


def _ablation_importance(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    baseline_rmse: float,
    alpha: float = 1e-10,
) -> list[tuple[str, float, float]]:
    groups = _feature_groups(feature_names)
    rows = []
    all_idx = np.arange(x.shape[1])
    for group, idx in groups.items():
        if not idx:
            continue
        keep = np.array([i for i in all_idx if i not in idx])
        beta = _ridge_fit(x[:, keep], y, alpha=alpha)
        pred = np.maximum(x[:, keep] @ beta, 0.0)
        rmse = _metrics(y, pred)["rmse"]
        rows.append((group, rmse - baseline_rmse, rmse))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rows = _collect_rows(args.repeats)
    x, feature_names = _per_family_features(rows)
    y_cpu = np.asarray([float(row["cpu_s"]) for row in rows])
    y_gpu = np.asarray([float(row["gpu_s"]) for row in rows])

    beta_cpu = _ridge_fit(x, y_cpu, alpha=1e-10)
    beta_gpu = _ridge_fit(x, y_gpu, alpha=1e-10)
    pred_cpu = np.maximum(x @ beta_cpu, 0.0)
    pred_gpu = np.maximum(x @ beta_gpu, 0.0)
    true_cpu = y_cpu
    true_gpu = y_gpu

    print("cpu_metrics", _metrics(true_cpu, pred_cpu))
    print("gpu_metrics", _metrics(true_gpu, pred_gpu))
    print("\nlinear model")
    print("time_seconds = bias + input_gb terms + flops_g terms + family terms")
    print("\ncoefficients_cpu")
    for name, value in zip(feature_names, beta_cpu, strict=True):
        print(f"{name},{value:.8g}")
    print("\ncoefficients_gpu")
    for name, value in zip(feature_names, beta_gpu, strict=True):
        print(f"{name},{value:.8g}")

    print("\nstandardized_importance_cpu")
    for name, score in _standardized_importance(x, beta_cpu, feature_names):
        print(f"{name},{score:.8g}")
    print("\nstandardized_importance_gpu")
    for name, score in _standardized_importance(x, beta_gpu, feature_names):
        print(f"{name},{score:.8g}")

    cpu_rmse = _metrics(true_cpu, pred_cpu)["rmse"]
    gpu_rmse = _metrics(true_gpu, pred_gpu)["rmse"]
    print("\nablation_importance_cpu")
    for name, delta_rmse, rmse in _ablation_importance(x, true_cpu, feature_names, cpu_rmse):
        print(f"{name},delta_rmse={delta_rmse:.8g},rmse={rmse:.8g}")
    print("\nablation_importance_gpu")
    for name, delta_rmse, rmse in _ablation_importance(x, true_gpu, feature_names, gpu_rmse):
        print(f"{name},delta_rmse={delta_rmse:.8g},rmse={rmse:.8g}")

    print("\nname,cpu_s,pred_cpu_s,gpu_s,pred_gpu_s,true_speedup,pred_speedup,decision_ok")
    correct = 0
    for row, cpu, pcpu, gpu, pgpu in zip(
        rows, true_cpu, pred_cpu, true_gpu, pred_gpu, strict=True
    ):
        true_gpu_wins = gpu < cpu
        pred_gpu_wins = pgpu < pcpu
        correct += int(true_gpu_wins == pred_gpu_wins)
        print(
            f"{row['name']},{cpu:.8f},{pcpu:.8f},{gpu:.8f},{pgpu:.8f},"
            f"{cpu / gpu:.6f},{pcpu / pgpu:.6f},{true_gpu_wins == pred_gpu_wins}"
        )
    print(f"\ndecision_accuracy,{correct}/{len(rows)},{correct / len(rows):.3f}")


if __name__ == "__main__":
    main()
