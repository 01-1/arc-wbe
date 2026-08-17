#!/usr/bin/env python3
"""Aggregate the Fly JSONL for the block predictability gate."""

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
        if obj.get("ok") is True and "block_rows" in obj:
            rows.append(obj)
    return rows


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    z = (x - mu) / sigma
    design = np.column_stack([np.ones(z.shape[0]), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta, mu, sigma


def _ridge_predict(x: np.ndarray, beta: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (x - mu) / sigma
    return np.column_stack([np.ones(z.shape[0]), z]) @ beta


def _weighted_mse(block_errors: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    equal = np.mean(block_errors, axis=0)
    equal_mse = float(np.mean(equal * equal))
    risk = np.maximum(pred, np.quantile(pred, 0.1))
    weights = 1.0 / np.maximum(risk, 1e-30)
    weights = weights / weights.sum()
    weighted = weights @ block_errors
    return equal_mse, float(np.mean(weighted * weighted))


def aggregate(rows: list[dict]) -> dict[str, object]:
    mlp_payloads = []
    for row in rows:
        block_rows = row["block_rows"]
        x = np.asarray([b["features"] for b in block_rows], dtype=np.float64)
        y = np.asarray([b["sqerr"] for b in block_rows], dtype=np.float64)
        dots = np.asarray([b["dot_equal_error"] for b in block_rows], dtype=np.float64)
        reps = np.asarray([b["rep"] for b in block_rows], dtype=np.int64)
        blocks = np.asarray([b["block"] for b in block_rows], dtype=np.int64)
        mlp_payloads.append(
            {
                "mlp_index": int(row["mlp_index"]),
                "features": x,
                "sqerr": y,
                "dot_equal_error": dots,
                "rep": reps,
                "block": blocks,
                "rep_mse": np.asarray(row["rep_mse"], dtype=np.float64),
            }
        )

    alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
    per_mlp = []
    all_true = []
    all_pred = []
    paired_reductions = []
    for holdout in mlp_payloads:
        train = [item for item in mlp_payloads if item is not holdout]
        train_x = np.concatenate([item["features"] for item in train], axis=0)
        train_y = np.log(np.concatenate([item["sqerr"] for item in train], axis=0) + 1e-30)
        best_alpha = alphas[0]
        best_loss = float("inf")
        if len(train) >= 2:
            for alpha in alphas:
                losses = []
                for idx, valid in enumerate(train):
                    inner = [item for j, item in enumerate(train) if j != idx]
                    beta, mu, sigma = _ridge_fit(
                        np.concatenate([item["features"] for item in inner], axis=0),
                        np.log(np.concatenate([item["sqerr"] for item in inner], axis=0) + 1e-30),
                        alpha,
                    )
                    pred = _ridge_predict(valid["features"], beta, mu, sigma)
                    losses.append(float(np.mean((pred - np.log(valid["sqerr"] + 1e-30)) ** 2)))
                loss = float(np.mean(losses))
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
        else:
            best_alpha = 10.0
        beta, mu, sigma = _ridge_fit(train_x, train_y, best_alpha)
        pred_log = _ridge_predict(holdout["features"], beta, mu, sigma)
        pred = np.exp(pred_log)
        all_true.append(holdout["sqerr"])
        all_pred.append(pred)
        rep_equal = []
        rep_weighted = []
        for rep in sorted(set(holdout["rep"].tolist())):
            mask = holdout["rep"] == rep
            order = np.argsort(holdout["block"][mask])
            block_sq = holdout["sqerr"][mask][order]
            block_dot = holdout["dot_equal_error"][mask][order]
            # Recover a conservative vector-error proxy: use block-wise scalar
            # covariance with equal error for pairing, and MSE labels for weights.
            equal_mse = float(np.mean(holdout["rep_mse"][rep]))
            risk = np.maximum(pred[mask][order], np.quantile(pred[mask][order], 0.1))
            weights = 1.0 / np.maximum(risk, 1e-30)
            weights = weights / weights.sum()
            # Scalar approximation of weighted MSE from block MSE and covariance
            # to the equal error; exact vector errors are intentionally not stored.
            weighted_mse = float(np.sum(weights * block_dot))
            weighted_mse = max(weighted_mse, 0.0)
            rep_equal.append(equal_mse)
            rep_weighted.append(weighted_mse)
            high = int(np.argmax(pred[mask][order]))
            low = int(np.argmin(pred[mask][order]))
            paired_base = 0.5 * (block_sq[high] + block_sq[low])
            paired_mix = 0.25 * (block_sq[high] + block_sq[low] + 2.0 * np.sqrt(block_sq[high] * block_sq[low]))
            if paired_base > 0.0:
                paired_reductions.append(1.0 - paired_mix / paired_base)
        equal_mean = float(np.mean(rep_equal))
        weighted_mean = float(np.mean(rep_weighted))
        ratio = equal_mean / weighted_mean if weighted_mean > 0.0 else 0.0
        per_mlp.append(
            {
                "mlp_index": holdout["mlp_index"],
                "alpha": best_alpha,
                "equal_mse": equal_mean,
                "weighted_mse_proxy": weighted_mean,
                "variance_ratio_proxy": ratio,
            }
        )

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    corr = float(np.corrcoef(np.log(true + 1e-30), np.log(pred + 1e-30))[0, 1])
    ratios = np.asarray([item["variance_ratio_proxy"] for item in per_mlp], dtype=np.float64)
    pair = np.asarray(paired_reductions, dtype=np.float64)
    return {
        "n_mlps": len(rows),
        "n_block_rows": int(sum(len(row["block_rows"]) for row in rows)),
        "log_sqerr_pred_corr": corr,
        "variance_ratio_mean": float(np.mean(ratios)),
        "variance_ratio_median": float(np.median(ratios)),
        "variance_ratio_q10": float(np.quantile(ratios, 0.10)),
        "variance_ratio_q90": float(np.quantile(ratios, 0.90)),
        "paired_reduction_mean": float(np.mean(pair)) if pair.size else 0.0,
        "paired_reduction_median": float(np.median(pair)) if pair.size else 0.0,
        "paired_reduction_q10": float(np.quantile(pair, 0.10)) if pair.size else 0.0,
        "pass_weighting": bool(
            np.mean(ratios) >= 1.30 and np.median(ratios) >= 1.20 and np.quantile(ratios, 0.10) >= 0.95
        ),
        "pass_pairing": bool(
            pair.size
            and np.mean(pair) >= 0.20
            and np.median(pair) >= 0.10
            and np.quantile(pair, 0.10) >= -0.05
        ),
        "per_mlp": per_mlp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(_read_rows(args.jsonl))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
