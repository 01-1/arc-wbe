#!/usr/bin/env python3
"""Deep-collapse structure gate experiment.

Offline research-only probe.  Uses self-generated local_engine MLPs and
independent Monte Carlo streams; writes outputs beside this script.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_engine import build_mlp  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
WIDTH = 256
DEPTH = 32
SEEDS = (11, 22)
SPECTRUM_LAYERS = (2, 4, 8, 16, 24, 28, 31)
NS = (2_000, 5_000, 20_000)
RS = (1, 2, 4)
TRUTH_N = 400_000
TRUTH_CHUNK = 10_000
INDEP_MAX_N = 20_000


def np_weights(mlp) -> list[np.ndarray]:
    return [np.asarray(w, dtype=np.float32) for w in mlp.weights]


def forward_collect(x: np.ndarray, weights: list[np.ndarray], layers: set[int]) -> dict[int, np.ndarray]:
    y = x.astype(np.float32, copy=False)
    out = {}
    for idx, w in enumerate(weights, start=1):
        y = np.maximum(y @ w, 0.0).astype(np.float32, copy=False)
        if idx in layers:
            out[idx] = y.copy()
    return out


@dataclass
class RunningLayerStats:
    n: int
    sum_y: np.ndarray
    sum_yy: np.ndarray
    sum_normed: np.ndarray
    sum_normed_norm2: float
    sum_centered_normed: np.ndarray
    sum_centered_normed_norm2: float

    @classmethod
    def make(cls) -> "RunningLayerStats":
        z = np.zeros(WIDTH, dtype=np.float64)
        return cls(0, z.copy(), np.zeros((WIDTH, WIDTH), dtype=np.float64), z.copy(), 0.0, z.copy(), 0.0)

    def update(self, y: np.ndarray) -> None:
        yd = y.astype(np.float64, copy=False)
        self.n += yd.shape[0]
        self.sum_y += yd.sum(axis=0)
        self.sum_yy += yd.T @ yd

        norms = np.linalg.norm(yd, axis=1)
        ok = norms > 0
        if np.any(ok):
            u = yd[ok] / norms[ok, None]
            self.sum_normed += u.sum(axis=0)
            self.sum_normed_norm2 += float(np.sum(u * u))

        centered = yd - yd.mean(axis=1, keepdims=True)
        cnorms = np.linalg.norm(centered, axis=1)
        ok = cnorms > 0
        if np.any(ok):
            cu = centered[ok] / cnorms[ok, None]
            self.sum_centered_normed += cu.sum(axis=0)
            self.sum_centered_normed_norm2 += float(np.sum(cu * cu))

    def finish(self) -> dict:
        mean = self.sum_y / self.n
        cov = self.sum_yy / self.n - np.outer(mean, mean)
        cov = (cov + cov.T) * 0.5
        evals = np.linalg.eigvalsh(cov)[::-1]
        total = float(np.sum(np.maximum(evals, 0.0)))
        shares = {}
        for k in (1, 2, 4):
            shares[f"top{k}_share"] = float(np.sum(evals[:k]) / total) if total > 0 else float("nan")
        pr = float(total * total / np.sum(evals * evals)) if total > 0 else float("nan")

        denom = self.n * (self.n - 1)
        c_unc = float((np.dot(self.sum_normed, self.sum_normed) - self.sum_normed_norm2) / denom)
        c_ctr = float(
            (np.dot(self.sum_centered_normed, self.sum_centered_normed) - self.sum_centered_normed_norm2) / denom
        )
        return {
            **shares,
            "participation_rank": pr,
            "mean_pairwise_cosine_uncentered": c_unc,
            "mean_pairwise_corr_centered": c_ctr,
            "one_minus_uncentered_c_times_k2": None,
            "evals_top8": [float(x) for x in evals[:8]],
        }


def antithetic_truth(weights: list[np.ndarray], seed: int) -> tuple[dict[int, dict], np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    stats = {layer: RunningLayerStats.make() for layer in SPECTRUM_LAYERS}
    final_sum = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    layers = set(SPECTRUM_LAYERS) | {DEPTH}
    while done < TRUTH_N:
        n = min(TRUTH_CHUNK, TRUTH_N - done)
        half = (n + 1) // 2
        x0 = rng.standard_normal((half, WIDTH), dtype=np.float32)
        x = np.concatenate([x0, -x0], axis=0)[:n]
        got = forward_collect(x, weights, layers)
        for layer in SPECTRUM_LAYERS:
            stats[layer].update(got[layer])
        final_sum += got[DEPTH].astype(np.float64).sum(axis=0)
        done += n
    spectra = {layer: stats[layer].finish() for layer in SPECTRUM_LAYERS}
    for layer, row in spectra.items():
        c = row["mean_pairwise_cosine_uncentered"]
        row["one_minus_uncentered_c_times_k2"] = float((1.0 - c) * layer * layer)
    return spectra, final_sum / TRUTH_N


def independent_activations(weights: list[np.ndarray], seed: int, n: int = INDEP_MAX_N) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20_000 + seed)
    x = rng.standard_normal((n, WIDTH), dtype=np.float32)
    got = forward_collect(x, weights, {31, 32})
    return got[31].astype(np.float64), got[32].astype(np.float64)


def gaussian_relu_mean(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    sig = np.sqrt(np.maximum(var, 1e-18))
    alpha = mu / sig
    phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(alpha / math.sqrt(2.0)))
    return sig * phi + mu * Phi


def pca_scores(y: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y.mean(axis=0)
    yc = y - mean
    cov = (yc.T @ yc) / y.shape[0]
    evals, evecs = np.linalg.eigh((cov + cov.T) * 0.5)
    order = np.argsort(evals)[::-1]
    vecs = evecs[:, order[:r]]
    return yc @ vecs, mean, vecs


def quantile_bins(a: np.ndarray, bins_per_dim: int) -> tuple[np.ndarray, int]:
    codes = np.zeros(a.shape[0], dtype=np.int64)
    mult = 1
    used_bins = []
    for d in range(a.shape[1]):
        qs = np.linspace(0.0, 1.0, bins_per_dim + 1)
        edges = np.quantile(a[:, d], qs)
        edges[0] = -np.inf
        edges[-1] = np.inf
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        b = np.searchsorted(edges[1:-1], a[:, d], side="right")
        codes += mult * b
        mult *= bins_per_dim
        used_bins.append(int(np.unique(b).size))
    return codes, int(mult)


def latent_readout(y31: np.ndarray, y32: np.ndarray, weights: list[np.ndarray], truth_final: np.ndarray, n: int, r: int) -> dict:
    y = y31[:n]
    z = y @ weights[31].astype(np.float64)
    plain = y32[:n].mean(axis=0)
    scores, _, _ = pca_scores(y, r)

    if r == 1:
        bins_per_dim = min(96, max(12, int(round(n ** 0.50))))
    elif r == 2:
        bins_per_dim = min(24, max(8, int(round(n ** 0.25 * 2.2))))
    else:
        bins_per_dim = min(8, max(4, int(round(n ** 0.125 * 2.3))))

    codes, max_code = quantile_bins(scores, bins_per_dim)
    counts = np.bincount(codes, minlength=max_code).astype(np.float64)
    nonzero = counts > max(2 * r + 2, 5)
    pred_sum = np.zeros(WIDTH, dtype=np.float64)
    kept = 0
    fallback_var = z.var(axis=0)
    for code in np.flatnonzero(nonzero):
        idx = codes == code
        zz = z[idx]
        mu = zz.mean(axis=0)
        var = zz.var(axis=0) if zz.shape[0] > 1 else fallback_var
        pred_sum += counts[code] * gaussian_relu_mean(mu, var)
        kept += int(counts[code])

    if kept < n:
        idx = ~nonzero[codes]
        if np.any(idx):
            zz = z[idx]
            pred_sum += zz.shape[0] * gaussian_relu_mean(zz.mean(axis=0), zz.var(axis=0) if zz.shape[0] > 1 else fallback_var)
            kept += int(zz.shape[0])

    readout = pred_sum / kept
    plain_err = plain - truth_final
    readout_err = readout - truth_final
    plain_mse = float(np.mean(plain_err * plain_err))
    readout_mse = float(np.mean(readout_err * readout_err))
    return {
        "n": n,
        "r": r,
        "bins_per_dim": bins_per_dim,
        "nonempty_cells": int(np.count_nonzero(counts)),
        "modeled_cells": int(np.count_nonzero(nonzero)),
        "plain_mse": plain_mse,
        "readout_mse": readout_mse,
        "improvement_mse_factor": float(plain_mse / readout_mse) if readout_mse > 0 else float("inf"),
        "plain_rmse": float(math.sqrt(plain_mse)),
        "readout_rmse": float(math.sqrt(readout_mse)),
        "plain_bias_mean": float(np.mean(plain_err)),
        "readout_bias_mean": float(np.mean(readout_err)),
        "plain_max_abs": float(np.max(np.abs(plain_err))),
        "readout_max_abs": float(np.max(np.abs(readout_err))),
    }


def residual_diagnostics(y31: np.ndarray, weights: list[np.ndarray], n: int = 20_000, r: int = 2) -> dict:
    y = y31[:n]
    z = y @ weights[31].astype(np.float64)
    scores, _, _ = pca_scores(y, r)
    codes, max_code = quantile_bins(scores, 18)
    counts = np.bincount(codes, minlength=max_code)
    pooled = []
    for code in np.flatnonzero(counts > 30):
        idx = codes == code
        zz = z[idx]
        sd = zz.std(axis=0)
        ok = sd > 1e-12
        if np.any(ok):
            pooled.append(((zz[:, ok] - zz[:, ok].mean(axis=0)) / sd[ok]).ravel())
    if not pooled:
        return {}
    e = np.concatenate(pooled)
    e = e[np.isfinite(e)]
    return {
        "standardized_residual_n": int(e.size),
        "mean": float(e.mean()),
        "variance": float(e.var()),
        "skew": float(np.mean((e - e.mean()) ** 3) / max(e.std() ** 3, 1e-18)),
        "excess_kurtosis": float(np.mean((e - e.mean()) ** 4) / max(e.var() ** 2, 1e-18) - 3.0),
        "q001": float(np.quantile(e, 0.001)),
        "q999": float(np.quantile(e, 0.999)),
    }


def summarize(results: dict) -> str:
    lines = []
    lines.append("# Collapse Gate: Filament Latent + Conditional-Gaussian Readout (2026-07-06)")
    lines.append("")
    lines.append("Offline research-only run with `local_engine.build_mlp`, width 256, depth 32, seeds 11 and 22. Truth uses 400k antithetic samples; latent/readout fits use an independent sample stream.")
    lines.append("")
    lines.append("## Collapse Spectrum")
    lines.append("| seed | layer | top1 | top2 | top4 | PR rank | mean cosine c | (1-c)k^2 | centered corr |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed_row in results["seeds"]:
        seed = seed_row["seed"]
        for layer in SPECTRUM_LAYERS:
            row = seed_row["spectra"][str(layer)]
            lines.append(
                f"| {seed} | {layer} | {row['top1_share']:.4f} | {row['top2_share']:.4f} | "
                f"{row['top4_share']:.4f} | {row['participation_rank']:.2f} | "
                f"{row['mean_pairwise_cosine_uncentered']:.6f} | {row['one_minus_uncentered_c_times_k2']:.3f} | "
                f"{row['mean_pairwise_corr_centered']:.6f} |"
            )
    lines.append("")
    lines.append("## Latent Readout")
    lines.append("| seed | n | r | cells | readout MSE | plain MSE | MSE improvement | readout RMSE | plain RMSE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed_row in results["seeds"]:
        for row in seed_row["readout"]:
            lines.append(
                f"| {seed_row['seed']} | {row['n']} | {row['r']} | {row['nonempty_cells']} | "
                f"{row['readout_mse']:.3e} | {row['plain_mse']:.3e} | {row['improvement_mse_factor']:.2f}x | "
                f"{row['readout_rmse']:.3e} | {row['plain_rmse']:.3e} |"
            )
    lines.append("")
    lines.append("## Residual Gaussianity")
    lines.append("| seed | n standardized | mean | var | skew | excess kurt | q0.001 | q0.999 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed_row in results["seeds"]:
        d = seed_row["residual_diagnostics"]
        lines.append(
            f"| {seed_row['seed']} | {d.get('standardized_residual_n', 0)} | {d.get('mean', float('nan')):.3e} | "
            f"{d.get('variance', float('nan')):.3f} | {d.get('skew', float('nan')):.3f} | "
            f"{d.get('excess_kurtosis', float('nan')):.3f} | {d.get('q001', float('nan')):.3f} | {d.get('q999', float('nan')):.3f} |"
        )
    lines.append("")
    top2 = [row["spectra"]["31"]["top2_share"] for row in results["seeds"]]
    imp5000 = [r["improvement_mse_factor"] for s in results["seeds"] for r in s["readout"] if r["n"] == 5000 and r["r"] == 2]
    best5000 = [min(r["readout_mse"] for r in s["readout"] if r["n"] == 5000) for s in results["seeds"]]
    if min(top2) >= 0.60 and min(imp5000 or [0]) >= 10.0 and max(best5000 or [1]) <= 0.3e-6:
        verdict = "MECHANISM CONFIRMED as exploitable by the preregistered gate."
    elif min(top2) < 0.30:
        verdict = "Hypothesis dead by the preregistered no-collapse gate."
    elif max(imp5000 or [0]) < 3.0:
        verdict = "Collapse is present, but this conditional-Gaussian residual readout is too crude."
    else:
        verdict = "Mixed: collapse is present, but the latent readout does not clear the 10x/0.3e-6 exploitation gate."
    results["verdict"] = verdict
    lines.append("## Verdict")
    lines.append(verdict)
    lines.append("")
    lines.append("Recommended next action: if pursuing this mechanism, replace the hard-cell conditional model with cross-fitted local polynomial or spline conditionals, then test against the same independent truth protocol before touching `estimator.py`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.time()
    results = {
        "created_at": "2026-07-06",
        "truth_n_antithetic": TRUTH_N,
        "independent_max_n": INDEP_MAX_N,
        "width": WIDTH,
        "depth": DEPTH,
        "seeds": [],
    }
    for seed in SEEDS:
        print(f"[seed {seed}] building MLP")
        mlp = build_mlp(WIDTH, DEPTH, seed=seed)
        weights = np_weights(mlp)
        print(f"[seed {seed}] 400k antithetic truth + spectra")
        spectra, truth_final = antithetic_truth(weights, seed)
        print(f"[seed {seed}] independent layer-31 sample")
        y31, y32 = independent_activations(weights, seed)
        readouts = []
        for n in NS:
            for r in RS:
                print(f"[seed {seed}] latent readout n={n} r={r}")
                readouts.append(latent_readout(y31, y32, weights, truth_final, n, r))
        diag = residual_diagnostics(y31, weights)
        results["seeds"].append(
            {
                "seed": seed,
                "spectra": {str(k): v for k, v in spectra.items()},
                "readout": readouts,
                "residual_diagnostics": diag,
            }
        )
    results["elapsed_s"] = time.time() - t0
    md = summarize(results)
    json_path = OUT_DIR / "collapse_gate_20260706_results.json"
    md_path = OUT_DIR / "collapse_gate_20260706.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
