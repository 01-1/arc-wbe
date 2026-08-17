#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


WIDTH = 256
DEPTH = 32
BLOCKS = 16
ROOT = Path(__file__).resolve().parent


def hadamard(n: int = WIDTH) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


H = hadamard()


def relu_kernel_from_cov(cov: np.ndarray, qx: np.ndarray | float, qy: np.ndarray | float) -> np.ndarray:
    denom = np.sqrt(np.asarray(qx) * np.asarray(qy))
    c = np.clip(cov / denom, -1.0, 1.0)
    theta = np.arccos(c)
    return denom * (np.sin(theta) + (math.pi - theta) * c) / math.pi


def nngp_from_normalized_dot(dot: np.ndarray, qx: np.ndarray | float = 1.0, qy: np.ndarray | float = 1.0) -> np.ndarray:
    k = dot.astype(np.float64, copy=True)
    qxl = np.asarray(qx, dtype=np.float64)
    qyl = np.asarray(qy, dtype=np.float64)
    for _ in range(DEPTH):
        k = relu_kernel_from_cov(k, qxl, qyl)
        qxl = qxl
        qyl = qyl
    return k


def nngp_pair(x: np.ndarray, y: np.ndarray) -> float:
    return float(nngp_from_normalized_dot(np.array(np.dot(x, y) / WIDTH))[()])


def build_mlp(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float64) for _ in range(DEPTH)]


def mlp_eval(weights: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64)
    for w in weights:
        y = np.maximum(y @ w, 0.0)
    return y


def validate_kernel(n_mlps: int) -> dict:
    rng = np.random.default_rng(12345)
    base = rng.standard_normal((6, WIDTH))
    base[0] = H[0]
    base[1] = H[1]
    base[2] = -H[0]
    pairs = [(0, 0), (0, 1), (0, 2), (3, 4), (3, 5), (4, 5)]
    vals = {p: [] for p in pairs}
    for seed in range(10_000, 10_000 + n_mlps):
        weights = build_mlp(seed)
        outs = [mlp_eval(weights, x) for x in base]
        for p in pairs:
            vals[p].append(float(np.mean(outs[p[0]] * outs[p[1]])))
    rows = []
    for p in pairs:
        empirical = float(np.mean(vals[p]))
        pred = nngp_pair(base[p[0]], base[p[1]])
        rows.append({
            "pair": list(p),
            "input_dot_over_width": float(np.dot(base[p[0]], base[p[1]]) / WIDTH),
            "kernel": pred,
            "empirical_mean_output_product": empirical,
            "rel_error": float((empirical - pred) / max(abs(pred), 1e-12)),
        })
    return {"n_mlps": n_mlps, "pairs": rows}


def design_current(seed: int = 20260705) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(BLOCKS):
        d = 2.0 * rng.integers(0, 2, size=WIDTH) - 1.0
        b = H * d[None, :]
        blocks.extend([b, -b])
    return np.concatenate(blocks, axis=0)


def design_iid_gaussian(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, WIDTH))
    return x / np.linalg.norm(x, axis=1, keepdims=True) * math.sqrt(WIDTH)


def design_multiradius_sign(seed: int = 20260706) -> np.ndarray:
    x = design_current(seed)
    radii = np.array([math.sqrt(WIDTH - 32), math.sqrt(WIDTH - 8), math.sqrt(WIDTH + 8), math.sqrt(WIDTH + 32)])
    scale = np.resize(radii / math.sqrt(WIDTH), x.shape[0])
    return x * scale[:, None]


def design_more_blocks(seed: int = 20260707) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(32):
        d = 2.0 * rng.integers(0, 2, size=WIDTH) - 1.0
        rows = H[::2] * d[None, :]
        blocks.extend([rows, -rows])
    return np.concatenate(blocks, axis=0)


def kernel_block(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dot = (x @ y.T) / WIDTH
    qx = np.sum(x * x, axis=1)[:, None] / WIDTH
    qy = np.sum(y * y, axis=1)[None, :] / WIDTH
    return nngp_from_normalized_dot(dot, qx, qy)


def estimate_kmu_m(x: np.ndarray, mc: int, seed: int) -> tuple[np.ndarray, float, dict]:
    rng = np.random.default_rng(seed)
    qx_all = np.sum(x * x, axis=1) / WIDTH
    unique_q = np.unique(np.round(qx_all, 12))
    z0 = rng.standard_normal(mc)
    chi_perp = rng.chisquare(WIDTH - 1, size=mc)
    qy = (z0 * z0 + chi_perp) / WIDTH
    kmu_by_q = {}
    for qx in unique_q:
        cov = math.sqrt(float(qx) / WIDTH) * z0
        kmu_by_q[float(qx)] = float(nngp_from_normalized_dot(cov, float(qx), qy).mean())
    kmu = np.array([kmu_by_q[float(q)] for q in np.round(qx_all, 12)], dtype=np.float64)

    m_sum = 0.0
    m_count = 0
    for start in range(0, mc, 20_000):
        n = min(20_000, mc - start)
        a = rng.standard_normal((n, WIDTH))
        b = rng.standard_normal((n, WIDTH))
        qa = np.sum(a * a, axis=1) / WIDTH
        qb = np.sum(b * b, axis=1) / WIDTH
        cov = np.sum(a * b, axis=1) / WIDTH
        m_sum += float(nngp_from_normalized_dot(cov, qa, qb).sum())
        m_count += n
    m = m_sum / m_count
    return kmu, m, {
        "mc": mc,
        "unique_q": [float(q) for q in unique_q],
        "kmu_by_q": kmu_by_q,
        "kmu_mean": float(kmu.mean()),
        "kmu_std": float(kmu.std()),
        "m": float(m),
    }


def dense_kernel(x: np.ndarray) -> np.ndarray:
    dot = (x @ x.T) / WIDTH
    q = np.sum(x * x, axis=1) / WIDTH
    return nngp_from_normalized_dot(dot, q[:, None], q[None, :])


def equal_wkw_and_row_stats(x: np.ndarray) -> tuple[float, dict]:
    k = dense_kernel(x)
    rows = k.sum(axis=1)
    n = x.shape[0]
    wkw = float(k.sum() / (n * n))
    return {
        "wkw": wkw,
        "mean": float(rows.mean()),
        "std": float(rows.std()),
        "rel_std": float(rows.std() / max(abs(rows.mean()), 1e-300)),
        "min": float(rows.min()),
        "max": float(rows.max()),
        "rel_range": float((rows.max() - rows.min()) / max(abs(rows.mean()), 1e-300)),
    }, k


def subset_optimal_error(x: np.ndarray, kmu: np.ndarray, m: float, n_sub: int = 1024) -> dict:
    t0 = time.time()
    xs = x[:n_sub]
    ks = kmu[:n_sub]
    k = dense_kernel(xs)
    jitter = 1e-9 * float(np.mean(np.diag(k)))
    sol = np.linalg.solve(k + jitter * np.eye(n_sub), ks)
    equal_wkw = float(k.mean())
    equal_err = float(m - 2.0 * float(ks.mean()) + equal_wkw)
    opt_err = float(m - float(ks @ sol))
    return {
        "n_sub": n_sub,
        "equal_err2": equal_err,
        "optimal_err2": opt_err,
        "ratio_equal_over_optimal": float(equal_err / opt_err),
        "weight_sum": float(sol.sum()),
        "weight_min": float(sol.min()),
        "weight_max": float(sol.max()),
        "seconds": time.time() - t0,
    }


def score_design(name: str, x: np.ndarray, kmu_mc: int, do_opt: bool) -> dict:
    t0 = time.time()
    kmu, m, integ = estimate_kmu_m(x, kmu_mc, 777)
    row_stats, _ = equal_wkw_and_row_stats(x)
    wkw = row_stats.pop("wkw")
    equal_err = float(m - 2.0 * float(kmu.mean()) + wkw)
    out = {
        "name": name,
        "n": int(x.shape[0]),
        "integrals": integ,
        "equal": {"err2": equal_err, "wKw": float(wkw)},
        "row_sum_stats": row_stats,
        "seconds": time.time() - t0,
    }
    if do_opt:
        out["subset_optimal"] = subset_optimal_error(x, kmu, m)
    return out


def write_report(results: dict) -> None:
    cur = results["scores"][0]["equal"]["err2"]
    lines = [
        "# NNGP-Kernel Cubature-Optimality Pre-Test",
        "",
        "Scope: offline analysis only. No estimator edits, no tracked-file edits, no Fly, no network, no pytest.",
        "",
        "## Kernel Validation",
        "",
        f"Validated against {results['validation']['n_mlps']} self-generated width-256/depth-32 He MLPs.",
        "",
        "| pair | dot/width | NNGP | empirical output product | rel error |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results["validation"]["pairs"]:
        lines.append(f"| {r['pair']} | {r['input_dot_over_width']:.4f} | {r['kernel']:.6g} | {r['empirical_mean_output_product']:.6g} | {r['rel_error']:.2%} |")
    lines += ["", "## Error Table", "", "| design | weighting | err^2 | ratio vs current equal | notes |", "|---|---:|---:|---:|---|"]
    for s in results["scores"]:
        lines.append(f"| {s['name']} | equal | {s['equal']['err2']:.8g} | {s['equal']['err2'] / cur:.4f} | wKw {s['equal']['wKw']:.8g} |")
        if "subset_optimal" in s:
            opt = s["subset_optimal"]
            lines.append(f"| {s['name']} first {opt['n_sub']} | BQ optimal | {opt['optimal_err2']:.8g} | n/a | subset equal/optimal {opt['ratio_equal_over_optimal']:.4f} |")
    best = min(s["equal"]["err2"] for s in results["scores"])
    win = cur / best
    verdict = "ALIVE" if win >= 1.3 else "DEAD"
    lines += [
        "",
        "## Gate Verdict",
        "",
        f"Best full-8192 equal-weight point-set improvement over current equal weights: `{win:.3f}x`.",
        f"Candidate is **{verdict}** under the pre-registered `>= 1.3x` gate.",
        f"Current-design row-sum relative std: `{results['scores'][0]['row_sum_stats']['rel_std']:.3e}`; relative range `{results['scores'][0]['row_sum_stats']['rel_range']:.3e}`. With constant `k_mu` on equal-radius sign points, this is the direct BQ equal-weight optimality diagnostic.",
        "",
        "Caveat: this is the raw NNGP prior/design proxy. The production estimator also applies first-layer recoloring and variance matching, so this bounds cubature-design headroom rather than proving production MSE exactly.",
    ]
    (ROOT / "nngp_design_pretest_20260705.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmu-mc", type=int, default=4096)
    ap.add_argument("--validation-mlps", type=int, default=24)
    args = ap.parse_args()
    results = {
        "config": {"width": WIDTH, "depth": DEPTH, "blocks": BLOCKS, "kmu_mc": args.kmu_mc},
        "validation": validate_kernel(args.validation_mlps),
        "scores": [],
    }
    designs = [
        ("current_hadamard_antithetic", design_current(20260705), True),
        ("iid_gaussian_sphere", design_iid_gaussian(BLOCKS * WIDTH * 2, 20260705), False),
        ("multiradius_sign", design_multiradius_sign(20260706), False),
        ("more_smaller_orthogonal_blocks", design_more_blocks(20260707), False),
    ]
    for name, x, opt in designs:
        print(f"scoring {name} n={x.shape[0]} opt={opt}", flush=True)
        results["scores"].append(score_design(name, x, args.kmu_mc, opt))
    best = min(s["equal"]["err2"] for s in results["scores"])
    results["gate"] = {
        "current_equal_err2": results["scores"][0]["equal"]["err2"],
        "best_err2": best,
        "best_improvement": results["scores"][0]["equal"]["err2"] / best,
        "threshold": 1.3,
        "verdict": "ALIVE" if results["scores"][0]["equal"]["err2"] / best >= 1.3 else "DEAD",
    }
    (ROOT / "nngp_design_pretest_results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results)
    print(json.dumps(results["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
