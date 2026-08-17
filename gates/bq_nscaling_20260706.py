#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


WIDTH = 256
DEPTH = 32
SEED = 20260705
ROOT = Path(__file__).resolve().parent
PRIOR_RESULTS = ROOT / "nngp_design_pretest_results.json"
OUT_JSON = ROOT / "bq_nscaling_20260706_results.json"
OUT_MD = ROOT / "bq_nscaling_20260706.md"

N_VALUES = [2048, 4096, 8192, 16384, 32768]
SPECTRUM_N = 8192
TOP_EIGENVALUES = 500

# External anchors requested in the task.
GRADER_ANCHOR_ERR2 = 2.0e-6
FLY_NET_ANCHOR_ERR2 = 2.36e-6
ENTRY1_ERR2_BUDGET = 1.0e-7
ENTRY1_DENSE_EVALS = 30_720


def hadamard(n: int = WIDTH) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


H = hadamard()


def relu_kernel_from_cov(
    cov: np.ndarray,
    qx: np.ndarray | float = 1.0,
    qy: np.ndarray | float = 1.0,
) -> np.ndarray:
    denom = np.sqrt(np.asarray(qx) * np.asarray(qy))
    c = np.clip(cov / denom, -1.0, 1.0)
    theta = np.arccos(c)
    return denom * (np.sin(theta) + (math.pi - theta) * c) / math.pi


def nngp_from_normalized_dot(
    dot: np.ndarray,
    qx: np.ndarray | float = 1.0,
    qy: np.ndarray | float = 1.0,
) -> np.ndarray:
    k = dot.astype(np.float64, copy=True)
    qxl = np.asarray(qx, dtype=np.float64)
    qyl = np.asarray(qy, dtype=np.float64)
    for _ in range(DEPTH):
        k = relu_kernel_from_cov(k, qxl, qyl)
    return k


def load_prior_integrals() -> dict:
    prior = json.loads(PRIOR_RESULTS.read_text())
    current = next(s for s in prior["scores"] if s["name"] == "current_hadamard_antithetic")
    integ = current["integrals"]
    kmu = float(integ["kmu_by_q"]["1.0"])
    m = float(integ["m"])
    return {
        "source": str(PRIOR_RESULTS.relative_to(ROOT.parent.parent)),
        "mc": int(integ["mc"]),
        "kmu_q1": kmu,
        "m": m,
        "constant_m_minus_2kmu": m - 2.0 * kmu,
        "prior_8192_equal_err2": float(current["equal"]["err2"]),
        "prior_8192_wKw": float(current["equal"]["wKw"]),
    }


def make_sign_blocks(n_blocks: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.0 * rng.integers(0, 2, size=(n_blocks, WIDTH)) - 1.0


def block_cross_spectra(signs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact block-pair mean plus FWHT spectra for even/odd sign modes.

    A design block consists of H*d and -(H*d). For blocks a,b, the base row dot
    products are the Walsh transform of d_a*d_b divided by WIDTH. The antithetic
    sign dimension splits exactly into even K(c)+K(-c) and odd K(c)-K(-c) modes.
    """
    b = signs.shape[0]
    pair_mean = np.empty((b, b), dtype=np.float64)
    even_hat = np.empty((WIDTH, b, b), dtype=np.float64)
    odd_hat = np.empty((WIDTH, b, b), dtype=np.float64)

    for a in range(b):
        for c in range(a, b):
            projected = (H @ (signs[a] * signs[c])) / WIDTH
            k_plus = nngp_from_normalized_dot(projected)
            k_minus = nngp_from_normalized_dot(-projected)
            even = k_plus + k_minus
            odd = k_plus - k_minus
            mean_ac = float(even.mean() / 2.0)
            ehat = H @ even
            ohat = H @ odd

            pair_mean[a, c] = pair_mean[c, a] = mean_ac
            even_hat[:, a, c] = even_hat[:, c, a] = ehat
            odd_hat[:, a, c] = odd_hat[:, c, a] = ohat

    return pair_mean, even_hat, odd_hat


def equal_and_optimal_rows(pair_mean: np.ndarray, integrals: dict) -> list[dict]:
    rows = []
    kmu = integrals["kmu_q1"]
    m = integrals["m"]
    cst = integrals["constant_m_minus_2kmu"]
    for n in N_VALUES:
        b = n // (2 * WIDTH)
        mb = pair_mean[:b, :b]
        wkw = float(mb.mean())
        equal_err2 = float(cst + wkw)

        t0 = time.time()
        rhs = np.full(b, kmu, dtype=np.float64)
        u = np.linalg.solve(mb, rhs)
        residual = mb @ u - rhs
        total_weight = float(u.sum())
        opt_err2 = float(m - kmu * total_weight)
        opt_seconds = time.time() - t0
        eig = np.linalg.eigvalsh(mb)

        rows.append(
            {
                "n": n,
                "blocks": b,
                "pair_mean_method": "exact_antithetic_hadamard_block_mean",
                "pair_mean_se": 0.0,
                "wKw_pair_mean": wkw,
                "equal_err2": equal_err2,
                "optimal_err2": opt_err2,
                "equal_over_optimal": float(equal_err2 / opt_err2),
                "optimal_weight_sum": total_weight,
                "optimal_block_weight_min": float(u.min()),
                "optimal_block_weight_max": float(u.max()),
                "optimal_solve_resid_linf": float(np.max(np.abs(residual))),
                "compressed_condition": float(eig[-1] / eig[0]),
                "optimal_solve_seconds": opt_seconds,
            }
        )
    return rows


def spectrum_from_block_hats(
    even_hat: np.ndarray,
    odd_hat: np.ndarray,
    n_blocks: int,
) -> dict:
    vals = []
    t0 = time.time()
    for omega in range(WIDTH):
        vals.extend(np.linalg.eigvalsh(even_hat[omega, :n_blocks, :n_blocks]).tolist())
        vals.extend(np.linalg.eigvalsh(odd_hat[omega, :n_blocks, :n_blocks]).tolist())
    vals = np.array(vals, dtype=np.float64) / (2 * WIDTH * n_blocks)
    vals.sort()
    vals = vals[::-1]

    positive = vals[vals > 1e-14]
    ranks = np.arange(1, min(TOP_EIGENVALUES, len(positive)) + 1, dtype=np.float64)
    top = positive[: TOP_EIGENVALUES]

    def fit_beta(start_rank: int, end_rank: int) -> dict:
        mask = (ranks >= start_rank) & (ranks <= min(end_rank, len(top)))
        x = np.log(ranks[mask])
        y = np.log(top[mask])
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "rank_start": start_rank,
            "rank_end": int(min(end_rank, len(top))),
            "beta": float(-slope),
            "intercept": float(intercept),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        }

    fits = {
        "ranks_2_500": fit_beta(2, TOP_EIGENVALUES),
        "ranks_20_500": fit_beta(20, TOP_EIGENVALUES),
        "ranks_50_500": fit_beta(50, TOP_EIGENVALUES),
    }
    return {
        "n": 2 * WIDTH * n_blocks,
        "blocks": n_blocks,
        "method": "exact_walsh_antithetic_block_diagonalization",
        "count": int(vals.size),
        "positive_count_gt_1e-14": int(positive.size),
        "sum_eigenvalues": float(vals.sum()),
        "top_500": [float(x) for x in vals[:TOP_EIGENVALUES]],
        "top_20": [float(x) for x in vals[:20]],
        "tail_fits": fits,
        "seconds": time.time() - t0,
    }


def calibration(rows: list[dict], spectrum: dict) -> dict:
    by_n = {r["n"]: r for r in rows}
    model_8192_equal = by_n[8192]["equal_err2"]

    # Interpolate the exact design-family model to the entry's approximately
    # 30k dense forward evaluations using a power law between 16384 and 32768.
    lo = by_n[16384]["optimal_err2"]
    hi = by_n[32768]["optimal_err2"]
    beta_n = math.log(lo / hi) / math.log(32768 / 16384)
    pred_30720 = lo * (ENTRY1_DENSE_EVALS / 16384) ** (-beta_n)

    anchors = {}
    for name, observed in [
        ("grader", GRADER_ANCHOR_ERR2),
        ("fly_net_of_floor", FLY_NET_ANCHOR_ERR2),
    ]:
        ratio = observed / model_8192_equal
        anchors[name] = {
            "observed_8192_err2": observed,
            "model_8192_equal_err2": model_8192_equal,
            "model_over_observed_ratio": model_8192_equal / observed,
            "calibration_ratio_observed_over_model": ratio,
            "calibrated_equal_32768_err2": ratio * by_n[32768]["equal_err2"],
            "calibrated_optimal_32768_err2": ratio * by_n[32768]["optimal_err2"],
            "calibrated_optimal_30720_err2": ratio * pred_30720,
            "mc_c_over_n_32768_through_8192_anchor": observed * 8192 / 32768,
            "mc_c_over_n_30720_through_8192_anchor": observed * 8192 / ENTRY1_DENSE_EVALS,
        }

    fly_pred = anchors["fly_net_of_floor"]["calibrated_optimal_30720_err2"]
    grader_pred = anchors["grader"]["calibrated_optimal_30720_err2"]
    beta_tail = spectrum["tail_fits"]["ranks_20_500"]["beta"]
    closed = min(fly_pred, grader_pred) >= 1.0e-6 and beta_tail <= 1.5
    reopened = min(fly_pred, grader_pred) <= 2.0e-7 or beta_tail > 1.5
    if reopened:
        verdict = "REOPENED"
    elif closed:
        verdict = "CLOSED"
    else:
        verdict = "CLOSED_BY_ERROR_SCALE"

    return {
        "entry1_dense_evals": ENTRY1_DENSE_EVALS,
        "entry1_err2_budget": ENTRY1_ERR2_BUDGET,
        "model_optimal_30720_err2": pred_30720,
        "model_optimal_30720_scaling_beta_between_16k_32k": beta_n,
        "anchors": anchors,
        "verdict": verdict,
        "verdict_reason": (
            "calibrated optimal 30k prediction is >=1e-6 under both anchors "
            "and spectral tail is slow"
            if verdict == "CLOSED"
            else "see calibrated predictions and spectral beta"
        ),
    }


def write_markdown(results: dict) -> None:
    rows = results["scaling"]
    cal = results["calibration"]
    spec = results["spectrum"]
    lines = [
        "# NNGP BQ Error-vs-N Scaling and Kernel Spectrum",
        "",
        "Scope: offline analysis only. No estimator edits, no Fly, no network, no pytest. "
        "All new artifacts are under `paired_fly_logs/fingerprint_theory/`.",
        "",
        "## Method",
        "",
        "- Reused the validated depth-32 arc-cosine NNGP kernel and current randomized antithetic Hadamard sign-block design.",
        "- Pair means are exact: each `512`-point antithetic block pair reduces to the Walsh spectrum of the relative sign mask.",
        "- Optimal BQ solves are exact in the block-constant invariant subspace. The right hand side `k_mu` is constant for all equal-radius sign points, so the full solve reduces to the block-pair mean matrix.",
        "- The `8192` spectrum of `K/N` is exact via Walsh diagonalization into `256 * 2` small block matrices.",
        "- `m` and `k_mu` use the prior validated MC estimates from `nngp_design_pretest_results.json`; the calibration step anchors the model to measured estimator variance at `N=8192`.",
        "",
        "## Error Scaling",
        "",
        "| N | blocks | equal err^2 | optimal err^2 | equal/optimal | pair mean | MC c/N through grader | MC c/N through Fly |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        n = r["n"]
        g_mc = GRADER_ANCHOR_ERR2 * 8192 / n
        f_mc = FLY_NET_ANCHOR_ERR2 * 8192 / n
        lines.append(
            f"| {n} | {r['blocks']} | {r['equal_err2']:.9g} | {r['optimal_err2']:.9g} | "
            f"{r['equal_over_optimal']:.6f} | {r['wKw_pair_mean']:.9g} | {g_mc:.3g} | {f_mc:.3g} |"
        )

    lines += [
        "",
        "Pair-mean estimator standard error: `0`, because this run used the exact antithetic-Hadamard block sum rather than random pair sampling.",
        "",
        "## Spectrum",
        "",
        f"`K/N` eigendecomposition at `N={spec['n']}` used `{spec['method']}`.",
        f"Top eigenvalue: `{spec['top_20'][0]:.9g}`. Sum of eigenvalues: `{spec['sum_eigenvalues']:.9g}`.",
        "",
        "| fit range | beta in lambda_k ~ k^-beta | R^2 |",
        "|---|---:|---:|",
    ]
    for fit in spec["tail_fits"].values():
        lines.append(
            f"| ranks {fit['rank_start']}-{fit['rank_end']} | {fit['beta']:.4f} | {fit['r2']:.4f} |"
        )
    lines += [
        "",
        "Top 20 eigenvalues:",
        "",
        "`" + ", ".join(f"{v:.6g}" for v in spec["top_20"]) + "`",
        "",
        "The full top-500 list is in `bq_nscaling_20260706_results.json`.",
        "",
        "## Calibration",
        "",
        "| anchor | model/reality @ 8192 | observed/model multiplier | calibrated optimal err^2 @ 32768 | calibrated optimal err^2 @ 30720 | MC c/N @ 30720 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, anchor in cal["anchors"].items():
        ratio = anchor["calibration_ratio_observed_over_model"]
        lines.append(
            f"| {name} | {anchor['model_over_observed_ratio']:.3g}x | `{ratio:.5g}` | "
            f"{anchor['calibrated_optimal_32768_err2']:.3g} | "
            f"{anchor['calibrated_optimal_30720_err2']:.3g} | "
            f"{anchor['mc_c_over_n_30720_through_8192_anchor']:.3g} |"
        )
    lines += [
        "",
        "Note: the table names the multiplicative calibration as observed/model; multiply raw NNGP err^2 by this ratio to match the measured `N=8192` variance anchor.",
        "",
        "## Verdict",
        "",
    ]
    if cal["verdict"] == "CLOSED":
        lines.append(
            "**EVALUATION-BASED LANE CLOSED** for the cluster's top entry under this validated average-case model. "
            "The calibrated optimal prediction near `30k` evaluations remains above `1e-6`, at least `10x` the entry-1 `~1e-7` error budget, and the kernel spectrum does not show the fast tail decay needed for quadrature superconvergence."
        )
    else:
        lines.append(
            f"Verdict: **{cal['verdict']}**. The calibrated predictions or spectral fit did not satisfy the pre-registered closure condition."
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    t0 = time.time()
    max_blocks = max(n // (2 * WIDTH) for n in N_VALUES)
    integrals = load_prior_integrals()
    signs = make_sign_blocks(max_blocks)
    pair_mean, even_hat, odd_hat = block_cross_spectra(signs)
    rows = equal_and_optimal_rows(pair_mean, integrals)
    spectrum = spectrum_from_block_hats(even_hat, odd_hat, SPECTRUM_N // (2 * WIDTH))
    cal = calibration(rows, spectrum)

    results = {
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "seed": SEED,
            "n_values": N_VALUES,
            "spectrum_n": SPECTRUM_N,
            "top_eigenvalues": TOP_EIGENVALUES,
            "anchors": {
                "grader_8192_err2": GRADER_ANCHOR_ERR2,
                "fly_net_8192_err2": FLY_NET_ANCHOR_ERR2,
                "entry1_dense_evals": ENTRY1_DENSE_EVALS,
                "entry1_err2_budget": ENTRY1_ERR2_BUDGET,
            },
        },
        "integrals": integrals,
        "scaling": rows,
        "spectrum": spectrum,
        "calibration": cal,
        "seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(results)

    print("NNGP BQ N-scaling summary")
    print("N equal_err2 optimal_err2 equal/optimal")
    for r in rows:
        print(f"{r['n']} {r['equal_err2']:.9g} {r['optimal_err2']:.9g} {r['equal_over_optimal']:.6f}")
    print(f"spectral_beta_20_500 {spectrum['tail_fits']['ranks_20_500']['beta']:.6f}")
    for name, anchor in cal["anchors"].items():
        print(
            f"{name}_calibration_observed_over_model "
            f"{anchor['calibration_ratio_observed_over_model']:.9g}"
        )
        print(
            f"{name}_calibrated_optimal_30720 "
            f"{anchor['calibrated_optimal_30720_err2']:.9g}"
        )
    print(f"verdict {cal['verdict']}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
