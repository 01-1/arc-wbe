#!/usr/bin/env python3
"""Fly-bank-style entrypoint for the readout-smoothing gate.

Runs one truth-bank shard machine-side and returns only compact per-MLP
statistics. The --estimator argument is accepted for compatibility with the
existing bank runner but is intentionally ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "local_engine.py").is_file():
    REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_engine import build_mlp  # noqa: E402

SCRIPT_VERSION = "readout-smoothing-fly-entrypoint-20260706-v1"
WIDTH = 256
DEPTH = 32
NS = (1024, 4096, 8192)
REPS = 8
BIAS_N = 262_144
PAIR_CHUNK = 8192
EPS = 1.0e-12

_erf = np.vectorize(math.erf, otypes=[np.float64])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument("--mode")
    args = parser.parse_args(argv)
    if args.shard_count <= 0:
        raise SystemExit("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")
    return args


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def gaussian_relu(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, EPS)
    alpha = mu / sigma
    return mu * normal_cdf(alpha) + sigma * normal_pdf(alpha)


def weights_sha256(weights: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for w in weights:
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def rebuilt_weights(seed: int) -> list[np.ndarray]:
    mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=int(seed))
    return [np.asarray(w, dtype=np.float32) for w in mlp.weights]


def replicate_seed(bank_seed: int, mlp_index: int, n: int, rep: int, anti: bool) -> int:
    label = f"readout-smoothing-fly-v1:{bank_seed}:{mlp_index}:{n}:{rep}:{int(anti)}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def large_seed(bank_seed: int, mlp_index: int) -> int:
    label = f"readout-smoothing-fly-large-v1:{bank_seed}:{mlp_index}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def forward_readouts(weights: list[np.ndarray], n: int, seed: int, antithetic: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n // 2
        x0 = rng.standard_normal((half, WIDTH), dtype=np.float32)
        x = np.concatenate([x0, -x0], axis=0)
    else:
        x = rng.standard_normal((n, WIDTH), dtype=np.float32)

    direct = np.empty((DEPTH, WIDTH), dtype=np.float64)
    smooth = np.empty((DEPTH, WIDTH), dtype=np.float64)
    for layer, w in enumerate(weights):
        z = x @ w
        z64 = z.astype(np.float64)
        mu = z64.mean(axis=0)
        sigma = np.sqrt(np.maximum(np.mean(z64 * z64, axis=0) - mu * mu, 0.0))
        r = np.maximum(z, 0.0)
        direct[layer] = r.mean(axis=0, dtype=np.float64)
        smooth[layer] = gaussian_relu(mu, sigma)
        x = r.astype(np.float32, copy=False)
    return direct, smooth


def large_pass_stats(weights: list[np.ndarray], n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    z_s1 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    z_s2 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    z_s3 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    z_s4 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    r_s1 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    r_s2 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    pair_s1 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    pair_s2 = np.zeros((DEPTH, WIDTH), dtype=np.float64)

    done = 0
    while done < half:
        b = min(PAIR_CHUNK, half - done)
        x0 = rng.standard_normal((b, WIDTH), dtype=np.float32)
        xp = x0
        xn = -x0
        for layer, w in enumerate(weights):
            zp = xp @ w
            zn = xn @ w
            zp64 = zp.astype(np.float64)
            zn64 = zn.astype(np.float64)
            z_s1[layer] += zp64.sum(axis=0) + zn64.sum(axis=0)
            z_s2[layer] += (zp64 * zp64).sum(axis=0) + (zn64 * zn64).sum(axis=0)
            z_s3[layer] += (zp64**3).sum(axis=0) + (zn64**3).sum(axis=0)
            z_s4[layer] += (zp64**4).sum(axis=0) + (zn64**4).sum(axis=0)
            rp = np.maximum(zp, 0.0).astype(np.float64)
            rn = np.maximum(zn, 0.0).astype(np.float64)
            r_s1[layer] += rp.sum(axis=0) + rn.sum(axis=0)
            r_s2[layer] += (rp * rp).sum(axis=0) + (rn * rn).sum(axis=0)
            pm = 0.5 * (rp + rn)
            pair_s1[layer] += pm.sum(axis=0)
            pair_s2[layer] += (pm * pm).sum(axis=0)
            xp = rp.astype(np.float32)
            xn = rn.astype(np.float32)
        done += b

    total = float(n)
    half_f = float(half)
    mu = z_s1 / total
    raw2 = z_s2 / total
    raw3 = z_s3 / total
    raw4 = z_s4 / total
    m2 = np.maximum(raw2 - mu * mu, 0.0)
    m3 = raw3 - 3.0 * mu * raw2 + 2.0 * mu**3
    m4 = raw4 - 4.0 * mu * raw3 + 6.0 * mu * mu * raw2 - 3.0 * mu**4
    sigma = np.sqrt(m2)
    skew = m3 / np.maximum(sigma**3, EPS)
    excess_kurtosis = m4 / np.maximum(sigma**4, EPS) - 3.0
    direct = r_s1 / total
    smooth = gaussian_relu(mu, sigma)
    relu_var = np.maximum(r_s2 / total - direct * direct, 0.0)
    pair_mean = pair_s1 / half_f
    anti_eff_var = 2.0 * np.maximum(pair_s2 / half_f - pair_mean * pair_mean, 0.0)
    return {
        "direct": direct,
        "smooth": smooth,
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "relu_var": relu_var,
        "anti_eff_var": anti_eff_var,
    }


def mse_by_layer(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean((pred - truth) ** 2, axis=1)


def run_one(bank_index: int, seed: int, truth: np.ndarray, expected_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    weights = rebuilt_weights(seed)
    local_sha = weights_sha256(weights)
    if local_sha != expected_sha:
        raise ValueError(f"weights checksum mismatch local={local_sha} expected={expected_sha}")

    replicate_records: list[dict[str, Any]] = []
    for n in NS:
        for anti in (False, True):
            direct_rows = []
            smooth_rows = []
            for rep in range(REPS):
                direct, smooth = forward_readouts(
                    weights,
                    n,
                    replicate_seed(seed, bank_index, n, rep, anti),
                    anti,
                )
                direct_rows.append(mse_by_layer(direct, truth))
                smooth_rows.append(mse_by_layer(smooth, truth))
            direct_mean = np.mean(np.asarray(direct_rows), axis=0)
            smooth_mean = np.mean(np.asarray(smooth_rows), axis=0)
            replicate_records.append(
                {
                    "n": n,
                    "antithetic": anti,
                    "direct_mse": direct_mean.tolist(),
                    "smooth_mse": smooth_mean.tolist(),
                    "ratio": (smooth_mean / np.maximum(direct_mean, EPS)).tolist(),
                }
            )

    large = large_pass_stats(weights, BIAS_N, large_seed(seed, bank_index))
    smooth_bias2_unit = (large["smooth"] - truth) ** 2
    direct_bias2_unit = (large["direct"] - truth) ** 2
    return {
        "bank_index": bank_index,
        "seed": seed,
        "weights_sha256": local_sha,
        "checksum_ok": True,
        "replicates": replicate_records,
        "large_n": BIAS_N,
        "smooth_bias2": np.mean(smooth_bias2_unit, axis=1).tolist(),
        "direct_large_error2": np.mean(direct_bias2_unit, axis=1).tolist(),
        "anti_eff_var": np.mean(large["anti_eff_var"], axis=1).tolist(),
        "relu_var": np.mean(large["relu_var"], axis=1).tolist(),
        "truth_floor_unit_se": (
            np.std(large["anti_eff_var"], axis=1, ddof=1) / math.sqrt(WIDTH)
        ).tolist(),
        "skew_abs_q": np.quantile(np.abs(large["skew"]), [0.10, 0.50, 0.90], axis=1).T.tolist(),
        "excess_kurtosis_q": np.quantile(large["excess_kurtosis"], [0.10, 0.50, 0.90], axis=1).T.tolist(),
        "wall_time_s": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    bank = np.load(args.bank)
    seeds = bank["seeds"]
    truths = bank["truths"].astype(np.float64)
    checksums = bank["weights_sha256"]
    n_rows = int(seeds.shape[0])
    indices = [
        index
        for index in range(n_rows)
        if index * args.shard_count // n_rows == args.shard_index
    ]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for bank_index in indices:
        seed = int(seeds[bank_index])
        try:
            records.append(
                run_one(
                    bank_index,
                    seed,
                    truths[bank_index],
                    str(checksums[bank_index]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - structured research failure.
            failures.append(
                {
                    "bank_index": bank_index,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    payload = {
        "task": "bank",
        "gate": "readout_smoothing",
        "script_version": SCRIPT_VERSION,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "bank_rows": n_rows,
        "config": {
            "sample_counts": list(NS),
            "reps": REPS,
            "bias_n": BIAS_N,
            "pair_chunk": PAIR_CHUNK,
            "estimator_argument_ignored": True,
        },
        "records": records,
        "failures": failures,
        "n_records": len(records),
        "n_failures": len(failures),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
