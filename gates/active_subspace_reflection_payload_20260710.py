#!/usr/bin/env python3
"""Truth-free Fly payload for active-subspace reflection covariance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from local_engine import build_mlp


SCRIPT_VERSION = "active-subspace-reflection-v1"
WIDTH = 256
DEPTH = 32
RANKS = (1, 2, 4, 8)
FAMILIES = ("input_gram", "downstream_gram", "reference_jacobian")
LAWS = ("sphere", "haar")
N_REFERENCE = 16
N_HELDOUT = 256
REFERENCE_REPS = 2
STREAM = 0xA17E5


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _sphere(rng: np.random.Generator, n: int, width: int) -> np.ndarray:
    x = rng.standard_normal((n, width))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _haar(rng: np.random.Generator, width: int) -> np.ndarray:
    z = rng.standard_normal((width, width))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def _forward_final(weights: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x
    for w in weights:
        h = np.maximum(h @ w, 0.0)
    return h


def _linear_gram(weights: list[np.ndarray]) -> np.ndarray:
    product = np.eye(weights[0].shape[0], dtype=np.float64)
    for w in weights:
        product = product @ w.astype(np.float64)
        scale = np.linalg.norm(product, ord="fro")
        if scale > 0.0 and np.isfinite(scale):
            product /= scale
    return product @ product.T


def _reference_jacobian_gram(
    weights: list[np.ndarray], references: np.ndarray
) -> np.ndarray:
    gram = np.zeros((WIDTH, WIDTH), dtype=np.float64)
    eye = np.eye(WIDTH, dtype=np.float64)
    for x in references:
        h = x[None, :]
        jac = eye
        for w in weights:
            w64 = w.astype(np.float64)
            z = h @ w64
            mask = (z[0] > 0.0).astype(np.float64)
            jac = jac @ w64
            jac *= mask[None, :]
            h = np.maximum(z, 0.0)
        scale = np.linalg.norm(jac, ord="fro")
        if scale > 0.0 and np.isfinite(scale):
            jac /= scale
        gram += jac @ jac.T
    return gram / max(len(references), 1)


def _projectors(
    weights: list[np.ndarray], references: np.ndarray
) -> dict[str, np.ndarray]:
    grams = {
        "input_gram": weights[0].astype(np.float64) @ weights[0].astype(np.float64).T,
        "downstream_gram": _linear_gram(weights),
        "reference_jacobian": _reference_jacobian_gram(weights, references),
    }
    out: dict[str, np.ndarray] = {}
    for family, gram in grams.items():
        gram = (gram + gram.T) * 0.5
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        out[family] = vectors[:, order]
    return out


def _trace_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    centered_a = a - mean_a[None, :]
    centered_b = b - mean_b[None, :]
    va = float(np.mean(np.sum(centered_a * centered_a, axis=1)))
    vb = float(np.mean(np.sum(centered_b * centered_b, axis=1)))
    cov = float(np.mean(np.sum(centered_a * centered_b, axis=1)))
    denom = math.sqrt(max(va * vb, 1e-300))
    rho = cov / denom if denom > 0.0 else 1.0
    gain = 1.0 / max(1.0 + rho, 1e-12)
    # This is a descriptive sanity statistic: two independent sample means
    # should differ on the scale predicted by their trace variances.
    diff = float(np.sum((mean_a - mean_b) ** 2))
    mean_scale = (va + vb) / max(a.shape[0], 1)
    sanity = diff / max(mean_scale, 1e-300)
    return {
        "var_a": va,
        "var_b": vb,
        "cov": cov,
        "rho": rho,
        "gain": gain,
        "mean_difference_stat": sanity,
    }


def _orbit_metric(
    weights: list[np.ndarray], x: np.ndarray, basis: np.ndarray, rank: int
) -> dict[str, float]:
    q = basis[:, :rank]
    reflection = np.eye(WIDTH, dtype=np.float64) - 2.0 * (q @ q.T)
    tx = x @ reflection.T
    values = _forward_final(weights, np.concatenate((x, -x, tx, -tx), axis=0))
    n = x.shape[0]
    g_x = 0.5 * (values[:n] + values[n : 2 * n])
    g_tx = 0.5 * (values[2 * n : 3 * n] + values[3 * n :])
    metric = _trace_metrics(g_x, g_tx)
    metric.update({"rank": rank, "n": n})
    return metric


def run_one(
    shard_index: int,
    bank_path: Path,
    n_reference: int = N_REFERENCE,
    n_heldout: int = N_HELDOUT,
    reference_reps: int = REFERENCE_REPS,
) -> dict[str, object]:
    bank = np.load(bank_path)
    seed = int(bank["seeds"][shard_index])
    expected = str(bank["weights_sha256"][shard_index])
    mlp = build_mlp(WIDTH, DEPTH, seed)
    weights = [np.ascontiguousarray(np.asarray(w), dtype=np.float64) for w in mlp.weights]
    actual = _weights_sha256(weights)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")

    results: list[dict[str, object]] = []
    for reference_rep in range(reference_reps):
        rng = np.random.default_rng((seed ^ STREAM ^ (reference_rep * 0x9E3779B9)) % (1 << 63))
        references = _sphere(rng, n_reference, WIDTH)
        bases = _projectors(weights, references)
        for law in LAWS:
            heldout = _sphere(rng, n_heldout, WIDTH)
            if law == "haar":
                heldout = heldout @ _haar(rng, WIDTH)
            for family in FAMILIES:
                for rank in RANKS:
                    metric = _orbit_metric(weights, heldout, bases[family], rank)
                    metric.update({"reference_rep": reference_rep, "law": law, "family": family})
                    results.append(metric)

    return {
        "ok": True,
        "script_version": SCRIPT_VERSION,
        "mlp_index": shard_index,
        "seed": seed,
        "checksum_ok": True,
        "weights_sha256": actual,
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "ranks": RANKS,
            "families": FAMILIES,
            "laws": LAWS,
            "n_reference": n_reference,
            "n_heldout": n_heldout,
            "reference_reps": reference_reps,
        },
        "metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("WHEST_SHARD_INDEX", "0")))
    parser.add_argument("--bank", type=Path, default=Path("analysis/truth_bank/truth_bank.npz"))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--n-reference", type=int, default=N_REFERENCE)
    parser.add_argument("--n-heldout", type=int, default=N_HELDOUT)
    parser.add_argument("--reference-reps", type=int, default=REFERENCE_REPS)
    args = parser.parse_args()
    result = run_one(
        args.shard_index,
        args.bank,
        max(args.n_reference, 1),
        max(args.n_heldout, 1),
        max(args.reference_reps, 1),
    )
    args.output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
