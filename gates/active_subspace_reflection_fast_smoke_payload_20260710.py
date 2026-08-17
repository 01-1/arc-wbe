#!/usr/bin/env python3
"""Fast, research-only active-subspace reflection smoke payload."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np


PAYLOAD_ROOT = Path(__file__).resolve().parents[2]
if str(PAYLOAD_ROOT) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_ROOT))

from local_engine import build_mlp  # noqa: E402


SCRIPT_VERSION = "active-subspace-reflection-fast-smoke-v1"
WIDTH = 256
DEPTH = 32
RANKS = (1, 8)
FAMILIES = ("input_gram", "downstream_gram", "reference_jacobian")
LAWS = ("sphere",)
N_REFERENCE = 1
N_HELDOUT = 64
REFERENCE_REPS = 1
STREAM = 0xA17E5


def _weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _sphere(rng: np.random.Generator, n: int, width: int) -> np.ndarray:
    x = rng.standard_normal((n, width)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    return x / norms


def _forward_final(weights: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for w in weights:
        h = np.maximum(h @ w, np.float32(0.0))
    return h


def _linear_gram(weights: list[np.ndarray]) -> np.ndarray:
    product = np.eye(WIDTH, dtype=np.float32)
    for w in weights:
        product = product @ w
        scale = np.linalg.norm(product, ord="fro")
        if scale > 0.0 and np.isfinite(scale):
            product /= np.float32(scale)
    product64 = product.astype(np.float64)
    return product64 @ product64.T


def _reference_jacobian_gram(
    weights: list[np.ndarray], references: np.ndarray
) -> np.ndarray:
    gram = np.zeros((WIDTH, WIDTH), dtype=np.float64)
    for x in references:
        h = x[None, :].astype(np.float32)
        jac = np.eye(WIDTH, dtype=np.float32)
        for w in weights:
            z = h @ w
            mask = (z[0] > 0.0).astype(np.float32)
            jac = jac @ w
            jac *= mask[None, :]
            h = np.maximum(z, np.float32(0.0))
        scale = np.linalg.norm(jac, ord="fro")
        if scale > 0.0 and np.isfinite(scale):
            jac /= np.float32(scale)
        jac64 = jac.astype(np.float64)
        gram += jac64 @ jac64.T
    return gram / max(len(references), 1)


def _projectors(
    weights: list[np.ndarray], references: np.ndarray
) -> dict[str, np.ndarray]:
    w0 = weights[0].astype(np.float64)
    grams = {
        "input_gram": w0 @ w0.T,
        "downstream_gram": _linear_gram(weights),
        "reference_jacobian": _reference_jacobian_gram(weights, references),
    }
    out: dict[str, np.ndarray] = {}
    for family, gram in grams.items():
        gram = (gram + gram.T) * 0.5
        _, vectors = np.linalg.eigh(gram)
        out[family] = vectors[:, ::-1]
    return out


def _trace_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    mean_a = np.mean(a64, axis=0)
    mean_b = np.mean(b64, axis=0)
    centered_a = a64 - mean_a[None, :]
    centered_b = b64 - mean_b[None, :]
    va = float(np.mean(np.sum(centered_a * centered_a, axis=1)))
    vb = float(np.mean(np.sum(centered_b * centered_b, axis=1)))
    cov = float(np.mean(np.sum(centered_a * centered_b, axis=1)))
    denom = math.sqrt(max(va * vb, 1e-300))
    rho = cov / denom if denom > 0.0 else 1.0
    gain = 1.0 / max(1.0 + rho, 1e-12)
    diff = float(np.sum((mean_a - mean_b) ** 2))
    mean_scale = (va + vb) / max(a.shape[0], 1)
    return {
        "var_a": va,
        "var_b": vb,
        "cov": cov,
        "rho": rho,
        "gain": gain,
        "mean_difference_stat": diff / max(mean_scale, 1e-300),
    }


def _batched_orbit_metrics(
    weights: list[np.ndarray], x: np.ndarray, bases: dict[str, np.ndarray]
) -> list[dict[str, float | int | str]]:
    plans = [(family, rank) for family in FAMILIES for rank in RANKS]
    reflections: list[np.ndarray] = []
    for family, rank in plans:
        q = bases[family][:, :rank]
        reflection = np.eye(WIDTH, dtype=np.float64) - 2.0 * (q @ q.T)
        reflections.append(reflection)
    n = x.shape[0]
    batches = [x, -x]
    for reflection in reflections:
        tx = x.astype(np.float64) @ reflection.T
        batches.extend((tx.astype(np.float32), (-tx).astype(np.float32)))
    values = _forward_final(weights, np.concatenate(batches, axis=0))
    g_x = 0.5 * (values[:n] + values[n : 2 * n])
    out: list[dict[str, float | int | str]] = []
    for index, (family, rank) in enumerate(plans):
        start = (2 + 2 * index) * n
        g_tx = 0.5 * (values[start : start + n] + values[start + n : start + 2 * n])
        metric = _trace_metrics(g_x, g_tx)
        metric.update({"rank": rank, "n": n, "family": family, "law": "sphere"})
        out.append(metric)
    return out


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
    weights = [np.ascontiguousarray(np.asarray(w), dtype=np.float32) for w in mlp.weights]
    actual = _weights_sha256(weights)
    if actual != expected:
        raise RuntimeError(f"weight checksum mismatch for shard {shard_index}")

    results: list[dict[str, object]] = []
    for reference_rep in range(reference_reps):
        rng = np.random.default_rng((seed ^ STREAM ^ (reference_rep * 0x9E3779B9)) % (1 << 63))
        references = _sphere(rng, n_reference, WIDTH)
        bases = _projectors(weights, references)
        heldout = _sphere(rng, n_heldout, WIDTH)
        for metric in _batched_orbit_metrics(weights, heldout, bases):
            metric["reference_rep"] = reference_rep
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
            "batched_orbit": True,
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
    args.output.write_text(__import__("json").dumps(result, sort_keys=True), encoding="utf-8")
    print(__import__("json").dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
