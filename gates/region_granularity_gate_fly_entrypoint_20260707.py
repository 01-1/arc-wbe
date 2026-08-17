#!/usr/bin/env python3
"""Machine-side ReLU region granularity gate for truth-bank MLPs."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_engine import build_mlp  # noqa: E402

SCRIPT_VERSION = "region-granularity-gate-fly-v1"
WIDTH = 256
DEPTH = 32
CHORDS = 12
GRID = 65
T_MIN = -1.0
T_MAX = 1.0
EPS = 1.0e-30


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", type=Path, required=True, help="Ignored; present for fly-bank wrapper compatibility.")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument("--mode")
    return parser.parse_args(argv)


def weights_sha256(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for w in weights:
        digest.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return digest.hexdigest()


def np_weights(seed: int) -> list[np.ndarray]:
    mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=int(seed))
    return [np.asarray(w, dtype=np.float32) for w in mlp.weights]


def sample_seed(bank_seed: int, bank_index: int) -> int:
    label = f"region-granularity-v1:{bank_seed}:{bank_index}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def forward_stack(xs: np.ndarray, weights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    acts = np.empty((DEPTH, xs.shape[0], WIDTH), dtype=np.float32)
    gates = np.empty((DEPTH, xs.shape[0], WIDTH), dtype=np.bool_)
    y = xs.astype(np.float32, copy=False)
    for layer, w in enumerate(weights):
        pre = y @ w
        gate = pre > 0.0
        y = np.maximum(pre, 0.0).astype(np.float32, copy=False)
        acts[layer] = y
        gates[layer] = gate
    return acts, gates


def chord_points(rng: np.random.Generator) -> np.ndarray:
    x0 = rng.standard_normal(WIDTH).astype(np.float32)
    u = rng.standard_normal(WIDTH).astype(np.float32)
    u /= max(float(np.linalg.norm(u)), EPS)
    ts = np.linspace(T_MIN, T_MAX, GRID, dtype=np.float32)
    return x0[None, :] + ts[:, None] * u[None, :]


def process_one(bank_index: int, original_index: int, seeds: np.ndarray, checksums: np.ndarray) -> dict[str, Any]:
    started = time.monotonic()
    seed = int(seeds[bank_index])
    weights = np_weights(seed)
    checksum = weights_sha256(weights)
    rng = np.random.default_rng(sample_seed(seed, int(original_index)))
    chord_length = T_MAX - T_MIN
    total_intervals = CHORDS * (GRID - 1)
    layer_flip_counts = np.zeros(DEPTH, dtype=np.int64)
    neuron_flipped = np.zeros((DEPTH, WIDTH), dtype=np.bool_)
    interval_has_flip = np.zeros((DEPTH, total_intervals), dtype=np.bool_)
    interval_flip_counts = np.zeros((DEPTH, total_intervals), dtype=np.int16)
    between_acc = np.zeros(DEPTH, dtype=np.float64)
    within_acc = np.zeros(DEPTH, dtype=np.float64)
    total_acc = np.zeros(DEPTH, dtype=np.float64)

    interval_offset = 0
    for _ in range(CHORDS):
        xs = chord_points(rng)
        acts, gates = forward_stack(xs, weights)
        diffs = gates[:, 1:, :] != gates[:, :-1, :]
        counts = np.sum(diffs, axis=2)
        layer_flip_counts += np.sum(counts, axis=1)
        neuron_flipped |= np.any(diffs, axis=1)
        span = slice(interval_offset, interval_offset + GRID - 1)
        interval_flip_counts[:, span] = counts.astype(np.int16)
        interval_has_flip[:, span] = counts > 0
        interval_offset += GRID - 1

        # For each layer, decompose output variation along the chord into the
        # variance of interval midpoints plus affine variation inside intervals
        # with unchanged sampled activation pattern.
        for layer in range(DEPTH):
            y = acts[layer].astype(np.float64)
            total_acc[layer] += float(np.mean(np.var(y, axis=0)))
            left = y[:-1]
            right = y[1:]
            mid = 0.5 * (left + right)
            between_acc[layer] += float(np.mean(np.var(mid, axis=0)))
            same = counts[layer] == 0
            if np.any(same):
                delta = right[same] - left[same]
                within_acc[layer] += float(np.mean(delta * delta) / 12.0)

    density_by_layer = layer_flip_counts.astype(np.float64) / (CHORDS * chord_length)
    live_counts = np.sum(neuron_flipped, axis=1).astype(np.int64)
    frozen_fraction = 1.0 - live_counts.astype(np.float64) / float(WIDTH)
    within_share = within_acc / np.maximum(within_acc + between_acc, EPS)
    within_share_total = within_acc / np.maximum(total_acc, EPS)
    deep_layers = np.arange(24, DEPTH)
    deep_flip_intervals = np.sum(interval_flip_counts[deep_layers], axis=0)
    flip_intervals = deep_flip_intervals[deep_flip_intervals > 0]
    cooccur_fraction = float(np.mean(flip_intervals >= 2)) if flip_intervals.size else 0.0

    return {
        "bank_index": int(original_index),
        "shard_bank_index": int(bank_index),
        "seed": seed,
        "weights_sha256": checksum,
        "bank_weights_sha256": str(checksums[bank_index]),
        "checksum_ok": checksum == str(checksums[bank_index]),
        "breakpoints_per_sigma_total": float(np.sum(density_by_layer)),
        "typical_region_extent_sigma": float(1.0 / max(np.sum(density_by_layer), EPS)),
        "effective_live_hyperplanes_total": int(np.sum(live_counts)),
        "layer31_live_hyperplanes": int(live_counts[31]),
        "deep_flip_cooccur_fraction": cooccur_fraction,
        "deep_flip_events_per_flipping_interval_median": float(np.median(flip_intervals)) if flip_intervals.size else 0.0,
        "layers": [
            {
                "layer": int(layer),
                "breakpoints_per_sigma": float(density_by_layer[layer]),
                "live_hyperplanes": int(live_counts[layer]),
                "frozen_fraction": float(frozen_fraction[layer]),
                "within_region_variance_share": float(within_share[layer]),
                "within_region_variance_share_total": float(within_share_total[layer]),
                "flip_interval_fraction": float(np.mean(interval_has_flip[layer])),
                "cooccurring_flip_interval_fraction": float(np.mean(interval_flip_counts[layer] >= 2)),
            }
            for layer in range(DEPTH)
        ],
        "wall_time_s": float(time.monotonic() - started),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    bank = np.load(args.bank)
    seeds = bank["seeds"]
    checksums = bank["weights_sha256"]
    original_indices = bank["original_indices"] if "original_indices" in bank.files else np.arange(seeds.shape[0])
    n_rows = int(seeds.shape[0])
    indices = [idx for idx in range(n_rows) if idx * args.shard_count // n_rows == args.shard_index]
    records = []
    failures = []
    for bank_index in indices:
        try:
            records.append(process_one(bank_index, int(original_indices[bank_index]), seeds, checksums))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "bank_index": int(bank_index),
                    "seed": int(seeds[bank_index]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    print(
        json.dumps(
            {
                "task": "region_granularity_gate",
                "script_version": SCRIPT_VERSION,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "bank_rows": n_rows,
                "records": records,
                "failures": failures,
                "n_records": len(records),
                "n_failures": len(failures),
                "config": {
                    "chords": CHORDS,
                    "grid": GRID,
                    "t_min": T_MIN,
                    "t_max": T_MAX,
                    "width": WIDTH,
                    "depth": DEPTH,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
