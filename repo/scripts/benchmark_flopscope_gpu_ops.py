"""Benchmark flopscope CPU vs local GPU bridge for representative operations."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    fn: Callable[[fnp.ndarray, fnp.ndarray], fnp.ndarray]
    flops: int
    input_bytes: int


def _matmul_case(n: int) -> Case:
    bytes_ = 2 * n * n * 8
    flops = n * n * n
    return Case(
        name=f"matmul_{n}",
        shape=(n, n),
        fn=lambda a, b: fnp.matmul(a, b),
        flops=flops,
        input_bytes=bytes_,
    )


def _einsum_mm_case(n: int) -> Case:
    bytes_ = 2 * n * n * 8
    flops = n * n * n
    return Case(
        name=f"einsum_mm_{n}",
        shape=(n, n),
        fn=lambda a, b: fnp.einsum("ij,jk->ik", a, b),
        flops=flops,
        input_bytes=bytes_,
    )


def _add_case(n: int) -> Case:
    bytes_ = 2 * n * n * 8
    flops = n * n
    return Case(
        name=f"add_{n}",
        shape=(n, n),
        fn=lambda a, b: fnp.add(a, b),
        flops=flops,
        input_bytes=bytes_,
    )


def _sqrt_case(n: int) -> Case:
    bytes_ = n * n * 8
    flops = n * n
    return Case(
        name=f"sqrt_{n}",
        shape=(n, n),
        fn=lambda a, b: fnp.sqrt(fnp.abs(a)),
        flops=2 * flops,
        input_bytes=bytes_,
    )


def _sum_case(n: int) -> Case:
    bytes_ = n * n * 8
    flops = n * n
    return Case(
        name=f"sum_{n}",
        shape=(n, n),
        fn=lambda a, b: fnp.sum(a, axis=0),
        flops=flops,
        input_bytes=bytes_,
    )


def _run_case(case: Case, repeats: int, gpu: bool) -> tuple[float, int, str, int, int]:
    rng = np.random.default_rng(123)
    a_np = rng.standard_normal(case.shape)
    b_np = rng.standard_normal(case.shape)

    flops.configure_gpu(gpu)
    with flops.BudgetContext(flop_budget=10**18, quiet=True):
        a = fnp.array(a_np)
        b = fnp.array(b_np)
        _ = case.fn(a, b)

    times = []
    checksum = ""
    last_flops = 0
    for _ in range(repeats):
        with flops.BudgetContext(flop_budget=10**18, quiet=True) as ctx:
            a = fnp.array(a_np)
            b = fnp.array(b_np)
            t0 = time.perf_counter()
            out = case.fn(a, b)
            elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_flops = ctx.flops_used
        checksum = f"{float(np.asarray(out).sum()):.9f}"

    status = flops.gpu_status()
    return (
        min(times),
        last_flops,
        checksum,
        int(status["gpu_call_count"]),
        int(status["fallback_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--min-flops",
        type=int,
        default=int(os.environ.get("FLOPSCOPE_GPU_MIN_FLOPS", "5000000")),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(os.environ.get("FLOPSCOPE_GPU_MIN_TRANSFER_BYTES", "0")),
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=float(os.environ.get("FLOPSCOPE_GPU_MIN_FLOPS_PER_BYTE", "0.05")),
    )
    args = parser.parse_args()

    os.environ["FLOPSCOPE_GPU_MIN_FLOPS"] = str(args.min_flops)
    os.environ["FLOPSCOPE_GPU_MIN_TRANSFER_BYTES"] = str(args.threshold)
    os.environ["FLOPSCOPE_GPU_MIN_FLOPS_PER_BYTE"] = str(args.intensity)

    cases: list[Case] = []
    for n in (128, 256, 512, 1024):
        cases.extend(
            [
                _matmul_case(n),
                _einsum_mm_case(n),
                _add_case(n),
                _sqrt_case(n),
                _sum_case(n),
            ]
        )

    print("gpu_status", flops.gpu_status())
    print(
        "name,n,input_bytes,est_flops,est_flops_per_byte,cpu_s,gpu_s,speedup,"
        "counted_flops,checksum_match,gpu_calls,fallbacks"
    )
    for case in cases:
        cpu_s, counted_flops, cpu_checksum, _cpu_gpu_calls, _cpu_fallbacks = _run_case(
            case, args.repeats, gpu=False
        )
        gpu_s, _gpu_counted_flops, gpu_checksum, gpu_calls, fallbacks = _run_case(
            case, args.repeats, gpu=True
        )
        print(
            f"{case.name},{case.shape[0]},{case.input_bytes},{case.flops},"
            f"{case.flops / max(case.input_bytes, 1):.6f},{cpu_s:.8f},{gpu_s:.8f},"
            f"{cpu_s / gpu_s if gpu_s else float('inf'):.6f},{counted_flops},"
            f"{cpu_checksum == gpu_checksum},{gpu_calls},{fallbacks}"
        )


if __name__ == "__main__":
    main()
