"""Optional GPU execution helpers for the local flopscope fork.

This module deliberately keeps GPU support behind an opt-in execution boundary.
``FlopscopeArray`` is a ``numpy.ndarray`` subclass, so public values still need
to be CPU ndarrays. The GPU backend accelerates selected backend calls, then
copies results back before normal flopscope wrapping and accounting continue.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as _np

_enabled = os.environ.get("FLOPSCOPE_GPU", "").lower() in {"1", "true", "yes", "on"}
_strict = os.environ.get("FLOPSCOPE_GPU_STRICT", "").lower() in {"1", "true", "yes", "on"}
_use_predictor = os.environ.get("FLOPSCOPE_GPU_USE_PREDICTOR", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_min_free_bytes = int(os.environ.get("FLOPSCOPE_GPU_MIN_FREE_BYTES", str(512 * 1024 * 1024)))
_min_flops = int(os.environ.get("FLOPSCOPE_GPU_MIN_FLOPS", "5000000"))
_min_transfer_bytes = int(os.environ.get("FLOPSCOPE_GPU_MIN_TRANSFER_BYTES", "0"))
_min_flops_per_byte = float(os.environ.get("FLOPSCOPE_GPU_MIN_FLOPS_PER_BYTE", "0.05"))
_min_predicted_speedup = float(os.environ.get("FLOPSCOPE_GPU_MIN_PREDICTED_SPEEDUP", "1.05"))
_max_transfer_fraction = float(os.environ.get("FLOPSCOPE_GPU_MAX_TRANSFER_FRACTION", "0.25"))
_cupy: Any | None = None
_cupy_error: Exception | None = None
_runtime_error: Exception | None = None
_runtime_checked = False
_last_skip_reason: str | None = None
_gpu_call_count = 0
_fallback_count = 0

_GPU_FUNCTIONS = {
    "absolute",
    "add",
    "clip",
    "divide",
    "dot",
    "einsum",
    "exp",
    "log",
    "matmul",
    "maximum",
    "mean",
    "minimum",
    "multiply",
    "negative",
    "power",
    "sqrt",
    "subtract",
    "sum",
    "tensordot",
    "true_divide",
    "where",
}


def configure_gpu(enabled: bool = True) -> None:
    """Enable or disable optional GPU-backed execution."""

    global _enabled, _runtime_error, _runtime_checked, _last_skip_reason
    _enabled = bool(enabled)
    if _enabled:
        _runtime_error = None
        _runtime_checked = False
        _last_skip_reason = None


def gpu_enabled() -> bool:
    """Return whether GPU execution has been requested."""

    return _enabled


def gpu_available() -> bool:
    """Return whether CuPy can import and the CUDA runtime is usable."""

    return _runtime_available()


def gpu_status() -> dict[str, Any]:
    """Return a small diagnostic snapshot for the optional GPU backend."""

    cp = _load_cupy()
    runtime_available = _runtime_available() if cp is not None else False
    free_bytes, total_bytes = _memory_info(cp) if runtime_available else (None, None)
    return {
        "enabled": _enabled,
        "available": runtime_available,
        "backend": "cupy" if runtime_available else None,
        "error": repr(_cupy_error) if _cupy_error is not None else None,
        "runtime_error": repr(_runtime_error) if _runtime_error is not None else None,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "min_flops": _min_flops,
        "min_transfer_bytes": _min_transfer_bytes,
        "min_flops_per_byte": _min_flops_per_byte,
        "use_predictor": _use_predictor,
        "min_predicted_speedup": _min_predicted_speedup,
        "last_skip_reason": _last_skip_reason,
        "gpu_call_count": _gpu_call_count,
        "fallback_count": _fallback_count,
        "supported_functions": tuple(sorted(_GPU_FUNCTIONS)),
    }


def maybe_call_gpu(
    fn: Any, *args: Any, flop_cost: int | None = None, **kwargs: Any
) -> tuple[bool, Any]:
    """Try to execute ``fn`` through CuPy.

    Returns ``(True, result)`` when the operation ran on GPU and was copied back
    to CPU, otherwise ``(False, None)`` so the caller can use the normal NumPy
    path. Exceptions from supported GPU calls are allowed to propagate; silent
    fallback after partial GPU execution would hide real numerical failures.
    """

    if not _enabled:
        return False, None
    name = getattr(fn, "__name__", "")
    if name not in _GPU_FUNCTIONS:
        return False, None
    cp = _load_cupy()
    if cp is None or not _runtime_available():
        _record_fallback("runtime_unavailable")
        return False, None
    gpu_fn = getattr(cp, name, None)
    if gpu_fn is None:
        _record_fallback("unsupported_function")
        return False, None
    if not _has_memory_headroom(cp, name, args, kwargs, flop_cost=flop_cost):
        _record_fallback(_last_skip_reason or "insufficient_memory")
        return False, None

    try:
        gpu_args = _to_gpu_tree(cp, args)
        gpu_kwargs = {key: _to_gpu_tree(cp, value) for key, value in kwargs.items()}
        result = gpu_fn(*gpu_args, **gpu_kwargs)
        cp.cuda.Stream.null.synchronize()
        _record_gpu_call()
        return True, _to_cpu_tree(cp, result)
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        _record_runtime_failure(exc)
        if _strict:
            raise
        _record_fallback(type(exc).__name__)
        return False, None


def _load_cupy() -> Any | None:
    global _cupy, _cupy_error
    if _cupy is not None:
        return _cupy
    if _cupy_error is not None:
        return None
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on optional package
        _cupy_error = exc
        return None
    _cupy = cp
    return cp


def _runtime_available() -> bool:
    """Return True only when CuPy can talk to CUDA and allocate a tiny array."""

    global _runtime_checked
    cp = _load_cupy()
    if cp is None:
        return False
    if _runtime_error is not None:
        return False
    if _runtime_checked:
        return True
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy found no CUDA devices")
        probe = cp.empty((1,), dtype=cp.float32)
        probe.fill(0)
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        _record_runtime_failure(exc)
        return False
    _runtime_checked = True
    return True


def _record_runtime_failure(exc: Exception) -> None:
    global _runtime_error, _runtime_checked
    _runtime_error = exc
    _runtime_checked = False
    cp = _cupy
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def _record_gpu_call() -> None:
    global _gpu_call_count
    _gpu_call_count += 1


def _record_fallback(reason: str) -> None:
    global _fallback_count, _last_skip_reason
    _fallback_count += 1
    _last_skip_reason = reason


def _memory_info(cp: Any) -> tuple[int | None, int | None]:
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        _record_runtime_failure(exc)
        return None, None
    return int(free_bytes), int(total_bytes)


def _has_memory_headroom(
    cp: Any, op_name: str, args: Any, kwargs: Any, *, flop_cost: int | None
) -> bool:
    """Cheap preflight so low-memory CUDA states fall back to CPU per op."""

    global _last_skip_reason
    free_bytes, _total_bytes = _memory_info(cp)
    if free_bytes is None:
        _last_skip_reason = "cuda_meminfo_failed"
        return False
    transfer_bytes = _tree_nbytes(args) + _tree_nbytes(kwargs)
    if _min_transfer_bytes > 0 and transfer_bytes < _min_transfer_bytes:
        _last_skip_reason = (
            f"below_min_transfer_bytes(transfer={transfer_bytes}, "
            f"minimum={_min_transfer_bytes})"
        )
        return False
    if not _use_predictor and flop_cost is not None and flop_cost < _min_flops:
        _last_skip_reason = (
            f"below_min_flops(flops={flop_cost}, minimum={_min_flops}, "
            f"transfer={transfer_bytes})"
        )
        return False
    if (
        not _use_predictor
        and _min_flops_per_byte > 0
        and flop_cost is not None
        and transfer_bytes > 0
    ):
        flops_per_byte = flop_cost / transfer_bytes
        if flops_per_byte < _min_flops_per_byte:
            _last_skip_reason = (
                f"below_min_flops_per_byte(value={flops_per_byte:.6g}, "
                f"minimum={_min_flops_per_byte:.6g}, flops={flop_cost}, "
                f"transfer={transfer_bytes})"
            )
            return False
    if _use_predictor and flop_cost is not None and transfer_bytes > 0:
        predicted = _predict_speedup(op_name, flop_cost, transfer_bytes)
        if predicted is not None and predicted < _min_predicted_speedup:
            _last_skip_reason = (
                f"below_predicted_speedup(value={predicted:.6g}, "
                f"minimum={_min_predicted_speedup:.6g}, op={op_name}, "
                f"flops={flop_cost}, transfer={transfer_bytes})"
            )
            return False
    # Account for host->device inputs, at least one output-sized scratch, and
    # backend temporaries. This is intentionally conservative because the public
    # API copies results back to CPU after every op.
    required = max(_min_free_bytes, int(transfer_bytes * 3))
    if transfer_bytes > 0:
        required = max(required, int(transfer_bytes / max(_max_transfer_fraction, 1e-9)))
    if free_bytes < required:
        _last_skip_reason = (
            f"insufficient_free_memory(free={free_bytes}, required={required}, "
            f"transfer={transfer_bytes})"
        )
        return False
    _last_skip_reason = None
    return True


def _tree_nbytes(value: Any) -> int:
    if isinstance(value, _np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (tuple, list)):
        return sum(_tree_nbytes(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_tree_nbytes(item) for item in value.values())
    return 0


def _op_family(op_name: str) -> str:
    if op_name in {"matmul", "dot", "tensordot"}:
        return "matmul"
    if op_name == "einsum":
        return "einsum_mm"
    if op_name in {"sum", "mean"}:
        return "sum"
    if op_name in {
        "add",
        "clip",
        "divide",
        "maximum",
        "minimum",
        "multiply",
        "subtract",
        "true_divide",
        "where",
    }:
        return "add"
    if op_name in {"absolute", "exp", "log", "negative", "power", "sqrt"}:
        return "sqrt"
    return "other"


def _predict_speedup(op_name: str, flop_cost: int, transfer_bytes: int) -> float | None:
    """Predict CPU/GPU speedup from the local per-family linear calibration."""

    family = _op_family(op_name)
    cpu_s = _predict_seconds("cpu", family, flop_cost, transfer_bytes)
    gpu_s = _predict_seconds("gpu", family, flop_cost, transfer_bytes)
    if cpu_s is None or gpu_s is None or gpu_s <= 0:
        return None
    return cpu_s / gpu_s


_CPU_LINEAR = {
    # family: (bias + family_intercept, input_gb_slope, flops_g_slope)
    "add": (-8.46838e-06, 0.055149452, 0.0034468408),
    "einsum_mm": (0.00065389923, -0.29327636, 0.22666659),
    "matmul": (0.00020809060, -0.31185279, 0.016144399),
    "sqrt": (-0.00010633678, 0.30440510, 0.076101275),
    "sum": (0.00003329585, 0.16882305, 0.021102882),
}

_GPU_LINEAR = {
    # family: (bias + family_intercept, input_gb_slope, flops_g_slope)
    "add": (0.000005522236, 0.061959127, 0.0038724454),
    "einsum_mm": (0.00039135901, 0.12503778, 0.0065383439),
    "matmul": (0.00010991234, 0.24604765, 0.0051278073),
    "sqrt": (-0.00018535899, 0.42722801, 0.10680700),
    "sum": (0.00010195763, 0.12028859, 0.015036074),
}


def _predict_seconds(
    target: str, family: str, flop_cost: int, transfer_bytes: int
) -> float | None:
    table = _CPU_LINEAR if target == "cpu" else _GPU_LINEAR
    coeffs = table.get(family)
    if coeffs is None:
        return None
    intercept, input_gb_slope, flops_g_slope = coeffs
    input_gb = transfer_bytes / 1e9
    flops_g = flop_cost / 1e9
    return max(intercept + input_gb_slope * input_gb + flops_g_slope * flops_g, 0.0)


def _to_gpu_tree(cp: Any, value: Any) -> Any:
    if isinstance(value, _np.ndarray):
        return cp.asarray(value)
    if isinstance(value, tuple):
        return tuple(_to_gpu_tree(cp, item) for item in value)
    if isinstance(value, list):
        return [_to_gpu_tree(cp, item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_gpu_tree(cp, item) for key, item in value.items()}
    return value


def _to_cpu_tree(cp: Any, value: Any) -> Any:
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(cp, item) for item in value)
    if isinstance(value, list):
        return [_to_cpu_tree(cp, item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(_to_cpu_tree(cp, item) for item in value)
    return value
