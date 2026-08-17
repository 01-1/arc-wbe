#!/usr/bin/env python3
"""Validation harness for the dependency-free Sobol runtime scratch generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sobol_runtime_feasibility_generator_20260710.py"
spec = importlib.util.spec_from_file_location("sobol_runtime_generator", SOURCE)
assert spec and spec.loader
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def main() -> None:
    seed = 0x1234_5678
    ours = generator.sobol_uniform(seed)
    scipy_scrambled = qmc.Sobol(d=256, scramble=True, seed=seed).random_base2(m=12)
    scipy_plain = qmc.Sobol(d=256, scramble=False, seed=seed).random_base2(m=12)
    plain = generator.sobol_uniform(seed, scramble=False)
    rows = generator.sobol_normal_rows(seed)
    anti = generator.antipodal_rows(seed)
    p = np.array([2.0**-53, 1e-10, 1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999999, 1 - 1e-10, 1 - 2.0**-53])
    acklam_error = float(np.max(np.abs(generator.ndtri_dependency_free(p) - ndtri(p))))
    result = {
        "fixed_config": {"d": generator.WIDTH, "m": generator.M, "rows": generator.ROWS, "bits": generator.BITS},
        "deterministic": bool(np.array_equal(rows, generator.sobol_normal_rows(seed))),
        "scipy_lms_uniform_max_abs_error": float(np.max(np.abs(ours - scipy_scrambled))),
        "scipy_lms_uniform_exact": bool(np.array_equal(ours, scipy_scrambled)),
        "scipy_plain_uniform_max_abs_error": float(np.max(np.abs(plain - scipy_plain))),
        "scipy_plain_uniform_exact": bool(np.array_equal(plain, scipy_plain)),
        "row_shape": list(rows.shape),
        "row_dtype": str(rows.dtype),
        "row_norm_min": float(np.min(np.linalg.norm(rows, axis=1))),
        "row_norm_max": float(np.max(np.linalg.norm(rows, axis=1))),
        "max_coordinate_abs_mean": float(np.max(np.abs(np.mean(rows, axis=0)))),
        "max_coordinate_std_error": float(np.max(np.abs(np.std(rows, axis=0) - 1.0))),
        "antipode_shape": list(anti.shape),
        "antipode_max_abs_error": float(np.max(np.abs(anti[generator.ROWS:] + anti[:generator.ROWS]))),
        "acklam_vs_scipy_ndtri_max_abs_error": acklam_error,
        "direction_asset_bytes_raw": int(generator._DIRECTION_NUMBERS.nbytes),
        "direction_asset_sha256": __import__("hashlib").sha256(generator._DIRECTION_NUMBERS.tobytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
