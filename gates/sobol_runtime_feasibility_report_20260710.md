# Sobol runtime feasibility audit (2026-07-10)

## Go/no-go

SciPy is **not submission-safe as an implicit dependency**. `pyproject.toml`
declares only `flopscope` and `whestbench` as runtime dependencies, and
`uv.lock` contains no `scipy` package. The Fly image installs exactly the
locked no-dev environment via `uv sync --frozen --no-dev`; it does not install
research payload dependencies. The normal estimator path downloads only the
single `estimator.py` file into that image. A local `uv run python` probe finds
NumPy but raises `ModuleNotFoundError: No module named 'scipy'`.

The current Sobol research payload imports both `scipy.stats.qmc.Sobol` and
`scipy.special.ndtri`, so that payload is research-only and cannot be inlined
into the submission unchanged.

## Dependency-free fallback

[sobol_runtime_feasibility_generator_20260710.py](sobol_runtime_feasibility_generator_20260710.py)
implements fixed `d=256`, `m=12` (4096 rows), 30-bit Joe--Kuo direction
numbers, SciPy-compatible LMS+digital-shift scrambling, Gray-code generation,
and row-normalized normal scores. It uses only the standard library and NumPy.
The rows are returned at norm `sqrt(256)` and `antipodal_rows()` returns an
8192-by-256 exact `(+x,-x)` concatenation.

The direction-number asset is 30,720 raw bytes (256x30 uint32), 21,814 bytes
zlib-compressed, and 27,268 bytes in the embedded base85 form. Exact SciPy
LMS reproduction therefore requires carrying this approximately 22--27 KB
asset (or an equivalent direction-number table) inside the estimator. The
smallest legal alternative is to keep this compact fixed table and generate
the scramble/Gray-code rows at runtime; do not replace it with an unregistered
different digital sequence without a new research preregistration.

## Inverse normal and flopscope

`flopscope.stats.norm.ppf` is already available in the locked runtime and is
explicitly metered at 83 FLOPs per output element. Its implementation is an
Acklam approximation plus one Newton refinement and does not import SciPy.
`flopscope.numpy` exposes `uint32`, `uint64`, `bitwise_xor`, `bitwise_and`,
`left_shift`, and `right_shift`, so the Sobol integer/Gray-code path is
flopscope-compatible. A custom inverse-normal routine is unnecessary for
estimator code; call `flops.stats.norm.ppf` on clipped uniforms instead.

## Verification

The companion check script compares the fallback against SciPy 1.16.2 for the
same seed and construction, and checks deterministic seeding, marginal scale,
norms, antipodes, and inverse-normal error. It is a read-only research check;
no Fly run was launched and no estimator/history files were changed.

Observed checks: scrambled and plain Sobol uniforms matched SciPy exactly
(maximum absolute error `0.0`); rows were `(4096, 256)` `float32`, with norms
in `[15.999999, 16.0]`, maximum coordinate mean magnitude `0.001261`, and
maximum coordinate standard-deviation error `0.002688`. Antipodes were exact
(`8192 x 256`, error `0.0`), deterministic replay passed, and the standalone
Acklam approximation differed from SciPy `ndtri` by at most `6.45e-9` on the
tail/central probe. The same generator also ran successfully under the locked
`uv` environment where SciPy import is unavailable.
