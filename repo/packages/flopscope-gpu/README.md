# flopscope GPU fork

This is a workspace-local fork of `flopscope` based on the installed
`0.8.0rc2+np2.2.6` package.

GPU execution is opt-in and currently uses CuPy as an execution accelerator for
selected NumPy calls while preserving the public `flopscope.numpy.ndarray`
contract. Results are copied back to CPU `FlopscopeArray` objects before they
leave a counted operation, so FLOP accounting and downstream code keep the same
semantics as upstream flopscope.

Enable it with either:

```python
import flopscope as flops

flops.configure_gpu(enabled=True)
```

or:

```sh
FLOPSCOPE_GPU=1 uv run python estimator.py
```

The backend falls back to CPU if CuPy is not installed or if an operation is not
in the supported GPU allowlist. This fork is intended for local experimentation;
the challenge grader may not provide GPU libraries, and estimators should not
depend on GPU support during evaluation.

The bridge copies inputs to GPU and results back to CPU for each offloaded
operation because `FlopscopeArray` remains a CPU `numpy.ndarray` subclass. To
avoid slowing down tiny kernels, the default gate is a FLOP floor: operations
below 5 million counted FLOPs stay on CPU. A low FLOPs-per-byte guard also
rejects unusually transfer-heavy edge cases below 0.05 FLOPs/byte. Tune these
with:

```sh
FLOPSCOPE_GPU_MIN_FLOPS=5000000
FLOPSCOPE_GPU_MIN_FLOPS_PER_BYTE=0.05
```

The transfer-size gate is available only as an explicit experiment via
`FLOPSCOPE_GPU_MIN_TRANSFER_BYTES`; it defaults to disabled.
