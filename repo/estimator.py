"""Cumulant-propagation estimator for ReLU MLPs.

Submission for https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026.

Implements the K=1 (mean propagation) and K=2 (covariance propagation)
algorithms from Wu et al., "Estimating the expected output of wide random
MLPs more efficiently than sampling" (arXiv:2605.05179).

The estimator automatically selects the highest-order algorithm that fits
comfortably inside the per-MLP FLOP budget:

* Covariance propagation (K=2) – full covariance matrix, O(depth·width³) FLOPs.
* Mean propagation (K=1)       – diagonal variance only, O(depth·width²) FLOPs.

Both variants use the exact ReLU Gaussian-moment formulas (power cumulants)
described in the paper.  When the supplied FLOP budget has room below the
score multiplier floor, the K=2 path is blended with a small antithetic
Monte-Carlo estimate to reduce residual bias without changing the default
sample-free behavior under tight budgets.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
from collections import defaultdict
from functools import cache
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

flops.configure(symmetry_warnings=False)

_MIN_VARIANCE = 1e-30
_K3_MIN_WIDTH = 16
_ENABLE_FACTOR_K3 = True


def _hermite_prob(n: int, x: fnp.ndarray) -> fnp.ndarray:
    if n == 0:
        return fnp.ones_like(x)
    if n == 1:
        return x
    prev2 = fnp.ones_like(x)
    prev1 = x
    for k in range(2, n + 1):
        cur = x * prev1 - (k - 1) * prev2
        prev2, prev1 = prev1, cur
    return prev1


def _relu_wick(mean: fnp.ndarray, var: fnp.ndarray, k: int, p: int = 1) -> fnp.ndarray:
    """Return ``E[d^k ReLU(Z)^p / dZ^k]`` for ``Z ~ N(mean, var)``.

    This is the ReLU Wick-coefficient helper used by the factorized algorithm.
    Only the orders needed by the K=3 factorized propagation are implemented.
    """
    sigma = fnp.sqrt(fnp.maximum(var, _MIN_VARIANCE))
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    if k < p:
        order = p - k
        falling = _prod(range(p - k + 1, p + 1))
        trunc = [Phi, phi]
        if order >= 2:
            trunc.append(Phi - alpha * phi)
        if order >= 3:
            trunc.append((alpha * alpha + 2.0) * phi)
        if order >= 4:
            trunc.append(3.0 * Phi - (alpha ** 3 + 3.0 * alpha) * phi)
        moment = 0.0
        for r in range(order + 1):
            moment = moment + math.comb(order, r) * mean ** (order - r) * sigma ** r * trunc[r]
        return falling * moment

    if p > 1:
        return math.factorial(p) * _relu_wick(mean, var, k - p + 1, 1)

    if p == 1:
        if k == 0:
            return sigma * phi + mean * Phi
        if k == 1:
            return Phi
        return (-1.0) ** (k - 2) * sigma ** (-(k - 1)) * _hermite_prob(k - 2, alpha) * phi

    raise ValueError(f"unsupported ReLU Wick coefficient p={p}")

def _relu_moments(mu: fnp.ndarray, var: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Return (mean, second_moment, gain) of ReLU(N(mu, var)).

    Uses the standard rectified-Gaussian formulas:

        alpha   = mu / sigma
        phi     = pdf(alpha)
        Phi     = cdf(alpha)
        E[z]    = mu * Phi + sigma * phi
        E[z²]   = (mu² + var) * Phi + mu * sigma * phi
        gain    = Phi(alpha)          # first Hermite coefficient b̃_1

    The ``gain`` is the c_i factor from the paper's Algorithm 2 (covariance
    propagation) and is used to approximate off-diagonal post-ReLU covariances.
    """
    sigma = fnp.sqrt(fnp.maximum(var, _MIN_VARIANCE))
    alpha = mu / sigma

    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    mean = mu * Phi + sigma * phi
    ez2 = (mu * mu + var) * Phi + mu * sigma * phi
    gain = Phi

    return mean, ez2, gain


def _prod(values):
    out = 1
    for value in values:
        out = out * value
    return out


@cache
def _set_partitions(items):
    items = tuple(range(items)) if isinstance(items, int) else tuple(items)
    if not items:
        return ((),)
    first = items[0]
    rest_parts = _set_partitions(items[1:])
    out = []
    for part in rest_parts:
        out.append((frozenset((first,)),) + part)
        for i, block in enumerate(part):
            out.append(part[:i] + (frozenset((first, *block)),) + part[i + 1:])
    return tuple(out)


@cache
def _weak_compositions(parts: int, total: int):
    if parts == 0:
        return ((),) if total == 0 else ()
    if parts == 1:
        return ((total,),)
    out = []
    for first in range(total + 1):
        for rest in _weak_compositions(parts - 1, total - first):
            out.append((first,) + rest)
    return tuple(out)


@cache
def _vector_partitions(vec: tuple[int, ...], prev: tuple[int, ...] | None = None):
    if all(v == 0 for v in vec):
        return ((),)
    out = []
    ranges = [range(v + 1) for v in vec]
    for block in itertools.product(*ranges):
        if sum(block) == 0 or (prev is not None and block < prev):
            continue
        resid = tuple(a - b for a, b in zip(vec, block))
        for rest in _vector_partitions(resid, block):
            out.append((block,) + rest)
    return tuple(out)


@cache
def _int_partitions(total: int):
    if total == 0:
        return ((),)
    return tuple(
        tuple(sorted((v[0] for v in part), reverse=True))
        for part in _vector_partitions((total,))
    )


def _vec_part_coef(part, divide_fac: bool = True):
    if not part:
        return 1.0
    counts = {v: part.count(v) for v in set(part)}
    n_by_axis = [sum(v[i] for v in part) for i in range(len(part[0]))]
    denom = _prod(
        math.factorial(count) * _prod(math.factorial(x) for x in vec) ** count
        for vec, count in counts.items()
    )
    out = 1.0 / denom
    if not divide_fac:
        out *= _prod(math.factorial(n) for n in n_by_axis)
    return out


def _check_vec_partition(part, dim: int):
    total = [0] * dim
    for vec in part:
        for i, value in enumerate(vec):
            total[i] += value
    return tuple(total)


def _vec_support_connected(part, dim: int) -> bool:
    parent = list(range(dim))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for vec in part:
        supp = [i for i, value in enumerate(vec) if value > 0]
        for i in supp[1:]:
            union(supp[0], i)
    return len({find(i) for i in range(dim)}) == 1


@cache
def _all_terms_iso_k3():
    """Terms used by upstream ``factored_nonlin_kprop_k3`` in SIMPLE mode."""
    terms = []
    for degree in range(1, 5):
        for int_part in _int_partitions(degree):
            if len(int_part) > 3:
                continue
            for total in range(9):
                for vec in _weak_compositions(len(int_part), total):
                    for vec_part in _vector_partitions(vec):
                        if not vec_part:
                            if len(int_part) == 1:
                                terms.append((int_part, vec_part))
                            continue
                        if any(
                            sum(v) <= 2 and sum(1 for x in v if x > 0) <= 1
                            for v in vec_part
                        ):
                            continue
                        if not _vec_support_connected(vec_part, len(int_part)):
                            continue
                        if any(sum(v) > 4 for v in vec_part):
                            continue
                        if sum(max(sum(math.ceil(v[i] / 2) for i in range(len(v))) - 1, 1) for v in vec_part) > 2:
                            continue
                        terms.append((int_part, vec_part))

    grouped = defaultdict(int)
    for int_part, vec_part in terms:
        best = None
        for perm in itertools.permutations(range(len(int_part))):
            if int_part != tuple(int_part[i] for i in perm):
                continue
            image = tuple(sorted(tuple(v[i] for i in perm) for v in vec_part))
            if best is None or image < best:
                best = image
        grouped[(int_part, best)] += 1
    return tuple((int_part, vec_part, count) for (int_part, vec_part), count in grouped.items())


def _zero_repeated(a: fnp.ndarray) -> fnp.ndarray:
    if a.ndim <= 1:
        return a
    idxs = fnp.indices(a.shape)
    mask = fnp.ones(a.shape)
    for i in range(a.ndim):
        for j in range(i + 1, a.ndim):
            mask = mask * (idxs[i] != idxs[j])
    return a * mask


def _symmetrize(a: fnp.ndarray, vec: tuple[int, ...] | None = None) -> fnp.ndarray:
    if a.ndim <= 1:
        return a
    if vec is None:
        vec = (1,) * a.ndim
    perms = [
        p for p in itertools.permutations(range(a.ndim))
        if tuple(vec[i] for i in p) == vec
    ]
    return sum(fnp.transpose(a, p) for p in perms) / float(len(perms))


def _expand(a: fnp.ndarray | float, positions: tuple[int, ...], dim: int):
    if not hasattr(a, "ndim"):
        return a
    shape = [1] * dim
    for axis, size in zip(positions, a.shape):
        shape[axis] = size
    return fnp.reshape(a, tuple(shape))


def _int_partition_coef(part: tuple[int, ...]) -> int:
    counts = defaultdict(int)
    for block in part:
        counts[block] += 1
    return math.factorial(sum(part)) // _prod(
        math.factorial(size) ** count * math.factorial(count)
        for size, count in counts.items()
    )


def _proj_coef(n: int, d: int, r: int) -> list[float]:
    c = d - r + n / 2.0 - 1.0
    out = [0.0] * r
    for j in range(r, d // 2 + 1):
        out.append(
            ((-1.0) ** r)
            * (c - r)
            / (4.0 ** j)
            / math.factorial(r)
            / math.factorial(j - r)
            / c
            / _prod(1.0 - c + m for m in range(j))
        )
    return out


@cache
def _multigraphs(vertices: int, edges: int):
    edge_types = [(i, j) for i in range(vertices) for j in range(i, vertices)]
    return tuple(
        tuple((edge, mult) for edge, mult in zip(edge_types, comp) if mult > 0)
        for comp in _weak_compositions(len(edge_types), edges)
    )


def _multigraph_coef(graph, arities: tuple[int, ...], lap_coef: bool = True) -> float:
    m = sum(mult for _, mult in graph)
    loops = sum(mult for (a, b), mult in graph if a == b)
    used = [0] * len(arities)
    for (a, b), mult in graph:
        used[a] += mult
        used[b] += mult
    if any(used[i] > arities[i] for i in range(len(arities))):
        return 0.0
    out = math.factorial(m) * (2 ** (m - loops))
    for i, arity in enumerate(arities):
        out *= _prod(range(arity - used[i] + 1, arity + 1))
    for _, mult in graph:
        out /= math.factorial(mult)
    if not lap_coef:
        out /= _prod(range(sum(arities) - 2 * m + 1, sum(arities) + 1))
    return out


def _harmonic_dslice_r2_scalar(core: fnp.ndarray, metric: fnp.ndarray, part: tuple[int, ...]) -> fnp.ndarray:
    n = metric.shape[0]
    out = fnp.zeros((n,) * len(part))
    for graph in _multigraphs(len(part), 2):
        coef = _multigraph_coef(graph, part, lap_coef=False)
        if coef == 0.0:
            continue
        factors = []
        labels = []
        out_labels = list("abcd"[: len(part)])
        for (a, b), mult in graph:
            for _ in range(mult):
                factors.append(metric)
                labels.append(f"{out_labels[a]} {out_labels[b]}")
        if not factors:
            continue
        expr = ", ".join(labels) + " -> " + " ".join(out_labels)
        out = out + coef * fnp.einsum(expr, *factors)
    return _zero_repeated(out * core)


class _HTensor:
    def __init__(self, core: fnp.ndarray | float, r: int = 0, n: int | None = None, metric=None):
        self.core = core
        self.r = r
        self.n = n if n is not None else core.shape[0]
        if metric is None:
            self.metric = fnp.eye(self.n)
        elif not hasattr(metric, "ndim") or metric.ndim == 0:
            self.metric = fnp.eye(self.n) * metric
        elif metric.ndim == 1:
            self.metric = fnp.diag(metric)
        else:
            self.metric = metric

    @property
    def ndim(self) -> int:
        return (self.core.ndim if hasattr(self.core, "ndim") else 0) + 2 * self.r

    def contract_w(self, w: fnp.ndarray) -> "_HTensor":
        w_t = w.T
        if hasattr(self.core, "ndim") and self.core.ndim > 0:
            letters = "abcd"[: self.core.ndim]
            outs = "wxyz"[: self.core.ndim]
            expr = f"{letters}," + ",".join(f"{outs[i]}{letters[i]}" for i in range(self.core.ndim))
            expr += f"->{outs}"
            core = fnp.einsum(expr, self.core, *([w_t] * self.core.ndim))
            core = _symmetrize(core)
        else:
            core = self.core
        metric = w_t @ self.metric @ w
        return _HTensor(core, r=self.r, n=w.shape[1], metric=metric)

    def get_dslice(self, part: tuple[int, ...]) -> fnp.ndarray:
        if self.r == 0:
            return _tensor_dslice(self.core, part)
        if self.r == 2 and (not hasattr(self.core, "ndim") or self.core.ndim == 0):
            return _harmonic_dslice_r2_scalar(self.core, self.metric, part)
        raise NotImplementedError("Only r=0 and scalar r=2 HTensors are needed for K=3 SIMPLE")


def _tensor_dslice(a: fnp.ndarray, part: tuple[int, ...], output_zero_repeated: bool = False) -> fnp.ndarray:
    if a.ndim == 1:
        out = a
    elif a.ndim == 2:
        out = fnp.diag(a) if part == (2,) else a
    elif a.ndim == 3:
        if part == (3,):
            idx = fnp.arange(a.shape[0])
            out = a[idx, idx, idx]
        elif part == (2, 1):
            idx = fnp.arange(a.shape[0])
            out = a[idx[:, None], idx[:, None], idx[None, :]]
        else:
            out = a
    elif a.ndim == 4:
        idx = fnp.arange(a.shape[0])
        if part == (4,):
            out = a[idx, idx, idx, idx]
        elif part == (3, 1):
            out = a[idx[:, None], idx[:, None], idx[:, None], idx[None, :]]
        elif part == (2, 2):
            out = a[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
        elif part == (2, 1, 1):
            out = a[idx[:, None, None], idx[:, None, None], idx[None, :, None], idx[None, None, :]]
        else:
            out = a
    else:
        raise NotImplementedError(f"Unsupported diagonal slice order {a.ndim}")
    return _zero_repeated(out) if output_zero_repeated else out


class _DSTensor:
    def __init__(self, slices: dict[tuple[int, ...], fnp.ndarray], n: int | None = None, d: int | None = None, autozero: bool = False):
        self.slices = {
            tuple(sorted(part, reverse=True)): (_zero_repeated(value) if autozero else value)
            for part, value in slices.items()
        }
        if self.slices:
            part, value = next(iter(self.slices.items()))
            self.n = n if n is not None else value.shape[0]
            self.d = d if d is not None else sum(part)
        else:
            assert n is not None and d is not None
            self.n = n
            self.d = d

    @property
    def ndim(self) -> int:
        return self.d

    def get_slice(self, part: tuple[int, ...], strict: bool = True):
        sorted_part = tuple(sorted(part, reverse=True))
        if sorted_part not in self.slices:
            if strict:
                raise KeyError(part)
            return 0.0
        tmp = list(sorted_part)
        perm = []
        for block in part:
            idx = tmp.index(block)
            perm.append(idx)
            tmp[idx] = -1
        value = self.slices[sorted_part]
        return fnp.transpose(value, tuple(perm)) if len(perm) > 1 else value

    def get_dslice(self, part: tuple[int, ...]):
        return self.get_slice(part)

    def to_tensor(self) -> fnp.ndarray:
        out = fnp.zeros((self.n,) * self.d)
        for part, dslice in self.slices.items():
            if self.d == 1:
                out = out + dslice
            elif self.d == 2:
                if part == (2,):
                    out = out + fnp.diag(dslice)
                else:
                    out = out + dslice
            elif self.d == 3:
                if part == (3,):
                    idx = fnp.arange(self.n)
                    tmp = fnp.zeros((self.n, self.n, self.n))
                    tmp[idx, idx, idx] = dslice
                elif part == (2, 1):
                    idx = fnp.arange(self.n)
                    tmp = fnp.zeros((self.n, self.n, self.n))
                    tmp[idx[:, None], idx[:, None], idx[None, :]] = dslice
                else:
                    tmp = dslice
                out = out + _int_partition_coef(part) * _symmetrize(tmp)
            elif self.d == 4:
                idx = fnp.arange(self.n)
                tmp = fnp.zeros((self.n,) * 4)
                if part == (4,):
                    tmp[idx, idx, idx, idx] = dslice
                elif part == (3, 1):
                    tmp[idx[:, None], idx[:, None], idx[:, None], idx[None, :]] = dslice
                elif part == (2, 2):
                    tmp[idx[:, None], idx[:, None], idx[None, :], idx[None, :]] = dslice
                elif part == (2, 1, 1):
                    tmp[idx[:, None, None], idx[:, None, None], idx[None, :, None], idx[None, None, :]] = dslice
                else:
                    tmp = dslice
                out = out + _int_partition_coef(part) * _symmetrize(tmp)
        return out


class _DSTower(dict):
    @classmethod
    def from_slices(cls, slices: dict[tuple[int, ...], fnp.ndarray], autozero: bool = False):
        grouped = defaultdict(dict)
        for part, value in slices.items():
            grouped[sum(part)][part] = value
        return cls({d: _DSTensor(parts, autozero=autozero) for d, parts in grouped.items()})

    def get_slice(self, part: tuple[int, ...], strict: bool = True):
        return self[sum(part)].get_slice(part, strict=strict)

    def is_downward_closed(self) -> bool:
        all_parts = {part for tensor in self.values() for part in tensor.slices}
        for part in all_parts:
            if sum(part) <= 1:
                continue
            for i in range(len(part)):
                p = list(part)
                p[i] -= 1
                if p[i] == 0:
                    p.pop(i)
                if tuple(sorted(p, reverse=True)) not in all_parts:
                    return False
        return True


def _ds_part_sum(tower: _DSTower, coef_fn, strict: bool = True) -> _DSTower:
    def get_block(block):
        nonzero = tuple(i for i, value in enumerate(block) if value > 0)
        part = tuple(block[i] for i in nonzero)
        if not part:
            return 1.0
        try:
            return _expand(tower.get_slice(part, strict=True), nonzero, len(block))
        except KeyError:
            if strict:
                raise
            return 0.0

    out = {}
    for degree in range(1, max(tower.keys()) + 1):
        if degree not in tower:
            continue
        slices = {}
        for int_part, like in tower[degree].slices.items():
            acc = fnp.zeros_like(like)
            for vpart in _vector_partitions(int_part):
                term = _prod(get_block(block) for block in vpart)
                acc = acc + term * _vec_part_coef(vpart, divide_fac=False) * coef_fn(vpart)
            slices[int_part] = _symmetrize(acc, vec=int_part)
        out[degree] = _DSTensor(slices, autozero=True)
    return _DSTower(out)


@cache
def _pk_to_k_coef(vpart) -> float:
    def disconnected(tau):
        for block in tau:
            for i in range(len(vpart[0])):
                if sum(1 for j in block if vpart[j][i] > 0) > 1:
                    return False
        return True

    out = 0.0
    for tau in _set_partitions(len(vpart)):
        if disconnected(tau):
            out += (-1.0) ** (len(tau) - 1) * math.factorial(len(tau) - 1)
    return out


def _ds_pk_to_k(pk: _DSTower, strict: bool = True) -> _DSTower:
    return _ds_part_sum(pk, _pk_to_k_coef, strict=strict)


def _lap_m_dslice_scalar(dslice: fnp.ndarray, part: tuple[int, ...], m: int = 2):
    acc = 0.0
    for graph in _weak_compositions(len(part), m):
        if any(2 * graph[i] > part[i] for i in range(len(part))):
            continue
        coef = _multigraph_coef(tuple(((i, i), g) for i, g in enumerate(graph) if g), part)
        reduced = tuple(part[i] - 2 * graph[i] for i in range(len(part)))
        axes = tuple(i for i, value in enumerate(reduced) if value == 0)
        term = fnp.sum(dslice, axis=axes) if axes else dslice
        acc = acc + coef * term
    return acc * _int_partition_coef(part)


def _ds_harmonic_proj_r2(d_tensor: _DSTensor) -> _HTensor:
    n = d_tensor.n
    coef = _proj_coef(n, 4, 2)[2]
    core = 0.0
    for part, dslice in d_tensor.slices.items():
        core = core + coef * _lap_m_dslice_scalar(dslice, part, 2)
    return _HTensor(core, r=2, n=n)


def _diagslice(obj, part: tuple[int, ...], output_zero_repeated: bool = False):
    if hasattr(obj, "get_dslice"):
        out = obj.get_dslice(part)
    else:
        out = _tensor_dslice(obj, part)
    return _zero_repeated(out) if output_zero_repeated and hasattr(out, "ndim") else out


def _expand_dslice(obj, vec: tuple[int, ...], output_zero_repeated: bool = True):
    nonzero = tuple(i for i, value in enumerate(vec) if value > 0)
    part = tuple(vec[i] for i in nonzero)
    dslice = _diagslice(obj, part, output_zero_repeated=output_zero_repeated)
    return _expand(dslice, nonzero, len(vec))


def _eval_part(tower: dict[int, object], vec_part, dim: int, output_zero_repeated: bool = True):
    n = tower[1].n
    if any(sum(vec) not in tower for vec in vec_part):
        return None
    if not vec_part:
        return fnp.ones((n,) * dim)
    factors = [
        _expand_dslice(tower[sum(vec)], vec, output_zero_repeated=output_zero_repeated)
        for vec in vec_part
    ]
    return _vec_part_coef(vec_part, divide_fac=True) * _prod(factors)


def _multiply_wicks(value, k_vec, p_vec, wick_lookup):
    for axis, (k, p) in enumerate(zip(k_vec, p_vec)):
        wick = wick_lookup(int(k), int(p))
        shape = [1] * len(k_vec)
        shape[axis] = -1
        value = value * fnp.reshape(wick, tuple(shape))
    return value


class _FactoredThird:
    """Symmetric third cumulant stored as ``Sym(sum_r A_i,r B_j,r C_k,r)``."""

    def __init__(self, width: int, factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray] | None = None):
        self.width = width
        if factors is None:
            empty = fnp.zeros((width, 0))
            self.factors = (empty, empty, empty)
        else:
            self.factors = factors
        self._dslice_cache = {}

    @property
    def n(self) -> int:
        return self.width

    @property
    def ndim(self) -> int:
        return 3

    def clone(self) -> "_FactoredThird":
        return _FactoredThird(self.width, tuple(fnp.array(factor) for factor in self.factors))

    def contract_w(self, w: fnp.ndarray) -> "_FactoredThird":
        return _FactoredThird(self.width, tuple(w.T @ factor for factor in self.factors))

    def contract_wick(self, wick: fnp.ndarray) -> "_FactoredThird":
        return _FactoredThird(self.width, tuple(factor * wick[:, None] for factor in self.factors))

    def add_factors(self, factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]) -> "_FactoredThird":
        self.factors = tuple(
            fnp.concatenate((old, new), axis=1) for old, new in zip(self.factors, factors)
        )
        return self

    def diag(self) -> fnp.ndarray:
        if (3,) in self._dslice_cache:
            return self._dslice_cache[(3,)]
        a, b, c = self.factors
        if a.shape[1] == 0:
            out = fnp.zeros(self.width)
        else:
            out = fnp.sum(a * b * c, axis=1)
        self._dslice_cache[(3,)] = out
        return out

    def dslice_21(self) -> fnp.ndarray:
        """Return the ``(2, 1)`` diagonal slice, zeroing its own diagonal."""
        if (2, 1) in self._dslice_cache:
            return self._dslice_cache[(2, 1)]
        a, b, c = self.factors
        if a.shape[1] == 0:
            out = fnp.zeros((self.width, self.width))
        else:
            out = (
                (a * b) @ c.T
                + (a * c) @ b.T
                + (b * c) @ a.T
            ) / 3.0
            out = fnp.array(out)
            fnp.fill_diagonal(out, 0.0)
        self._dslice_cache[(2, 1)] = out
        return out

    def get_dslice(self, part: tuple[int, ...]) -> fnp.ndarray:
        sorted_part = tuple(sorted(part, reverse=True))
        if sorted_part == (3,):
            return self.diag()
        if sorted_part == (2, 1):
            value = self.dslice_21()
            if part == sorted_part:
                return value
            tmp = list(sorted_part)
            perm = []
            for block in part:
                idx = tmp.index(block)
                perm.append(idx)
                tmp[idx] = -1
            return fnp.transpose(value, tuple(perm))
        if sorted_part == (1, 1, 1):
            raise NotImplementedError("The all-distinct factored slice is not materialized")
        raise NotImplementedError(f"Unsupported third-cumulant slice {part}")

    def get_repeated(self) -> _DSTensor:
        return _DSTensor(
            {
                (3,): self.diag(),
                (2, 1): self.dslice_21(),
            },
            n=self.width,
            d=3,
            autozero=True,
        )

    def __add__(self, other: "_FactoredThird") -> "_FactoredThird":
        return _FactoredThird(
            self.width,
            tuple(fnp.concatenate((a, b), axis=1) for a, b in zip(self.factors, other.factors)),
        )

    @staticmethod
    def from_dstensor(ds: _DSTensor) -> "_FactoredThird":
        eye = fnp.eye(ds.n)
        k3 = ds.slices.get((3,), fnp.zeros(ds.n))
        k21 = ds.slices.get((2, 1), fnp.zeros((ds.n, ds.n)))
        factors = ((k3[:, None] * eye + k21.T * 3.0), eye, eye)
        return _FactoredThird(ds.n, factors)


def _dst_sub(a: _DSTensor, b: _DSTensor) -> _DSTensor:
    parts = set(a.slices) | set(b.slices)
    slices = {}
    for part in parts:
        av = a.slices.get(part)
        bv = b.slices.get(part)
        if av is None:
            slices[part] = -bv
        elif bv is None:
            slices[part] = av
        else:
            slices[part] = av - bv
    return _DSTensor(slices, n=a.n, d=a.d, autozero=True)


def _factored_nonlin_k3(wk: dict[int, object]) -> dict[int, object]:
    width = wk[1].n
    mean = wk[1].core
    var = fnp.maximum(fnp.diag(wk[2].core), _MIN_VARIANCE)

    @cache
    def wick(k: int, p: int):
        return _relu_wick(mean, var, k, p)

    p_slices = {}
    for int_part, vec_part, count in _all_terms_iso_k3():
        if (
            len(int_part) > 3
            or int_part in ((3, 1), (2, 1, 1))
            or int_part == (1, 1, 1)
            or (int_part, set(vec_part)) == ((2, 1, 1), {(1, 1, 1)})
        ):
            continue
        term = _eval_part(wk, vec_part, len(int_part), output_zero_repeated=True)
        if term is None:
            continue
        k_vec = _check_vec_partition(vec_part, len(int_part))
        term = count * _multiply_wicks(term, k_vec, int_part, wick)
        p_slices[int_part] = p_slices.get(int_part, 0.0) + term

    for int_part, value in list(p_slices.items()):
        p_slices[int_part] = _symmetrize(value, vec=int_part)

    w1 = wick(1, 1)
    w2 = wick(2, 1)
    w3 = wick(3, 1)
    w4 = wick(4, 1)

    wk_11 = _zero_repeated(wk[2].core)
    if 3 in wk:
        p111 = wk[3].clone().contract_wick(w1)
        wk_21 = _diagslice(wk[3], (2, 1), output_zero_repeated=True)
    else:
        p111 = _FactoredThird(width)
        wk_21 = fnp.zeros_like(wk_11)
    if 4 in wk:
        wk_22 = _diagslice(wk[4], (2, 2), output_zero_repeated=True)
    else:
        wk_22 = fnp.zeros_like(wk_11)

    wk_21 = wk_21 / 2.0
    wk_12 = wk_21.T
    wk_22 = wk_22 / 4.0
    eye = fnp.eye(width)

    fac1 = w1[:, None] * wk_11 + w2[:, None] * wk_21
    fac2 = eye * 3.0
    fac3 = (
        w2[:, None] * w1[None, :] * wk_11
        + w3[:, None] * w1[None, :] * wk_21
        + w2[:, None] * w2[None, :] * wk_12
        + w3[:, None] * w2[None, :] * wk_22
    ).T
    p111.add_factors((fac1, fac2, fac3))

    fac1 = w1[:, None] * wk_12 + w2[:, None] * wk_22
    fac2 = eye * 3.0
    fac3 = (
        w3[:, None] * w1[None, :] * wk_11
        + w4[:, None] * w1[None, :] * wk_21
        + w3[:, None] * w2[None, :] * wk_12
        + w4[:, None] * w2[None, :] * wk_22
    ).T
    p111.add_factors((fac1, fac2, fac3))

    if 4 in wk and wk[4].metric.ndim == 2:
        core = wk[4].core
        metric = wk[4].metric
        metric_diag = fnp.diag(metric)
        ones = fnp.ones_like(metric_diag)

        fac1 = w2[:, None] * (core * metric_diag)[:, None] * ones[None, :]
        fac2 = w1[:, None] * eye
        fac3 = w1[:, None] * metric / 2.0
        p111.add_factors((fac1, fac2, fac3))

        fac1 = w1[:, None] * metric * core
        fac2 = w2[:, None] * eye
        fac3 = w1[:, None] * metric
        p111.add_factors((fac1, fac2, fac3))

    p_ds = _DSTower.from_slices(p_slices, autozero=True)
    k_ds = _ds_pk_to_k(p_ds, strict=True)
    if 3 in k_ds:
        k_ds[3] = _dst_sub(k_ds[3], p111.get_repeated())

    out: dict[int, object] = {
        1: _HTensor(k_ds[1].to_tensor(), r=0),
        2: _HTensor(k_ds[2].to_tensor(), r=0),
        3: p111 + _FactoredThird.from_dstensor(k_ds[3]),
    }
    if 4 in k_ds:
        out[4] = _ds_harmonic_proj_r2(k_ds[4])
    return out


def _factorized_k3_propagation(mlp: MLP) -> fnp.ndarray:
    """K=3 factorized cumulant propagation for ReLU hidden layers.

    This ports the upstream factorized K=3 path into the narrower whestbench
    setting: square ReLU MLPs, standard normal inputs, no biases, and per-hidden
    layer activation means as output.
    """
    width = mlp.width
    tower: dict[int, object] = {
        1: _HTensor(fnp.zeros(width), r=0),
        2: _HTensor(fnp.eye(width), r=0),
    }

    rows = []
    for w_mat in mlp.weights:
        wk = {}
        for degree, value in tower.items():
            wk[degree] = value.contract_w(w_mat)
        tower = _factored_nonlin_k3(wk)
        rows.append(tower[1].core)

    return fnp.stack(rows, axis=0)


def _antithetic_sample_means(mlp: MLP, n_samples: int, rng: fnp.random.Generator) -> fnp.ndarray:
    """Return batched Monte-Carlo layer means with paired x and -x inputs."""
    half = max(n_samples // 2, 1)
    x_half = rng.standard_normal((half, mlp.width))
    x = fnp.concatenate((x_half, -x_half), axis=0)
    x_std = fnp.sqrt(fnp.maximum(fnp.mean(x * x, axis=0), _MIN_VARIANCE))
    x = x / x_std

    rows = []
    for w in mlp.weights:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _mean_propagation(mlp: MLP) -> fnp.ndarray:
    """K=1 mean propagation (Algorithm 1 from the paper).

    Tracks the mean vector and a scalar average variance (trace of covariance)
    through each layer.  In practice we keep the full diagonal variance because
    it has the same asymptotic cost and strictly lower error than tracking only
    the trace.
    """
    width = mlp.width
    mu = fnp.zeros(width)
    var = fnp.ones(width)

    rows = []
    for w in mlp.weights:
        # Linear layer
        mu_pre = w.T @ mu
        var_pre = (w * w).T @ var

        # ReLU
        mu, ez2, _ = _relu_moments(mu_pre, var_pre)
        var = fnp.maximum(ez2 - mu * mu, 0.0)

        rows.append(mu)

    return fnp.stack(rows, axis=0)


def _covariance_propagation(mlp: MLP) -> fnp.ndarray:
    """K=2 covariance propagation (Algorithm 2 from the paper).

    Tracks the full mean vector and covariance matrix through each layer.
    The post-ReLU covariance is approximated with the leading-order Hermite
    term (gain method) while the diagonal is replaced by the exact marginal
    variance computed from power cumulants.
    """
    width = mlp.width
    mu = fnp.zeros(width)
    cov = fnp.eye(width)

    rows = []
    for w in mlp.weights:
        # Linear layer
        mu_pre = w.T @ mu
        # Use einsum so flopscope sees the symmetry of the two w operands.
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

        # Extract marginal variances
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-30)

        # ReLU moments
        mu_post, ez2, gain = _relu_moments(mu_pre, var_pre)
        var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)

        # Off-diagonal approximation: cov_post[i,j] ≈ gain[i]*gain[j]*cov_pre[i,j]
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)

        mu = mu_post
        rows.append(mu)

    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Adaptive cumulant-propagation estimator.

    Chooses between covariance propagation (K=2) and mean propagation (K=1)
    based on the FLOP budget.  Both are sample-free estimators from the paper.
    """

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer mean activations.

        Returns an array of shape ``(depth, width)``.
        """
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng

        width = mlp.width
        depth = mlp.depth

        # Rough FLOP estimates, calibrated against flopscope's measured counts.
        cov_flops = 3.05 * depth * width * width * width
        # Full factorized K=3 includes the factored third cumulant plus all
        # non-factored diagonal-slice power-cumulant conversion terms. The
        # factor rank grows by layer, so a conservative score-floor router uses
        # a depth^2 term.
        factor_k3_flops = 50 * depth * depth * width * width * width

        # Prefer the full factorized algorithm when it fits inside the actual
        # benchmark budget. On the contest-size MLP its final-layer MSE drop is
        # large enough to beat the score-floor K=2+sampling route despite the
        # higher compute multiplier.
        if _ENABLE_FACTOR_K3 and factor_k3_flops < budget * 0.8 and width >= _K3_MIN_WIDTH:
            return _factorized_k3_propagation(mlp)

        # Use covariance propagation if it fits inside the budget with some
        # headroom for overhead.  Otherwise fall back to mean propagation.
        if cov_flops < budget * 0.8:
            estimate = _covariance_propagation(mlp)

            # The benchmark score multiplier bottoms out at 10% of budget.
            # Spend otherwise-free headroom on a small unbiased correction, but
            # avoid it when the budget is too tight to get useful samples.
            sampling_flops_per_input = max(1, 2 * depth * width * width)
            spare_for_floor = int(max(0.0, budget * 0.098 - cov_flops))
            n_samples = min(100_000, spare_for_floor // sampling_flops_per_input)
            if n_samples >= 2_048:
                rng = fnp.random.default_rng(mlp.seed)
                sampled = _antithetic_sample_means(mlp, int(n_samples), rng)
                sample_weight = min(0.5, n_samples / (n_samples + 5_000.0))
                estimate = estimate * (1.0 - sample_weight) + sampled * sample_weight
            return estimate

        return _mean_propagation(mlp)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
