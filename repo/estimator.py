"""Optimized cumulant-propagation estimator for ReLU MLPs.

Submission for https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026.

Implements the low-order cumulant propagation family from Wu et al.,
"Estimating the expected output of wide random MLPs more efficiently than
sampling" (arXiv:2605.05179), specialized to the WhestBench setting.

The estimator uses the optimized factorized K=3 path with r=1 degree-4
harmonic tracking. It avoids materializing dense order-3/order-4 tensors by
carrying factored third-cumulant terms, cached diagonal slices, harmonic
projections, and a diagonal-only final-layer mean specialization.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import math
import os
from collections import defaultdict
from functools import cache
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

if hasattr(flops, "configure"):
    flops.configure(symmetry_warnings=False)

_MIN_VARIANCE = 1e-30
_FACTOR_K3_MODE = "r1"
_AUGMENTED_FACTOR_K3_MODE = "r1_slices_k211"
_DEEP_HADAMARD_MIN_DEPTH = 16
_DEEP_HADAMARD_BLOCKS = 12
_DEFAULT_R1_COMPRESSED_SCHEDULE = (768, 1024, 1280, 1536, 1536, 1536, 1536)


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


def _relu_wick_from_stats(
    mean: fnp.ndarray,
    var: fnp.ndarray,
    sigma: fnp.ndarray,
    alpha: fnp.ndarray,
    phi: fnp.ndarray,
    Phi: fnp.ndarray,
    k: int,
    p: int = 1,
) -> fnp.ndarray:
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
        return math.factorial(p) * _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, k - p + 1, 1)

    if p == 1:
        if k == 0:
            return sigma * phi + mean * Phi
        if k == 1:
            return Phi
        return (-1.0) ** (k - 2) * sigma ** (-(k - 1)) * _hermite_prob(k - 2, alpha) * phi

    raise ValueError(f"unsupported ReLU Wick coefficient p={p}")


def _relu_wick(mean: fnp.ndarray, var: fnp.ndarray, k: int, p: int = 1) -> fnp.ndarray:
    """Return ``E[d^k ReLU(Z)^p / dZ^k]`` for ``Z ~ N(mean, var)``.

    This is the ReLU Wick-coefficient helper used by the factorized algorithm.
    Only the orders needed by the K=3 factorized propagation are implemented.
    """
    sigma = fnp.sqrt(fnp.maximum(var, _MIN_VARIANCE))
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    return _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, k, p)

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


@cache
def _terms_iso_k3_for_mode(augment_mode: str):
    include_aug_slices = augment_mode in {"r1_slices", "r1_slices_111", "r1_slices_k211_only", "r1_slices_k211", "full"}
    skip_degree4 = augment_mode == "r1_no4"
    terms = []
    for int_part, vec_part, count in _all_terms_iso_k3():
        if (
            len(int_part) > 3
            or (skip_degree4 and sum(int_part) > 3)
            or (not include_aug_slices and int_part in ((3, 1), (2, 1, 1)))
            or int_part == (1, 1, 1)
            or (int_part, set(vec_part)) == ((2, 1, 1), {(1, 1, 1)})
        ):
            continue
        factors = []
        for vec in vec_part:
            nonzero = tuple(i for i, value in enumerate(vec) if value > 0)
            factors.append((sum(vec), vec, nonzero, tuple(vec[i] for i in nonzero)))
        if skip_degree4 and any(degree > 3 for degree, *_ in factors):
            continue
        terms.append((
            int_part,
            vec_part,
            count,
            _check_vec_partition(vec_part, len(int_part)),
            len(int_part),
            _vec_part_coef(vec_part, divide_fac=True),
            tuple(factors),
        ))
    return tuple(terms)


@cache
def _terms_iso_k3_grouped_for_mode(augment_mode: str):
    grouped = defaultdict(list)
    for term in _terms_iso_k3_for_mode(augment_mode):
        grouped[term[0]].append(term)
    return tuple((int_part, tuple(terms)) for int_part, terms in grouped.items())


@cache
def _zero_repeated_mask(shape: tuple[int, ...]) -> fnp.ndarray:
    idxs = fnp.indices(shape)
    mask = fnp.ones(shape)
    for i in range(len(shape)):
        for j in range(i + 1, len(shape)):
            mask = mask * (idxs[i] != idxs[j])
    return mask


@cache
def _eye(width: int) -> fnp.ndarray:
    return fnp.eye(width)


@cache
def _ones(width: int) -> fnp.ndarray:
    return fnp.ones(width)


@cache
def _zeros_vec(width: int) -> fnp.ndarray:
    return fnp.zeros(width)


@cache
def _empty_factor(width: int) -> fnp.ndarray:
    return fnp.zeros((width, 0))


@cache
def _idx(width: int) -> fnp.ndarray:
    return fnp.arange(width)


@cache
def _hadamard(width: int) -> fnp.ndarray:
    if width < 1 or width & (width - 1):
        raise ValueError(f"Hadamard cubature requires power-of-two width; got {width}")
    rows = [[1.0]]
    while len(rows) < width:
        rows = [row + row for row in rows] + [row + [-value for value in row] for row in rows]
    return fnp.array(rows)


def _zero_repeated(a: fnp.ndarray) -> fnp.ndarray:
    if a.ndim <= 1:
        return a
    if a.ndim == 2:
        out = fnp.array(a)
        fnp.fill_diagonal(out, 0.0)
        return out
    mask = _zero_repeated_mask(tuple(a.shape))
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
    if len(perms) == 1:
        return a
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


def _harmonic_dslice_general(core: fnp.ndarray, metric: fnp.ndarray, r: int, part: tuple[int, ...]) -> fnp.ndarray:
    n = metric.shape[0]
    s = core.ndim if hasattr(core, "ndim") else 0
    out = fnp.zeros((n,) * len(part))
    out_labels = list("abcd"[: len(part)])
    core_base = list("wxyz"[:s])
    for graph in _multigraphs(len(part), r):
        coef = _multigraph_coef(graph, part, lap_coef=False)
        if coef == 0.0:
            continue
        core_labels = list(core_base)
        metric_labels = [["m", "n"] for _ in range(r)]
        used = [0] * len(part)
        edge_idx = 0
        for (a, b), mult in graph:
            for _ in range(mult):
                metric_labels[edge_idx] = [out_labels[a], out_labels[b]]
                used[a] += 1
                used[b] += 1
                edge_idx += 1
        core_idx = 0
        for axis, block_size in enumerate(part):
            while used[axis] < block_size:
                core_labels[core_idx] = out_labels[axis]
                used[axis] += 1
                core_idx += 1
        factors = []
        labels = []
        if s > 0:
            factors.append(core)
            labels.append(" ".join(core_labels))
        factors.extend(metric for _ in range(r))
        labels.extend(" ".join(labels_i) for labels_i in metric_labels)
        expr = ", ".join(labels) + " -> " + " ".join(out_labels)
        term = fnp.einsum(expr, *factors) if factors else core
        out = out + coef * term
    return _zero_repeated(out)


class _HTensor:
    def __init__(self, core: fnp.ndarray | float, r: int = 0, n: int | None = None, metric=None):
        self.core = core
        self.r = r
        self.n = n if n is not None else core.shape[0]
        if r == 0:
            self.metric = None
        elif metric is None:
            self.metric = _eye(self.n)
        elif not hasattr(metric, "ndim") or metric.ndim == 0:
            self.metric = _eye(self.n) * metric
        elif metric.ndim == 1:
            self.metric = fnp.diag(metric)
        else:
            self.metric = metric
        self._dslice_cache = {}

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
        metric = w_t @ self.metric @ w if self.r > 0 else None
        return _HTensor(core, r=self.r, n=w.shape[1], metric=metric)

    def get_dslice(self, part: tuple[int, ...]) -> fnp.ndarray:
        cache_key = tuple(part)
        if cache_key in self._dslice_cache:
            return self._dslice_cache[cache_key]
        if self.r == 0:
            out = _tensor_dslice(self.core, part)
        elif self.r == 2 and (not hasattr(self.core, "ndim") or self.core.ndim == 0):
            out = _harmonic_dslice_r2_scalar(self.core, self.metric, part)
        elif self.r == 1 and hasattr(self.core, "ndim") and self.core.ndim == 2:
            out = _harmonic_dslice_general(self.core, self.metric, self.r, part)
        else:
            raise NotImplementedError("Unsupported K=3 harmonic tensor diagonal slice")
        self._dslice_cache[cache_key] = out
        return out


def _tensor_dslice(a: fnp.ndarray, part: tuple[int, ...], output_zero_repeated: bool = False) -> fnp.ndarray:
    if a.ndim == 1:
        out = a
    elif a.ndim == 2:
        out = fnp.diag(a) if part == (2,) else a
    elif a.ndim == 3:
        idx = _idx(a.shape[0])
        if part == (3,):
            out = a[idx, idx, idx]
        elif part == (2, 1):
            out = a[idx[:, None], idx[:, None], idx[None, :]]
        else:
            out = a
    elif a.ndim == 4:
        idx = _idx(a.shape[0])
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
                idx = _idx(self.n)
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
    block_cache = {}

    def get_block(block):
        nonzero = tuple(i for i, value in enumerate(block) if value > 0)
        part = tuple(block[i] for i in nonzero)
        if not part:
            return 1.0
        key = (part, nonzero, len(block))
        if key in block_cache:
            return block_cache[key]
        try:
            out = _expand(tower.get_slice(part, strict=True), nonzero, len(block))
        except KeyError:
            if strict:
                raise
            out = 0.0
        block_cache[key] = out
        return out

    out = {}
    for degree in range(1, max(tower.keys()) + 1):
        if degree not in tower:
            continue
        slices = {}
        for int_part, like in tower[degree].slices.items():
            acc = fnp.zeros_like(like)
            for vpart in _vector_partitions(int_part):
                coef = _vec_part_coef(vpart, divide_fac=False) * coef_fn(vpart)
                term = None
                for block in vpart:
                    factor = get_block(block)
                    if term is None:
                        term = coef * factor if coef != 1.0 else factor
                    else:
                        term = term * factor
                if term is None:
                    term = coef
                acc = acc + term
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
        term = dslice
        for axis in reversed(axes):
            term = fnp.sum(term, axis=axis)
        acc = acc + coef * term
    return acc * _int_partition_coef(part)


def _embed_dslice(dslice: fnp.ndarray, part: tuple[int, ...], d_out: int, n: int):
    if d_out == 0:
        return dslice
    if d_out == 1:
        return dslice
    if d_out == 2:
        if part == (2,):
            return fnp.diag(dslice)
        return dslice
    raise NotImplementedError("Only degree-4, r=1 projection is needed for K=3 augment")


def _lap_m_dslice_tensor(dslice: fnp.ndarray, part: tuple[int, ...], m: int, n: int):
    d_out = sum(part) - 2 * m
    acc = 0.0 if d_out == 0 else fnp.zeros((n,) * d_out)
    for graph in _weak_compositions(len(part), m):
        if any(2 * graph[i] > part[i] for i in range(len(part))):
            continue
        coef = _multigraph_coef(tuple(((i, i), g) for i, g in enumerate(graph) if g), part)
        reduced = tuple(part[i] - 2 * graph[i] for i in range(len(part)))
        axes = tuple(i for i, value in enumerate(reduced) if value == 0)
        term = dslice
        for axis in reversed(axes):
            term = fnp.sum(term, axis=axis)
        reduced_part = tuple(value for value in reduced if value > 0)
        acc = acc + coef * _embed_dslice(term, reduced_part, d_out, n)
    if d_out >= 2:
        acc = _symmetrize(acc)
    return acc * _int_partition_coef(part)


def _ds_harmonic_proj_r2(d_tensor: _DSTensor) -> _HTensor:
    n = d_tensor.n
    coef = _proj_coef(n, 4, 2)[2]
    core = 0.0
    for part, dslice in d_tensor.slices.items():
        core = core + coef * _lap_m_dslice_scalar(dslice, part, 2)
    return _HTensor(core, r=2, n=n)


def _ds_harmonic_proj_r1(d_tensor: _DSTensor) -> _HTensor:
    n = d_tensor.n
    p1 = _proj_coef(n, 4, 1)
    p2 = _proj_coef(n, 4, 2)
    coef_l1 = p1[1]
    coef_l2 = p1[2] + p2[2]
    eye = _eye(n)
    core = fnp.zeros((n, n))
    for part, dslice in d_tensor.slices.items():
        core = core + coef_l1 * _lap_m_dslice_tensor(dslice, part, 1, n)
        core = core + coef_l2 * _lap_m_dslice_scalar(dslice, part, 2) * eye
    return _HTensor(core, r=1, n=n)


def _diagslice(obj, part: tuple[int, ...], output_zero_repeated: bool = False):
    if hasattr(obj, "get_dslice"):
        out = obj.get_dslice(part)
    else:
        out = _tensor_dslice(obj, part)
    if output_zero_repeated and getattr(obj, "_repeated_slices_zeroed", False):
        return out
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


def _dslice_21_diag_middle(a: fnp.ndarray, diag_b: fnp.ndarray, c: fnp.ndarray) -> fnp.ndarray:
    diag_a = fnp.diag(a)
    diag_c = fnp.diag(c)
    out = (
        (diag_a * diag_b)[:, None] * c.T
        + a * c * diag_b[None, :]
        + (diag_b * diag_c)[:, None] * a.T
    ) / 3.0
    return _zero_repeated(out)


def _diag_diag_middle(a: fnp.ndarray, diag_b: fnp.ndarray, c: fnp.ndarray) -> fnp.ndarray:
    return fnp.diag(a) * diag_b * fnp.diag(c)


class _DiagFactor:
    def __init__(self, diag: fnp.ndarray):
        self.diag = diag
        self.shape = (diag.shape[0], diag.shape[0])


def _diag_factor(diag: fnp.ndarray | float, width: int | None = None) -> _DiagFactor:
    if hasattr(diag, "shape"):
        return _DiagFactor(diag)
    if width is None:
        raise ValueError("width is required for scalar diagonal factors")
    return _DiagFactor(_ones(width) * diag)


def _materialize_factor(factor):
    if isinstance(factor, _DiagFactor):
        return fnp.diag(factor.diag)
    return factor


def _copy_factor(factor):
    if isinstance(factor, _DiagFactor):
        return _DiagFactor(fnp.array(factor.diag))
    return fnp.array(factor)


def _scale_factor_rows(factor, scale: fnp.ndarray):
    if isinstance(factor, _DiagFactor):
        return _DiagFactor(factor.diag * scale)
    return factor * scale[:, None]


def _contract_factor_w(factor, w: fnp.ndarray):
    if isinstance(factor, _DiagFactor):
        return w.T * factor.diag[None, :]
    return w.T @ factor


def _factor_rank(factor) -> int:
    return len(factor.diag) if isinstance(factor, _DiagFactor) else factor.shape[1]


def _slice_factor_columns(factor, start: int, stop: int):
    if isinstance(factor, _DiagFactor):
        return _materialize_factor(factor)[:, start:stop]
    return factor[:, start:stop]


def _take_factor_columns(factor, columns: fnp.ndarray):
    if isinstance(factor, _DiagFactor):
        return _materialize_factor(factor)[:, columns]
    return factor[:, columns]


def _factor_column_norms_sq(factor) -> fnp.ndarray:
    if isinstance(factor, _DiagFactor):
        return factor.diag * factor.diag
    return fnp.sum(factor * factor, axis=0)


class _FactoredThird:
    """Symmetric third cumulant stored as ``Sym(sum_r A_i,r B_j,r C_k,r)``."""

    _repeated_slices_zeroed = True

    def __init__(self, width: int, factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray] | None = None):
        self.width = width
        self._groups: tuple[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray], ...] = (
            () if factors is None or factors[0].shape[1] == 0 else (factors,)
        )
        self._factor_cache = None
        self._dslice_cache = {}

    @property
    def n(self) -> int:
        return self.width

    @property
    def ndim(self) -> int:
        return 3

    @property
    def factors(self) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
        if self._factor_cache is not None:
            return self._factor_cache
        if not self._groups:
            empty = _empty_factor(self.width)
            self._factor_cache = (empty, empty, empty)
            return self._factor_cache
        self._factor_cache = tuple(
            fnp.concatenate(tuple(_materialize_factor(group[axis]) for group in self._groups), axis=1)
            for axis in range(3)
        )
        return self._factor_cache

    def _replace_groups(self, groups: tuple[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray], ...]) -> "_FactoredThird":
        out = _FactoredThird(self.width)
        out._groups = groups
        out._factor_cache = None
        return out

    @staticmethod
    def _map_unique_factors(
        groups: tuple[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray], ...],
        transform,
    ) -> tuple[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray], ...]:
        mapped = {}
        out_groups = []
        for group in groups:
            out_group = []
            for factor in group:
                key = id(factor)
                value = mapped.get(key)
                if value is None:
                    value = transform(factor)
                    mapped[key] = value
                out_group.append(value)
            out_groups.append(tuple(out_group))
        return tuple(out_groups)

    @staticmethod
    def _contract_groups(
        groups: tuple[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray], ...],
        w: fnp.ndarray,
    ) -> tuple[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray], ...]:
        if not groups:
            return ()
        unique = {}
        dense_ordered = []
        dense_keys = []
        diag_keys = []
        for group in groups:
            for factor in group:
                key = id(factor)
                if key not in unique:
                    unique[key] = factor
                    if isinstance(factor, _DiagFactor):
                        diag_keys.append(key)
                    else:
                        dense_keys.append(key)
                        dense_ordered.append(factor)
        mapped = {}
        if dense_ordered:
            slab = fnp.concatenate(tuple(dense_ordered), axis=1)
            contracted = w.T @ slab
            dense_start = 0
            for key in dense_keys:
                factor = unique[key]
                width = factor.shape[1]
                mapped[key] = contracted[:, dense_start : dense_start + width]
                dense_start += width
        for key in diag_keys:
            factor = unique[key]
            mapped[key] = _contract_factor_w(factor, w)
        return tuple(tuple(mapped[id(factor)] for factor in group) for group in groups)

    def clone(self) -> "_FactoredThird":
        out = _FactoredThird(self.width)
        out._groups = tuple(
            tuple(_copy_factor(factor) for factor in group)
            for group in self._groups
        )
        out._factor_cache = None
        return out

    def contract_w(self, w: fnp.ndarray) -> "_FactoredThird":
        return self._replace_groups(self._contract_groups(self._groups, w))

    @property
    def rank(self) -> int:
        return sum(_factor_rank(group[0]) for group in self._groups)

    def keep_recent_rank(self, rank_cap: int) -> "_FactoredThird":
        if rank_cap <= 0:
            return _FactoredThird(self.width)
        total_rank = self.rank
        if total_rank <= rank_cap:
            return self

        remaining = rank_cap
        groups = []
        for group in reversed(self._groups):
            group_rank = _factor_rank(group[0])
            if remaining <= 0:
                break
            if group_rank <= remaining:
                groups.append(group)
                remaining -= group_rank
            else:
                start = group_rank - remaining
                groups.append(tuple(_slice_factor_columns(factor, start, group_rank) for factor in group))
                remaining = 0

        out = _FactoredThird(self.width)
        out._groups = tuple(reversed(groups))
        return out

    def keep_top_rank(self, rank_cap: int) -> "_FactoredThird":
        if rank_cap <= 0:
            return _FactoredThird(self.width)
        total_rank = self.rank
        if total_rank <= rank_cap:
            return self

        factors = self.factors
        scores = _ones(total_rank)
        for factor in factors:
            scores = scores * fnp.sum(factor * factor, axis=0)
        keep = fnp.argsort(scores)[-rank_cap:]
        return _FactoredThird(self.width, tuple(factor[:, keep] for factor in factors))

    def keep_top_group_rank(self, rank_cap: int) -> "_FactoredThird":
        if rank_cap <= 0:
            return _FactoredThird(self.width)
        total_rank = self.rank
        if total_rank <= rank_cap:
            return self

        scored_groups = []
        for group_idx, group in enumerate(self._groups):
            scores = _ones(_factor_rank(group[0]))
            for factor in group:
                scores = scores * _factor_column_norms_sq(factor)
            scored_groups.append((fnp.sum(scores), group_idx, group))

        remaining = rank_cap
        selected = []
        for _, group_idx, group in sorted(scored_groups, key=lambda item: float(item[0]), reverse=True):
            group_rank = _factor_rank(group[0])
            if group_rank <= remaining:
                selected.append((group_idx, group))
                remaining -= group_rank
            if remaining <= 0:
                break

        if not selected:
            best_group = max(scored_groups, key=lambda item: float(item[0]))[2]
            group_rank = _factor_rank(best_group[0])
            sliced = tuple(_slice_factor_columns(factor, 0, min(rank_cap, group_rank)) for factor in best_group)
            return _FactoredThird(self.width, sliced)

        groups = tuple(group for _, group in sorted(selected, key=lambda item: item[0]))
        return self._replace_groups(groups)

    def keep_top_group_boundary_rank(self, rank_cap: int) -> "_FactoredThird":
        if rank_cap <= 0:
            return _FactoredThird(self.width)
        total_rank = self.rank
        if total_rank <= rank_cap:
            return self

        scored_groups = []
        for group_idx, group in enumerate(self._groups):
            scores = _ones(_factor_rank(group[0]))
            for factor in group:
                scores = scores * _factor_column_norms_sq(factor)
            scored_groups.append((fnp.sum(scores), group_idx, group, scores))

        remaining = rank_cap
        selected = []
        skipped = []
        for group_score, group_idx, group, scores in sorted(
            scored_groups, key=lambda item: float(item[0]), reverse=True
        ):
            group_rank = _factor_rank(group[0])
            if group_rank <= remaining:
                selected.append((group_idx, group))
                remaining -= group_rank
            else:
                skipped.append((group_score, group_idx, group, scores))
            if remaining <= 0:
                break

        if remaining > 0 and skipped:
            _, group_idx, group, scores = max(skipped, key=lambda item: float(item[0]))
            keep = fnp.argsort(scores)[-remaining:]
            selected.append((group_idx, tuple(_take_factor_columns(factor, keep) for factor in group)))

        groups = tuple(group for _, group in sorted(selected, key=lambda item: item[0]))
        return self._replace_groups(groups)

    def contract_wick(self, wick: fnp.ndarray, propagate_cache: bool = True) -> "_FactoredThird":
        out = self._replace_groups(
            self._map_unique_factors(self._groups, lambda factor: _scale_factor_rows(factor, wick))
        )
        if propagate_cache and (2, 1) in self._dslice_cache:
            out._dslice_cache[(2, 1)] = self._dslice_cache[(2, 1)] * (wick[:, None] * wick[:, None] * wick[None, :])
        if propagate_cache and (3,) in self._dslice_cache:
            out._dslice_cache[(3,)] = self._dslice_cache[(3,)] * wick * wick * wick
        return out

    def add_factors(
        self,
        factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray],
        dslice_21_increment: fnp.ndarray | None = None,
        diag_increment: fnp.ndarray | None = None,
    ) -> "_FactoredThird":
        if factors[0].shape[1] != 0:
            self._groups = self._groups + (factors,)
            self._factor_cache = None
        if (2, 1) in self._dslice_cache and dslice_21_increment is not None:
            self._dslice_cache[(2, 1)] = self._dslice_cache[(2, 1)] + dslice_21_increment
        else:
            self._dslice_cache.pop((2, 1), None)
        if (3,) in self._dslice_cache and diag_increment is not None:
            self._dslice_cache[(3,)] = self._dslice_cache[(3,)] + diag_increment
        else:
            self._dslice_cache.pop((3,), None)
        return self

    def add_factor_groups(
        self,
        factor_groups: list[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]],
        dslice_21_increments: list[fnp.ndarray | None],
        diag_increments: list[fnp.ndarray | None],
    ) -> "_FactoredThird":
        if not factor_groups:
            return self
        self._groups = self._groups + tuple(factor_groups)
        self._factor_cache = None
        if (2, 1) in self._dslice_cache and all(value is not None for value in dslice_21_increments):
            for value in dslice_21_increments:
                self._dslice_cache[(2, 1)] = self._dslice_cache[(2, 1)] + value
        else:
            self._dslice_cache.pop((2, 1), None)
        if (3,) in self._dslice_cache and all(value is not None for value in diag_increments):
            for value in diag_increments:
                self._dslice_cache[(3,)] = self._dslice_cache[(3,)] + value
        else:
            self._dslice_cache.pop((3,), None)
        return self

    def diag(self) -> fnp.ndarray:
        if (3,) in self._dslice_cache:
            return self._dslice_cache[(3,)]
        if not self._groups:
            out = _zeros_vec(self.width)
        else:
            out = 0.0
            for a, b, c in self._groups:
                a = _materialize_factor(a)
                b = _materialize_factor(b)
                c = _materialize_factor(c)
                out = out + fnp.sum(a * b * c, axis=1)
        self._dslice_cache[(3,)] = out
        return out

    def dslice_21(self) -> fnp.ndarray:
        """Return the ``(2, 1)`` diagonal slice, zeroing its own diagonal."""
        if (2, 1) in self._dslice_cache:
            return self._dslice_cache[(2, 1)]
        if not self._groups:
            out = fnp.zeros((self.width, self.width))
        else:
            out = 0.0
            middle_terms = {}
            for a, b, c in self._groups:
                same_bc = b is c
                b_key = id(b)
                a = _materialize_factor(a)
                b = _materialize_factor(b)
                c = _materialize_factor(c)
                ab_c = (a * b) @ c.T
                out = out + ab_c
                if same_bc:
                    out = out + ab_c
                else:
                    middle_terms.setdefault(b_key, [b, 0.0])
                    middle_terms[b_key][1] = middle_terms[b_key][1] + a * c
                out = out + (b * c) @ a.T
            for b, ac_sum in middle_terms.values():
                out = out + ac_sum @ b.T
            out = out / 3.0
            out = fnp.array(out)
            fnp.fill_diagonal(out, 0.0)
        self._dslice_cache[(2, 1)] = out
        return out

    def contracted_diag(self, w: fnp.ndarray) -> fnp.ndarray:
        if not self._groups:
            return _zeros_vec(w.shape[1])
        out = 0.0
        for a, b, c in self._contract_groups(self._groups, w):
            out = out + fnp.sum(a * b * c, axis=1)
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
            autozero=False,
        )

    def __add__(self, other: "_FactoredThird") -> "_FactoredThird":
        out = _FactoredThird(self.width)
        out._groups = self._groups + other._groups
        out._factor_cache = None
        return out

    @staticmethod
    def from_dstensor(ds: _DSTensor) -> "_FactoredThird":
        eye = _eye(ds.n)
        k3 = ds.slices.get((3,), _zeros_vec(ds.n))
        k21 = ds.slices.get((2, 1), fnp.zeros((ds.n, ds.n)))
        diag_eye = _diag_factor(1.0, ds.n)
        factors = ((k3[:, None] * eye + k21.T * 3.0), diag_eye, diag_eye)
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


def _factored_nonlin_k3_r1_fast(wk: dict[int, object]) -> dict[int, object]:
    width = wk[1].n
    mean = wk[1].core
    var = fnp.maximum(fnp.diag(wk[2].core), _MIN_VARIANCE)
    sigma = fnp.sqrt(var)
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    wick_values: dict[tuple[int, int], fnp.ndarray] = {}

    def wick(k: int, p: int):
        key = (k, p)
        value = wick_values.get(key)
        if value is None:
            if p > 1 and k >= p:
                value = math.factorial(p) * wick(k - p + 1, 1)
            else:
                value = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, k, p)
            wick_values[key] = value
        return value

    wick_views: dict[tuple[int, int, int, int], fnp.ndarray] = {}

    def wick_view(k: int, p: int, dim: int, axis: int):
        key = (k, p, dim, axis)
        value = wick_views.get(key)
        if value is None:
            shape = [1] * dim
            shape[axis] = -1
            value = fnp.reshape(wick(k, p), tuple(shape))
            wick_views[key] = value
        return value

    wick_term_views: dict[tuple[tuple[int, ...], tuple[int, ...], int, int], fnp.ndarray] = {}

    def wick_term_view(int_part: tuple[int, ...], k_vec, dim: int, count: int):
        key = (int_part, tuple(int(k) for k in k_vec), dim, count)
        value = wick_term_views.get(key)
        if value is None:
            value = None
            for axis, (k, p) in enumerate(zip(k_vec, int_part)):
                view = wick_view(int(k), int(p), dim, axis)
                if value is None:
                    value = count * view if count != 1 else view
                else:
                    value = value * view
            wick_term_views[key] = value
        return value

    dslice_cache = {
        (2, (1, 1)): _zero_repeated(wk[2].core),
        (2, (2,)): fnp.diag(wk[2].core),
    }
    if 3 in wk:
        dslice_cache[(3, (3,))] = _diagslice(wk[3], (3,), output_zero_repeated=True)
        dslice_cache[(3, (2, 1))] = _diagslice(wk[3], (2, 1), output_zero_repeated=True)
        dslice_cache[(3, (1, 2))] = fnp.transpose(dslice_cache[(3, (2, 1))])
        dslice_cache[(3, (3, 0))] = dslice_cache[(3, (3,))]
        dslice_cache[(3, (0, 3))] = dslice_cache[(3, (3,))]
    if 4 in wk:
        dslice_cache[(4, (4,))] = _diagslice(wk[4], (4,), output_zero_repeated=True)
        dslice_cache[(4, (2, 2))] = _diagslice(wk[4], (2, 2), output_zero_repeated=True)
        dslice_cache[(4, (3, 1))] = _diagslice(wk[4], (3, 1), output_zero_repeated=True)
        dslice_cache[(4, (1, 3))] = fnp.transpose(dslice_cache[(4, (3, 1))])
        dslice_cache[(4, (4, 0))] = dslice_cache[(4, (4,))]
        dslice_cache[(4, (0, 4))] = dslice_cache[(4, (4,))]

    eval_cache = {}

    def eval_term(vec_part, dim: int, coef: float, factors):
        key = (vec_part, dim)
        value = eval_cache.get(key, ...)
        if value is not ...:
            return value
        if any((degree, part) not in dslice_cache for degree, _, _, part in factors):
            eval_cache[key] = None
            return None
        if not factors:
            value = fnp.ones((width,) * dim)
        else:
            value = None
            for degree, _, nonzero, part in factors:
                factor = _expand(dslice_cache[(degree, part)], nonzero, dim)
                if value is None:
                    value = coef * factor if coef != 1.0 else factor
                else:
                    value = value * factor
        eval_cache[key] = value
        return value

    p_slices = {}
    for int_part, terms in _terms_iso_k3_grouped_for_mode("r1"):
        acc = None
        for _, vec_part, count, k_vec, dim, coef, factors in terms:
            term = eval_term(vec_part, dim, coef, factors)
            if term is None:
                continue
            term = term * wick_term_view(int_part, k_vec, dim, count)
            acc = term if acc is None else acc + term
        p_slices[int_part] = _symmetrize(acc, vec=int_part)

    w1 = wick(1, 1)
    w2 = wick(2, 1)
    w3 = wick(3, 1)
    w4 = wick(4, 1)

    wk_11 = dslice_cache[(2, (1, 1))]
    if 3 in wk:
        wk_21_raw = dslice_cache[(3, (2, 1))]
        wk_3 = dslice_cache[(3, (3,))]
        p111 = wk[3].contract_wick(w1, propagate_cache=False)
        p111._dslice_cache[(2, 1)] = wk_21_raw * (w1[:, None] * w1[:, None] * w1[None, :])
        p111._dslice_cache[(3,)] = wk_3 * w1 * w1 * w1
    else:
        p111 = _FactoredThird(width)
        p111._dslice_cache[(2, 1)] = fnp.zeros_like(wk_11)
        p111._dslice_cache[(3,)] = _zeros_vec(width)
        wk_21_raw = fnp.zeros_like(wk_11)
    wk_22_raw = dslice_cache.get((4, (2, 2)), fnp.zeros_like(wk_11))

    wk_21 = wk_21_raw / 2.0
    wk_12 = wk_21.T
    wk_22 = wk_22_raw / 4.0
    ones = _ones(width)
    diag2 = ones * 3.0

    fac1a = w1[:, None] * wk_11 + w2[:, None] * wk_21
    fac2a = _DiagFactor(diag2)
    fac3a = (
        w2[:, None] * w1[None, :] * wk_11
        + w3[:, None] * w1[None, :] * wk_21
        + w2[:, None] * w2[None, :] * wk_12
        + w3[:, None] * w2[None, :] * wk_22
    ).T

    fac1b = w1[:, None] * wk_12 + w2[:, None] * wk_22
    fac2b = fac2a
    fac3b = (
        w3[:, None] * w1[None, :] * wk_11
        + w4[:, None] * w1[None, :] * wk_21
        + w3[:, None] * w2[None, :] * wk_12
        + w4[:, None] * w2[None, :] * wk_22
    ).T

    p111.add_factor_groups(
        [(fac1a, fac2a, fac3a), (fac1b, fac2b, fac3b)],
        [_dslice_21_diag_middle(fac1a, diag2, fac3a), _dslice_21_diag_middle(fac1b, diag2, fac3b)],
        [_diag_diag_middle(fac1a, diag2, fac3a), _diag_diag_middle(fac1b, diag2, fac3b)],
    )

    p_ds = _DSTower.from_slices(p_slices, autozero=True)
    k_ds = _ds_pk_to_k(p_ds, strict=True)
    if 3 in k_ds:
        k_ds[3] = _dst_sub(k_ds[3], p111.get_repeated())

    return {
        1: _HTensor(k_ds[1].to_tensor(), r=0),
        2: _HTensor(k_ds[2].to_tensor(), r=0),
        3: p111 + _FactoredThird.from_dstensor(k_ds[3]),
        4: _ds_harmonic_proj_r1(k_ds[4]),
    }

def _factored_nonlin_k3(wk: dict[int, object], augment: bool | str = "r1") -> dict[int, object]:
    if augment is True:
        augment_mode = "full"
    elif augment is False:
        augment_mode = "none"
    else:
        augment_mode = augment
    project_r1 = augment_mode in {"r1", "r1_no4", "r1_111", "r1_slices", "r1_slices_111", "r1_slices_k211_only", "r1_slices_k211", "full"}
    include_aug_slices = augment_mode in {"r1_slices", "r1_slices_111", "r1_slices_k211_only", "r1_slices_k211", "full"}
    include_aug_111 = augment_mode in {"r1_111", "r1_slices_111", "r1_slices_k211", "full"}
    include_k211 = augment_mode in {"r1_slices_k211_only", "r1_slices_k211", "full"}

    width = wk[1].n
    mean = wk[1].core
    var = fnp.maximum(fnp.diag(wk[2].core), _MIN_VARIANCE)
    sigma = fnp.sqrt(var)
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)
    eval_cache = {}
    dslice_cache = {}

    @cache
    def wick(k: int, p: int):
        if p > 1 and k >= p:
            return math.factorial(p) * wick(k - p + 1, 1)
        return _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, k, p)

    @cache
    def wick_view(k: int, p: int, dim: int, axis: int):
        shape = [1] * dim
        shape[axis] = -1
        return fnp.reshape(wick(k, p), tuple(shape))

    @cache
    def wick_term_view(int_part: tuple[int, ...], k_vec: tuple[int, ...], dim: int, count: int):
        value = None
        for axis, (k, p) in enumerate(zip(k_vec, int_part)):
            view = wick_view(int(k), int(p), dim, axis)
            if value is None:
                value = count * view if count != 1 else view
            else:
                value = value * view
        return value

    def dslice(degree: int, part: tuple[int, ...]):
        key = (degree, part)
        if key not in dslice_cache:
            dslice_cache[key] = _diagslice(wk[degree], part, output_zero_repeated=True)
        return dslice_cache[key]

    def eval_term(vec_part, dim: int, coef: float, factors):
        key = (vec_part, dim)
        if key not in eval_cache:
            if any(degree not in wk for degree, _, _, _ in factors):
                eval_cache[key] = None
            elif not factors:
                eval_cache[key] = fnp.ones((width,) * dim)
            else:
                product = None
                for degree, _, nonzero, part in factors:
                    factor = _expand(dslice(degree, part), nonzero, dim)
                    if product is None:
                        product = coef * factor if coef != 1.0 else factor
                    else:
                        product = product * factor
                eval_cache[key] = product
        return eval_cache[key]

    p_slices = {}
    for int_part, terms in _terms_iso_k3_grouped_for_mode(augment_mode):
        acc = None
        for _, vec_part, count, k_vec, dim, coef, factors in terms:
            term = eval_term(vec_part, dim, coef, factors)
            if term is None:
                continue
            term = term * wick_term_view(int_part, tuple(int(k) for k in k_vec), dim, count)
            acc = term if acc is None else acc + term
        p_slices[int_part] = _symmetrize(acc, vec=int_part)

    w1 = wick(1, 1)
    w2 = wick(2, 1)
    w3 = wick(3, 1)
    w4 = wick(4, 1)

    wk_11 = _zero_repeated(wk[2].core)
    if 3 in wk:
        wk_21 = dslice(3, (2, 1))
        wk_3 = dslice(3, (3,))
        p111 = wk[3].contract_wick(w1, propagate_cache=False)
        p111._dslice_cache[(2, 1)] = wk_21 * (w1[:, None] * w1[:, None] * w1[None, :])
        p111._dslice_cache[(3,)] = wk_3 * w1 * w1 * w1
    else:
        p111 = _FactoredThird(width)
        p111._dslice_cache[(2, 1)] = fnp.zeros_like(wk_11)
        p111._dslice_cache[(3,)] = _zeros_vec(width)
        wk_21 = fnp.zeros_like(wk_11)
    if 4 in wk:
        wk_22 = dslice(4, (2, 2))
    else:
        wk_22 = fnp.zeros_like(wk_11)

    wk_21 = wk_21 / 2.0
    wk_12 = wk_21.T
    wk_22 = wk_22 / 4.0
    eye = _eye(width)
    ones = _ones(width)
    factor_groups: list[tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]] = []
    dslice_21_increments: list[fnp.ndarray | None] = []
    diag_increments: list[fnp.ndarray | None] = []

    def queue_factors(
        factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray],
        dslice_21_increment: fnp.ndarray | None,
        diag_increment: fnp.ndarray | None,
    ) -> None:
        factor_groups.append(factors)
        dslice_21_increments.append(dslice_21_increment)
        diag_increments.append(diag_increment)

    fac1 = w1[:, None] * wk_11 + w2[:, None] * wk_21
    fac3 = (
        w2[:, None] * w1[None, :] * wk_11
        + w3[:, None] * w1[None, :] * wk_21
        + w2[:, None] * w2[None, :] * wk_12
        + w3[:, None] * w2[None, :] * wk_22
    ).T
    diag2 = ones * 3.0
    fac2 = _DiagFactor(diag2)
    queue_factors(
        (fac1, fac2, fac3),
        _dslice_21_diag_middle(fac1, diag2, fac3),
        _diag_diag_middle(fac1, diag2, fac3),
    )

    fac1 = w1[:, None] * wk_12 + w2[:, None] * wk_22
    fac2 = _DiagFactor(diag2)
    fac3 = (
        w3[:, None] * w1[None, :] * wk_11
        + w4[:, None] * w1[None, :] * wk_21
        + w3[:, None] * w2[None, :] * wk_12
        + w4[:, None] * w2[None, :] * wk_22
    ).T
    queue_factors(
        (fac1, fac2, fac3),
        _dslice_21_diag_middle(fac1, diag2, fac3),
        _diag_diag_middle(fac1, diag2, fac3),
    )

    if not project_r1 and 4 in wk and wk[4].r == 2 and wk[4].metric.ndim == 2:
        core = wk[4].core
        metric = wk[4].metric
        metric_diag = fnp.diag(metric)

        fac1 = w2[:, None] * (core * metric_diag)[:, None] * ones[None, :]
        fac2 = _DiagFactor(w1)
        fac3 = w1[:, None] * metric / 2.0
        queue_factors(
            (fac1, fac2, fac3),
            _dslice_21_diag_middle(fac1, w1, fac3),
            _diag_diag_middle(fac1, w1, fac3),
        )

        fac1 = w1[:, None] * metric * core
        fac2 = _DiagFactor(w2)
        fac3 = w1[:, None] * metric
        queue_factors(
            (fac1, fac2, fac3),
            _dslice_21_diag_middle(fac1, w2, fac3),
            _diag_diag_middle(fac1, w2, fac3),
        )

    if include_aug_111 and 4 in wk:
        core = wk[4].core
        metric = wk[4].metric
        metric_diag = fnp.diag(metric)
        metric_full = metric

        fac1 = w2[:, None] * fnp.diag(core)[:, None] * ones[None, :]
        fac2 = w1[:, None] * metric_full
        fac3 = _DiagFactor(w1 / 4.0)
        queue_factors(
            (fac1, fac3, fac2),
            _dslice_21_diag_middle(fac1, w1 / 4.0, fac2),
            _diag_diag_middle(fac1, w1 / 4.0, fac2),
        )

        fac1 = w1[:, None] * core
        fac2 = _DiagFactor(w2)
        fac3 = w1[:, None] * metric_full
        queue_factors(
            (fac1, fac2, fac3),
            _dslice_21_diag_middle(fac1, w2, fac3),
            _diag_diag_middle(fac1, w2, fac3),
        )

        fac1 = w1[:, None] * core
        fac2 = _DiagFactor(w1)
        fac3 = w2[:, None] * metric_diag[:, None] * ones[None, :] / 4.0
        queue_factors(
            (fac1, fac2, fac3),
            _dslice_21_diag_middle(fac1, w1, fac3),
            _diag_diag_middle(fac1, w1, fac3),
        )

    p111.add_factor_groups(factor_groups, dslice_21_increments, diag_increments)

    p_ds = _DSTower.from_slices(p_slices, autozero=True)
    k_ds = _ds_pk_to_k(p_ds, strict=not include_aug_slices)
    if 3 in k_ds:
        k_ds[3] = _dst_sub(k_ds[3], p111.get_repeated())

    k211_contrib = None
    if include_k211:
        rep = _FactoredThird.from_dstensor(p111.get_repeated())
        a = fnp.concatenate((p111.factors[0], -rep.factors[0]), axis=1)
        b = fnp.concatenate((p111.factors[1], rep.factors[1]), axis=1)
        c = fnp.concatenate((p111.factors[2], rep.factors[2]), axis=1)
        pk1 = p_ds[1].slices[(1,)]
        p111_k211 = _symmetrize(
            (fnp.sum(pk1[:, None] * a, axis=0) * b) @ c.T
            + (fnp.sum(pk1[:, None] * b, axis=0) * c) @ a.T
            + (fnp.sum(pk1[:, None] * c, axis=0) * a) @ b.T
        ) / 3.0
        k211_contrib = p111_k211 * (-2.0 * 6.0 * 2.0 / (2.0 * width + 8.0))

        if 3 in wk:
            rep = _FactoredThird.from_dstensor(wk[3].get_repeated())
            a = fnp.concatenate((wk[3].factors[0], -rep.factors[0]), axis=1)
            b = fnp.concatenate((wk[3].factors[1], rep.factors[1]), axis=1)
            c = fnp.concatenate((wk[3].factors[2], rep.factors[2]), axis=1)
            w12 = wick(1, 2)
            p211_k211 = _symmetrize(
                (fnp.sum(w12[:, None] * a, axis=0) * w1[:, None] * b) @ (w1[:, None] * c).T
                + (fnp.sum(w12[:, None] * b, axis=0) * w1[:, None] * c) @ (w1[:, None] * a).T
                + (fnp.sum(w12[:, None] * c, axis=0) * w1[:, None] * a) @ (w1[:, None] * b).T
            ) / 3.0
            k211_contrib = k211_contrib + p211_k211 * (6.0 * 2.0 / (2.0 * width + 8.0))

    out: dict[int, object] = {
        1: _HTensor(k_ds[1].to_tensor(), r=0),
        2: _HTensor(k_ds[2].to_tensor(), r=0),
        3: p111 + _FactoredThird.from_dstensor(k_ds[3]),
    }
    if 4 in k_ds:
        if project_r1:
            out[4] = _ds_harmonic_proj_r1(k_ds[4])
            if k211_contrib is not None:
                out[4].core = out[4].core + k211_contrib
                out[4]._dslice_cache = {}
        else:
            out[4] = _ds_harmonic_proj_r2(k_ds[4])
    return out


def _factored_relu_mean_from_pre(wk: dict[int, object], augment: bool | str = "r1") -> fnp.ndarray:
    """Return only the post-ReLU mean from a pre-activation K=3 tower."""
    if augment is True:
        augment_mode = "full"
    elif augment is False:
        augment_mode = "none"
    else:
        augment_mode = augment

    width = wk[1].n
    mean = wk[1].core
    var = fnp.maximum(fnp.diag(wk[2].core), _MIN_VARIANCE)
    sigma = fnp.sqrt(var)
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    wick_values: dict[tuple[int, int], fnp.ndarray] = {}

    def wick(k: int, p: int):
        key = (k, p)
        value = wick_values.get(key)
        if value is None:
            if p > 1 and k >= p:
                value = math.factorial(p) * wick(k - p + 1, 1)
            else:
                value = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, k, p)
            wick_values[key] = value
        return value

    dslice_cache = {}

    def dslice(degree: int, part: tuple[int, ...]):
        key = (degree, part)
        if key not in dslice_cache:
            dslice_cache[key] = _diagslice(wk[degree], part, output_zero_repeated=True)
        return dslice_cache[key]

    acc = 0.0
    for int_part, vec_part, count, k_vec, dim, coef, factors in _terms_iso_k3_for_mode(augment_mode):
        if int_part != (1,):
            continue
        if any(degree not in wk for degree, _, _, _ in factors):
            continue
        if not factors:
            term = fnp.ones(width)
        else:
            term = None
            for degree, _, _, part in factors:
                factor = dslice(degree, part)
                if term is None:
                    term = coef * factor if coef != 1.0 else factor
                else:
                    term = term * factor
        value = term * wick(int(k_vec[0]), 1)
        if count != 1:
            value = count * value
        acc = acc + value
    return acc


def _relu_mean_from_cumulant_diags(
    mean: fnp.ndarray,
    var: fnp.ndarray,
    k3_diag: fnp.ndarray | float,
    k4_diag: fnp.ndarray | float,
) -> fnp.ndarray:
    var = fnp.maximum(var, _MIN_VARIANCE)
    sigma = fnp.sqrt(var)
    alpha = mean / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)

    base = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, 0, 1)
    w3 = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, 3, 1)
    w4 = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, 4, 1)
    w6 = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, 6, 1)
    w7 = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, 7, 1)
    w8 = _relu_wick_from_stats(mean, var, sigma, alpha, phi, Phi, 8, 1)
    return (
        base
        + (1.0 / 6.0) * k3_diag * w3
        + (1.0 / 24.0) * k4_diag * w4
        - (1.0 / 72.0) * k3_diag * k3_diag * w6
        - (1.0 / 144.0) * k3_diag * k4_diag * w7
        - (1.0 / 1152.0) * k4_diag * k4_diag * w8
    )


def _final_r1_relu_mean_from_tower(tower: dict[int, object], w: fnp.ndarray) -> fnp.ndarray:
    mean = w.T @ tower[1].core
    var = fnp.einsum("ij,ia,ja->a", tower[2].core, w, w)

    if 3 in tower:
        k3_diag = tower[3].contracted_diag(w)
    else:
        k3_diag = 0.0

    if 4 in tower and tower[4].r == 1:
        core_diag = fnp.einsum("ij,ia,ja->a", tower[4].core, w, w)
        metric_diag = fnp.einsum("ij,ia,ja->a", tower[4].metric, w, w)
        k4_diag = core_diag * metric_diag
    else:
        k4_diag = 0.0

    return _relu_mean_from_cumulant_diags(mean, var, k3_diag, k4_diag)


def _factorized_k3_propagation(
    mlp: MLP,
    augment: bool | str = "r1",
    rank_schedule: tuple[int, ...] | None = None,
    rank_compression: str = "topk",
) -> fnp.ndarray:
    """K=3 factorized cumulant propagation for ReLU hidden layers.

    This ports the upstream factorized K=3 path into the narrower whestbench
    setting: square ReLU MLPs, standard normal inputs, no biases, and per-hidden
    layer activation means as output.
    """
    was_gc_enabled = gc.isenabled()
    if was_gc_enabled:
        gc.disable()
    try:
        width = mlp.width
        tower: dict[int, object] = {
            1: _HTensor(fnp.zeros(width), r=0),
            2: _HTensor(fnp.eye(width), r=0),
        }

        rows = []
        mixed_suffix = "_r1_slices_k211"
        switch_at = None
        if isinstance(augment, str) and augment.startswith("last") and augment.endswith(mixed_suffix):
            n_aug_layers = int(augment[len("last") : -len(mixed_suffix)])
            switch_at = max(0, mlp.depth - n_aug_layers)

        for layer_idx, w_mat in enumerate(mlp.weights):
            layer_augment = _AUGMENTED_FACTOR_K3_MODE if switch_at is not None and layer_idx >= switch_at else augment
            if switch_at is not None and layer_idx < switch_at:
                layer_augment = _FACTOR_K3_MODE
            if layer_idx == mlp.depth - 1 and layer_augment == "r1":
                rows.append(_final_r1_relu_mean_from_tower(tower, w_mat))
                break
            wk = {}
            for degree, value in tower.items():
                wk[degree] = value.contract_w(w_mat)
            if layer_idx == mlp.depth - 1:
                rows.append(_factored_relu_mean_from_pre(wk, augment=layer_augment))
                break
            if layer_augment == "r1":
                tower = _factored_nonlin_k3_r1_fast(wk)
            else:
                tower = _factored_nonlin_k3(wk, augment=layer_augment)
            if rank_schedule is not None and 3 in tower and layer_idx < len(rank_schedule):
                if rank_compression == "recent":
                    tower[3] = tower[3].keep_recent_rank(rank_schedule[layer_idx])
                elif rank_compression in ("boundary", "hybrid", "structured"):
                    tower[3] = tower[3].keep_top_group_boundary_rank(rank_schedule[layer_idx])
                elif rank_compression in ("group", "groups", "group_topk"):
                    tower[3] = tower[3].keep_top_group_rank(rank_schedule[layer_idx])
                else:
                    tower[3] = tower[3].keep_top_rank(rank_schedule[layer_idx])
            rows.append(tower[1].core)

        return fnp.stack(rows, axis=0)
    finally:
        if was_gc_enabled:
            gc.enable()


def _parse_rank_schedule(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _extend_rank_schedule(schedule: tuple[int, ...], mlp: MLP) -> tuple[int, ...]:
    hidden_layers = max(mlp.depth - 1, 0)
    if hidden_layers == 0:
        return ()
    if not schedule:
        return ()
    if len(schedule) >= hidden_layers:
        return schedule[:hidden_layers]
    return schedule + (schedule[-1],) * (hidden_layers - len(schedule))


def _r1_rank_schedule_for_mode(mode: str, mlp: MLP) -> tuple[int, ...]:
    explicit = os.environ.get("WHEST_R1_RANK_SCHEDULE")
    if explicit:
        return _extend_rank_schedule(_parse_rank_schedule(explicit), mlp)
    if mode.startswith("r1_cap"):
        cap = int(mode[len("r1_cap") :])
        return (cap,) * max(mlp.depth - 1, 0)
    cap = os.environ.get("WHEST_R1_RANK_CAP")
    if cap:
        return (int(cap),) * max(mlp.depth - 1, 0)
    return _extend_rank_schedule(_DEFAULT_R1_COMPRESSED_SCHEDULE, mlp)


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


def _rademacher_sample_means(mlp: MLP, n_samples: int, rng: fnp.random.Generator) -> fnp.ndarray:
    """Return layer means from antithetic +/-1 input cubature samples."""
    half = max(n_samples // 2, 1)
    signs = rng.integers(0, 2, size=(half, mlp.width))
    x_half = 2.0 * signs - 1.0
    x = fnp.concatenate((x_half, -x_half), axis=0)

    rows = []
    for w in mlp.weights:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _hadamard_rademacher_samples(mlp: MLP, n_samples: int, rng: fnp.random.Generator) -> fnp.ndarray:
    """Return randomized antithetic Hadamard sign blocks for the input layer."""
    width = mlp.width
    rows_per_block = 2 * width
    n_blocks = max(n_samples // rows_per_block, 1)
    base = _hadamard(width)
    blocks = []
    for _ in range(n_blocks):
        flips = 2.0 * rng.integers(0, 2, size=width) - 1.0
        x_half = base * flips[None, :]
        blocks.append(x_half)
        blocks.append(-x_half)
    return fnp.concatenate(tuple(blocks), axis=0)


def _hadamard_rademacher_means(mlp: MLP, n_samples: int, rng: fnp.random.Generator) -> fnp.ndarray:
    """Return layer means from randomized antithetic Hadamard sign blocks."""
    x = _hadamard_rademacher_samples(mlp, n_samples, rng)

    rows = []
    for w in mlp.weights:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _zero_mean_relu_mean_cov(cov_pre: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
    """Exact mean/covariance of ReLU(Z) for zero-mean Gaussian ``Z``."""
    var = fnp.maximum(fnp.diag(cov_pre), _MIN_VARIANCE)
    std = fnp.sqrt(var)
    denom = std[:, None] * std[None, :]
    rho = cov_pre / denom
    rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)
    second = denom * (
        fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
        + (math.pi - fnp.arccos(rho)) * rho
    ) / (2.0 * math.pi)
    mean = std / math.sqrt(2.0 * math.pi)
    cov = second - fnp.outer(mean, mean)
    return mean, cov


def _hadamard_first_cov_recolored_means(
    mlp: MLP,
    n_samples: int,
    rng: fnp.random.Generator,
) -> fnp.ndarray:
    """Hadamard ensemble recolored to the exact first ReLU covariance."""
    x = _hadamard_rademacher_samples(mlp, n_samples, rng)
    w0 = mlp.weights[0]
    y = fnp.maximum(x @ w0, 0.0)

    target_mean, target_cov = _zero_mean_relu_mean_cov(w0.T @ w0)
    sample_mean = fnp.mean(y, axis=0)
    centered = y - sample_mean[None, :]
    sample_cov = (centered.T @ centered) / float(centered.shape[0])
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = _eye(mlp.width)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = centered @ recolor + target_mean[None, :]

    rows = [target_mean]
    for w in mlp.weights[1:]:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _axis_cubature_means(mlp: MLP) -> fnp.ndarray:
    """Return layer means from the 2n axis points that match input covariance."""
    width = mlp.width
    x = fnp.sqrt(float(width)) * fnp.concatenate((fnp.eye(width), -fnp.eye(width)), axis=0)

    rows = []
    for w in mlp.weights:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _sample_count_for_budget(mlp: MLP, budget: int, reserve_flops: float = 0.0) -> int:
    explicit = os.environ.get("WHEST_EXPERIMENT_SAMPLES")
    if explicit:
        return max(int(explicit), 2)
    target_fraction = float(os.environ.get("WHEST_EXPERIMENT_BUDGET_FRACTION", "0.1"))
    target = max(0.0, budget * target_fraction - reserve_flops)
    rough_cost_per_sample = 2.0 * mlp.depth * mlp.width * mlp.width
    return max(2, int(target // rough_cost_per_sample))


def _hadamard_sample_count_for_budget(mlp: MLP, budget: int) -> int:
    explicit = os.environ.get("WHEST_EXPERIMENT_SAMPLES")
    if explicit:
        return max(int(explicit), 2)
    blocks = max(int(os.environ.get("WHEST_HADAMARD_BLOCKS", str(_DEEP_HADAMARD_BLOCKS))), 1)
    rows = blocks * 2 * mlp.width
    rough_cost_per_sample = 2.0 * mlp.depth * mlp.width * mlp.width
    max_rows = max(2, int((0.2 * budget) // rough_cost_per_sample))
    return min(rows, max_rows)


def _covariance_plus_sampling(mlp: MLP, budget: int) -> fnp.ndarray:
    """Blend K=2 covariance propagation with a score-floor antithetic correction."""
    cov_pred = _covariance_propagation(mlp)
    rough_cov_flops = 3.05 * mlp.depth * mlp.width ** 3
    n_samples = _sample_count_for_budget(mlp, budget, reserve_flops=rough_cov_flops)
    sampled = _antithetic_sample_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
    blend = min(0.5, n_samples / (n_samples + 5_000.0))
    return (1.0 - blend) * cov_pred + blend * sampled


def _r1_sample_blend(mlp: MLP, budget: int) -> fnp.ndarray:
    """Diagnostic blend of the default K=3 estimate with antithetic sampling."""
    k3_pred = _factorized_k3_propagation(mlp, augment=_FACTOR_K3_MODE)
    n_samples = _sample_count_for_budget(mlp, budget)
    sampled = _antithetic_sample_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
    blend = float(os.environ.get("WHEST_EXPERIMENT_BLEND", "0.05"))
    return (1.0 - blend) * k3_pred + blend * sampled


def _lowrank_covariance_propagation(mlp: MLP, rank: int) -> fnp.ndarray:
    """Low-rank ensemble covariance propagation with exact marginal variances."""
    width = mlp.width
    rng = fnp.random.default_rng(mlp.seed)
    mu = fnp.zeros(width)
    factors = rng.standard_normal((width, rank)) / math.sqrt(float(rank))

    rows = []
    for w in mlp.weights:
        mu_pre = w.T @ mu
        factors_pre = w.T @ factors
        var_pre = fnp.maximum(fnp.sum(factors_pre * factors_pre, axis=1), _MIN_VARIANCE)

        mu_post, ez2, gain = _relu_moments(mu_pre, var_pre)
        var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)

        factors = factors_pre * gain[:, None]
        factor_var = fnp.maximum(fnp.sum(factors * factors, axis=1), _MIN_VARIANCE)
        factors = factors * fnp.sqrt(var_post / factor_var)[:, None]

        mu = mu_post
        rows.append(mu)

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
    """Factorized K=3 cumulant-propagation estimator."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        _all_terms_iso_k3()
        _terms_iso_k3_for_mode("none")
        _terms_iso_k3_grouped_for_mode("none")
        _terms_iso_k3_for_mode(_FACTOR_K3_MODE)
        _terms_iso_k3_grouped_for_mode(_FACTOR_K3_MODE)
        _terms_iso_k3_for_mode("r1_slices_k211")
        _terms_iso_k3_grouped_for_mode("r1_slices_k211")
        _terms_iso_k3_for_mode("r1_slices")
        _terms_iso_k3_grouped_for_mode("r1_slices")
        for vertices in range(1, 5):
            _multigraphs(vertices, 2)
        for width in (16, 32, 64, 128, 256):
            _eye(width)
            _ones(width)
            _zeros_vec(width)
            _empty_factor(width)
            _idx(width)
            if width & (width - 1) == 0:
                _hadamard(width)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer mean activations.

        Returns an array of shape ``(depth, width)``.
        """
        mode = os.environ.get("WHEST_EXPERIMENT_MODE") or os.environ.get("WHEST_K3_MODE", "")
        if mode in ("", "default"):
            if mlp.depth >= _DEEP_HADAMARD_MIN_DEPTH:
                n_samples = _hadamard_sample_count_for_budget(mlp, budget)
                return _hadamard_first_cov_recolored_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
            return _factorized_k3_propagation(mlp, augment=_FACTOR_K3_MODE)
        if mode == "r1":
            return _factorized_k3_propagation(mlp, augment=_FACTOR_K3_MODE)
        if mode in ("r1_compressed", "r1_rank_schedule") or mode.startswith("r1_cap"):
            schedule = _r1_rank_schedule_for_mode(mode, mlp)
            compression = os.environ.get("WHEST_R1_COMPRESS", "topk")
            return _factorized_k3_propagation(
                mlp,
                augment=_FACTOR_K3_MODE,
                rank_schedule=schedule,
                rank_compression=compression,
            )
        if mode in ("none", "simple", "r1_no4", "r1_slices", "r1_slices_k211") or (
            mode.startswith("last") and mode.endswith("_r1_slices_k211")
        ):
            return _factorized_k3_propagation(mlp, augment=mode)
        if mode == "mean":
            return _mean_propagation(mlp)
        if mode == "cov":
            return _covariance_propagation(mlp)
        if mode == "sample":
            n_samples = _sample_count_for_budget(mlp, budget)
            return _antithetic_sample_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
        if mode == "rademacher":
            n_samples = _sample_count_for_budget(mlp, budget)
            return _rademacher_sample_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
        if mode == "hadamard":
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_rademacher_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
        if mode == "hadamard_first_cov":
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_first_cov_recolored_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
        if mode == "axis":
            return _axis_cubature_means(mlp)
        if mode == "k2_sample":
            return _covariance_plus_sampling(mlp, budget)
        if mode == "r1_sample_blend":
            return _r1_sample_blend(mlp, budget)
        if mode == "lr_cov":
            rank = int(os.environ.get("WHEST_LR_RANK", "256"))
            return _lowrank_covariance_propagation(mlp, rank=rank)
        raise ValueError(f"Unsupported WHEST_EXPERIMENT_MODE/WHEST_K3_MODE: {mode}")


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
    parser.add_argument("--depth", type=int, default=32)  # phase-1 competition shape (warmup was 8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
