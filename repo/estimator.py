"""ARC WhestBench estimator for ReLU MLPs.

Submission for https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026.

Depth-32 contest MLPs use randomized antithetic Hadamard cubature with exact
first-layer ReLU mean/covariance recoloring. Shallower MLPs use the optimized
factorized K=3 r=1 cumulant route.
"""

from __future__ import annotations

import gc
import itertools
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import cache

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

if hasattr(flops, "configure"):
    flops.configure(symmetry_warnings=False)

_MIN_VARIANCE = 1e-30
_DEEP_HADAMARD_MIN_DEPTH = 16
_DEEP_HADAMARD_BLOCKS = 16
_DEEP_STRASSEN_LEVELS = 3
_DEEP_VARIANCE_MATCH_STRENGTH = 1.5
_DEEP_HADAMARD_MAX_BLOCKS = 32
_DEEP_BLOCK_FIXED_OVERHEAD = 1.09
_DEEP_BLOCK_COST_SAFETY = 1.03
_DEEP_MEASURED_ROW_FLOPS = {
    3: 3.17e6,
    4: 2.94e6,
}
_DEEP_RESIDUAL_BLOCK_SAFETY = {
    3: 1.0,
    4: 1.06,
}
_HYBRID_ANALYTIC_LAYER_FLOPS = 1.05e8
_HYBRID_JOINT_K3_PREFIX_FLOPS = 1.1e9
_HYBRID_JOINT_K3_MAX_TERMS = 128
_HYBRID_JOINT_K3_COV_FRACTION = 0.5
_HYBRID_JOINT_K3_GLOBAL_DAMP = 0.516
_ZERO_MEAN_RELU_THIRD_CENTRAL_COEFF = (
    math.sqrt(2.0 / math.pi)
    - 1.5 / math.sqrt(2.0 * math.pi)
    + 2.0 / ((2.0 * math.pi) ** 1.5)
)
_BIVAR_RELU_GL16_NODES = (
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.7554044083550030,
    -0.6178762444026438,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.0950125098376374,
    0.0950125098376374,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.7554044083550030,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
)
_BIVAR_RELU_GL16_WEIGHTS = (
    0.0271524594117541,
    0.0622535239386479,
    0.0951585116824928,
    0.1246289712555339,
    0.1495959888165767,
    0.1691565193950025,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541,
)


@dataclass(frozen=True)
class _HadamardPosthocConfig:
    scale_cap: float | None = None
    kurtosis_gate: float | None = None
    gaussian_pull: float = 0.0
    edgeworth_blend: float = 0.0
    trimmed_final: bool = False
    second_variance_strength: float | None = None


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
def _terms_iso_k3():
    terms = []
    for int_part, vec_part, count in _all_terms_iso_k3():
        if (
            len(int_part) > 3
            or int_part in ((3, 1), (2, 1, 1))
            or int_part == (1, 1, 1)
            or (int_part, set(vec_part)) == ((2, 1, 1), {(1, 1, 1)})
        ):
            continue
        factors = []
        for vec in vec_part:
            nonzero = tuple(i for i, value in enumerate(vec) if value > 0)
            factors.append((sum(vec), vec, nonzero, tuple(vec[i] for i in nonzero)))
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
def _terms_iso_k3_grouped():
    grouped = defaultdict(list)
    for term in _terms_iso_k3():
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


def _scale_factor_rows(factor, scale: fnp.ndarray):
    if isinstance(factor, _DiagFactor):
        return _DiagFactor(factor.diag * scale)
    return factor * scale[:, None]


def _contract_factor_w(factor, w: fnp.ndarray):
    if isinstance(factor, _DiagFactor):
        return w.T * factor.diag[None, :]
    return w.T @ factor


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

    def contract_w(self, w: fnp.ndarray) -> "_FactoredThird":
        return self._replace_groups(self._contract_groups(self._groups, w))

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
    for int_part, terms in _terms_iso_k3_grouped():
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


def _factorized_k3_propagation(mlp: MLP) -> fnp.ndarray:
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
        for layer_idx, w_mat in enumerate(mlp.weights):
            if layer_idx == mlp.depth - 1:
                rows.append(_final_r1_relu_mean_from_tower(tower, w_mat))
                break
            wk = {}
            for degree, value in tower.items():
                wk[degree] = value.contract_w(w_mat)
            tower = _factored_nonlin_k3_r1_fast(wk)
            rows.append(tower[1].core)

        return fnp.stack(rows, axis=0)
    finally:
        if was_gc_enabled:
            gc.enable()


def _hadamard_sign_half_blocks(
    mlp: MLP,
    n_samples: int,
    rng: fnp.random.Generator,
    split_factor: int = 1,
) -> fnp.ndarray:
    """Return the positive halves of randomized antithetic Hadamard sign blocks.

    The antithetic halves are exact negations, so the first-layer matmul only
    needs these rows; callers recover the antithetic activations from
    ``relu(-pre)`` without a second matmul.
    """
    width = mlp.width
    rows_per_block = 2 * width
    n_blocks = max(n_samples // rows_per_block, 1)
    base = _hadamard(width)
    blocks = []
    split_factor = max(int(split_factor), 1)
    if width % split_factor:
        split_factor = 1
    rows_per_split = width // split_factor
    for block_idx in range(n_blocks * split_factor):
        split_idx = block_idx % split_factor
        row_start = split_idx * rows_per_split
        row_stop = row_start + rows_per_split
        flips = 2.0 * rng.integers(0, 2, size=width) - 1.0
        blocks.append(base[row_start:row_stop] * flips[None, :])
    return fnp.concatenate(tuple(blocks), axis=0)


def _hadamard_sign_fresh_half_blocks(
    mlp: MLP,
    n_half_blocks: int,
    rng: fnp.random.Generator,
) -> fnp.ndarray:
    """Return fresh randomized Hadamard half-blocks without antithetic pairs."""
    width = mlp.width
    base = _hadamard(width)
    blocks = []
    for _ in range(max(int(n_half_blocks), 1)):
        flips = 2.0 * rng.integers(0, 2, size=width) - 1.0
        blocks.append(base * flips[None, :])
    return fnp.concatenate(tuple(blocks), axis=0)


def _norm_ppf(p: float) -> float:
    """Acklam rational approximation to the standard normal quantile."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > 1.0 - plow:
        return -_norm_ppf(1.0 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


@cache
def _chi_stratified_radial_scales(width: int) -> tuple[float, ...]:
    """Stratified chi_width radius factors normalized to unit mean square.

    Wilson-Hilferty chi-square quantiles at midpoint strata, expressed as
    per-row multipliers relative to the rigid Hadamard radius ``sqrt(width)``.
    These are MLP-independent constants, like the Hadamard matrix itself.
    """
    n = width
    radii_sq = []
    for k in range(n):
        z = _norm_ppf((k + 0.5) / n)
        radii_sq.append(n * (1.0 - 2.0 / (9.0 * n) + z * math.sqrt(2.0 / (9.0 * n))) ** 3)
    mean_sq = sum(radii_sq) / n
    return tuple(math.sqrt(r_sq / mean_sq / n) for r_sq in radii_sq)


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


def _gaussian_relu_variance(mean_pre: fnp.ndarray, var_pre: fnp.ndarray) -> fnp.ndarray:
    """Gaussian marginal ReLU variance for possibly nonzero preactivations."""
    var_pre = fnp.maximum(var_pre, _MIN_VARIANCE)
    sigma = fnp.sqrt(var_pre)
    alpha = mean_pre / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)
    relu_mean = sigma * phi + mean_pre * Phi
    second = (mean_pre * mean_pre + var_pre) * Phi + mean_pre * sigma * phi
    return fnp.maximum(second - relu_mean * relu_mean, _MIN_VARIANCE)


def _gaussian_relu_mean(mean_pre: fnp.ndarray, var_pre: fnp.ndarray) -> fnp.ndarray:
    var_pre = fnp.maximum(var_pre, _MIN_VARIANCE)
    sigma = fnp.sqrt(var_pre)
    alpha = mean_pre / sigma
    return sigma * flops.stats.norm.pdf(alpha) + mean_pre * flops.stats.norm.cdf(alpha)


def _edgeworth_relu_mean(pre: fnp.ndarray) -> fnp.ndarray:
    mean_pre = fnp.mean(pre, axis=0)
    centered = pre - mean_pre[None, :]
    var_pre = fnp.maximum(fnp.mean(centered * centered, axis=0), _MIN_VARIANCE)
    sigma = fnp.sqrt(var_pre)
    z = centered / sigma[None, :]
    skew = fnp.mean(z * z * z, axis=0)
    excess = fnp.mean(z * z * z * z, axis=0) - 3.0
    alpha = mean_pre / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)
    gaussian = sigma * phi + mean_pre * Phi
    h3_int = -sigma * (alpha * alpha - 1.0) * phi
    h4_int = sigma * (alpha * alpha * alpha - 3.0 * alpha) * phi
    return gaussian + (skew / 6.0) * h3_int + (excess / 24.0) * h4_int


def _gaussian_relu_mean_cov(
    mean_pre: fnp.ndarray,
    cov_pre: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    """Gaussian ReLU mean/covariance for nonzero-mean preactivations."""
    var_pre = fnp.maximum(fnp.diag(cov_pre), _MIN_VARIANCE)
    sigma = fnp.sqrt(var_pre)
    beta = mean_pre / sigma
    phi = flops.stats.norm.pdf(beta)
    Phi = flops.stats.norm.cdf(beta)
    relu_mean = sigma * phi + mean_pre * Phi
    marginal_second = (mean_pre * mean_pre + var_pre) * Phi + mean_pre * sigma * phi

    denom = sigma[:, None] * sigma[None, :]
    rho = cov_pre / denom
    rho = fnp.maximum(fnp.minimum(rho, 1.0 - 1e-7), -1.0 + 1e-7)
    alpha_i = -beta[:, None]
    alpha_j = -beta[None, :]
    tail_i = Phi[:, None]
    tail_j = Phi[None, :]
    rho_int = fnp.zeros_like(rho)
    for node, weight in zip(_BIVAR_RELU_GL16_NODES, _BIVAR_RELU_GL16_WEIGHTS):
        r = 0.5 * rho * (node + 1.0)
        one_minus = fnp.maximum(1.0 - r * r, _MIN_VARIANCE)
        exponent = -(
            alpha_i * alpha_i - 2.0 * r * alpha_i * alpha_j + alpha_j * alpha_j
        ) / (2.0 * one_minus)
        phi2 = fnp.exp(exponent) / (2.0 * math.pi * fnp.sqrt(one_minus))
        rho_int = rho_int + weight * (rho - r) * phi2
    second = relu_mean[:, None] * relu_mean[None, :] + denom * (
        rho * tail_i * tail_j + 0.5 * rho * rho_int
    )
    eye = _eye(mean_pre.shape[0])
    second = second * (1.0 - eye) + fnp.diag(marginal_second) * eye
    cov = second - fnp.outer(relu_mean, relu_mean)
    return relu_mean, cov


def _strassen_matmul_batched(a: fnp.ndarray, b: fnp.ndarray, levels: int) -> fnp.ndarray:
    """Batched-leaf rectangular Strassen matmul for eligible shapes."""
    if levels <= 0:
        return a @ b
    rows, inner = a.shape
    b_rows, cols = b.shape
    divisor = 1 << levels
    if (
        inner != b_rows
        or rows % divisor
        or inner % divisor
        or cols % divisor
        or inner // divisor < 16
    ):
        return a @ b

    a_stack = fnp.reshape(a, (1, rows, inner))
    b_stack = fnp.reshape(b, (1, inner, cols))
    shapes = []
    for _ in range(levels):
        _, a_rows, a_cols = a_stack.shape
        _, b_rows, b_cols = b_stack.shape
        shapes.append((a_rows, b_cols))
        row_mid = a_rows // 2
        inner_mid = a_cols // 2
        col_mid = b_cols // 2
        a11 = a_stack[:, :row_mid, :inner_mid]
        a12 = a_stack[:, :row_mid, inner_mid:]
        a21 = a_stack[:, row_mid:, :inner_mid]
        a22 = a_stack[:, row_mid:, inner_mid:]
        b11 = b_stack[:, :inner_mid, :col_mid]
        b12 = b_stack[:, :inner_mid, col_mid:]
        b21 = b_stack[:, inner_mid:, :col_mid]
        b22 = b_stack[:, inner_mid:, col_mid:]

        stack_count = a_stack.shape[0]
        a_stack = fnp.reshape(fnp.stack(
            (
                a11 + a22,
                a21 + a22,
                a11,
                a22,
                a11 + a12,
                a21 - a11,
                a12 - a22,
            ),
            axis=1,
        ), (stack_count * 7, row_mid, inner_mid))
        b_stack = fnp.reshape(fnp.stack(
            (
                b11 + b22,
                b11,
                b12 - b22,
                b21 - b11,
                b22,
                b11 + b12,
                b21 + b22,
            ),
            axis=1,
        ), (stack_count * 7, inner_mid, col_mid))

    if a_stack.shape[1] <= 512:
        products = a_stack @ b_stack
    else:
        products = fnp.einsum("brk,bkc->brc", a_stack, b_stack)
    for out_rows, out_cols in reversed(shapes):
        leaf_rows = products.shape[1]
        leaf_cols = products.shape[2]
        groups = products.shape[0] // 7
        p = fnp.reshape(products, (groups, 7, leaf_rows, leaf_cols))
        m1 = p[:, 0]
        m2 = p[:, 1]
        m3 = p[:, 2]
        m4 = p[:, 3]
        m5 = p[:, 4]
        m6 = p[:, 5]
        m7 = p[:, 6]
        c11 = m1 + m4 - m5 + m7
        c12 = m3 + m5
        c21 = m2 + m4
        c22 = m1 - m2 + m3 + m6
        products = fnp.concatenate(
            (
                fnp.concatenate((c11, c12), axis=2),
                fnp.concatenate((c21, c22), axis=2),
            ),
            axis=1,
        )
    return fnp.reshape(products, (rows, cols))


def _strassen_matmul_hybrid_l4(a: fnp.ndarray, b: fnp.ndarray) -> fnp.ndarray:
    """One explicit Strassen level over batched L3 leaves."""
    rows, inner = a.shape
    _, cols = b.shape
    row_mid = rows // 2
    inner_mid = inner // 2
    col_mid = cols // 2
    a11 = a[:row_mid, :inner_mid]
    a12 = a[:row_mid, inner_mid:]
    a21 = a[row_mid:, :inner_mid]
    a22 = a[row_mid:, inner_mid:]
    b11 = b[:inner_mid, :col_mid]
    b12 = b[:inner_mid, col_mid:]
    b21 = b[inner_mid:, :col_mid]
    b22 = b[inner_mid:, col_mid:]

    m1 = _strassen_matmul_batched(a11 + a22, b11 + b22, 3)
    m2 = _strassen_matmul_batched(a21 + a22, b11, 3)
    m3 = _strassen_matmul_batched(a11, b12 - b22, 3)
    m4 = _strassen_matmul_batched(a22, b21 - b11, 3)
    m5 = _strassen_matmul_batched(a11 + a12, b22, 3)
    m6 = _strassen_matmul_batched(a21 - a11, b11 + b12, 3)
    m7 = _strassen_matmul_batched(a12 - a22, b21 + b22, 3)
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return fnp.concatenate(
        (
            fnp.concatenate((c11, c12), axis=1),
            fnp.concatenate((c21, c22), axis=1),
        ),
        axis=0,
    )


def _strassen_matmul(a: fnp.ndarray, b: fnp.ndarray, levels: int) -> fnp.ndarray:
    """Rectangular-by-square Strassen matmul for propagation."""
    if levels == 4:
        rows, inner = a.shape
        b_rows, cols = b.shape
        if (
            inner == b_rows
            and inner == cols
            and rows % 16 == 0
            and inner % 16 == 0
            and cols % 16 == 0
        ):
            return _strassen_matmul_hybrid_l4(a, b)
    return _strassen_matmul_batched(a, b, levels)


def _hybrid_analytic_blocks_for_budget(
    mlp: MLP,
    budget: int,
    strassen_levels: int,
    prefix_layers: int,
    joint_k3_matched: bool = False,
) -> int:
    rows_per_block = 2 * mlp.width
    suffix_layers = max(mlp.depth - prefix_layers, 1)
    measured_row_cost = _DEEP_MEASURED_ROW_FLOPS.get(strassen_levels)
    analytic_cost = (
        _HYBRID_JOINT_K3_PREFIX_FLOPS
        if joint_k3_matched
        else max(prefix_layers, 1) * _HYBRID_ANALYTIC_LAYER_FLOPS
    )
    score_floor_budget = max(0.1 * budget - analytic_cost, rows_per_block)
    if measured_row_cost is not None and mlp.width == 256 and mlp.depth == 32:
        per_block_cost = (
            rows_per_block
            * measured_row_cost
            * suffix_layers
            / mlp.depth
            * _DEEP_BLOCK_COST_SAFETY
            * _DEEP_RESIDUAL_BLOCK_SAFETY.get(strassen_levels, 1.0)
        )
    else:
        plain_layer_cost = rows_per_block * mlp.width * (2 * mlp.width - 1)
        strassen_discount = (7.0 / 8.0) ** max(strassen_levels, 0)
        per_block_cost = (
            suffix_layers
            * plain_layer_cost
            * strassen_discount
            * _DEEP_BLOCK_FIXED_OVERHEAD
            * _DEEP_BLOCK_COST_SAFETY
        )
    return min(max(int(score_floor_budget // per_block_cost), 1), _DEEP_HADAMARD_MAX_BLOCKS)


def _layer2_gaussian_prefix_and_factored_k3(mlp: MLP) -> tuple[fnp.ndarray, fnp.ndarray, _FactoredThird]:
    width = mlp.width
    w0 = mlp.weights[0]
    w1 = mlp.weights[1]
    z1_cov = w0.T @ w0
    y1_mean, y1_cov = _zero_mean_relu_mean_cov(z1_cov)
    tower: dict[int, object] = {
        1: _HTensor(_zeros_vec(width), r=0),
        2: _HTensor(_eye(width), r=0),
    }
    wk1 = {degree: value.contract_w(w0) for degree, value in tower.items()}
    y1_tower = _factored_nonlin_k3_r1_fast(wk1)
    z2_k3 = y1_tower[3].contract_w(w1)
    return y1_mean @ w1, w1.T @ y1_cov @ w1, z2_k3


def _joint_k3_quadratic_terms(
    chol: fnp.ndarray,
    k3: _FactoredThird,
    max_terms: int | None = None,
    symmetric_order: bool = False,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    factors = []
    eye_jitter = fnp.maximum(fnp.mean(fnp.diag(chol @ chol.T)), _MIN_VARIANCE) * 1e-10
    solve_chol = chol + eye_jitter * _eye(chol.shape[0])
    for group in k3._groups:
        a, b, c = (_materialize_factor(factor) for factor in group)
        for out, left, right in ((c, a, b), (a, b, c), (b, c, a)):
            u = fnp.linalg.solve(solve_chol, left)
            v = fnp.linalg.solve(solve_chol, right)
            factors.append((out / 6.0, u, v))
    if not factors:
        empty = _empty_factor(k3.width)
        return empty, empty, empty
    if max_terms is not None and len(factors) > max_terms and not symmetric_order:
        scored = []
        for item in factors:
            gamma, u, v = item
            score = float(fnp.sum(gamma * gamma) * fnp.sum(u * u) * fnp.sum(v * v))
            scored.append((score, item))
        scored.sort(key=lambda item: item[0], reverse=True)
        factors = [item for _, item in scored[:max_terms]]
    gamma = fnp.concatenate(tuple(item[0] for item in factors), axis=1)
    u_all = fnp.concatenate(tuple(item[1] for item in factors), axis=1)
    v_all = fnp.concatenate(tuple(item[2] for item in factors), axis=1)
    if max_terms is not None and gamma.shape[1] > max_terms:
        if symmetric_order:
            gg = gamma.T @ gamma
            uu = u_all.T @ u_all
            vv = v_all.T @ v_all
            gu = gamma.T @ u_all
            gv = gamma.T @ v_all
            uv = u_all.T @ v_all
            gram = (
                gg * uu * vv
                + gg * uv * uv.T
                + gu * gu.T * vv
                + gu * uv.T * gv
                + gv * gu.T * uv.T
                + gv * uu * gv.T
            )
            eigvals, eigvecs = fnp.linalg.eigh((gram + gram.T) * 0.5)
            top = eigvecs[:, fnp.argsort(eigvals)[-max_terms:]]
            score = fnp.sum(top * top, axis=1)
        else:
            score = (
                fnp.sum(gamma * gamma, axis=0)
                * fnp.sum(u_all * u_all, axis=0)
                * fnp.sum(v_all * v_all, axis=0)
            )
        keep = fnp.argsort(score)[-max_terms:]
        gamma = gamma[:, keep]
        u_all = u_all[:, keep]
        v_all = v_all[:, keep]
    return gamma, u_all, v_all


def _joint_k3_quadratic_cov(gamma: fnp.ndarray, u: fnp.ndarray, v: fnp.ndarray) -> fnp.ndarray:
    if gamma.shape[1] == 0:
        return fnp.zeros((gamma.shape[0], gamma.shape[0]))
    uu = u.T @ u
    vv = v.T @ v
    uv = u.T @ v
    term = uu * vv + uv * uv.T
    return gamma @ term @ gamma.T


def _largest_pd_transport_scale(cov: fnp.ndarray, q_cov: fnp.ndarray) -> float:
    if fnp.max(fnp.abs(q_cov)) <= 0.0:
        return 1.0
    jitter = fnp.maximum(fnp.mean(fnp.diag(cov)), _MIN_VARIANCE) * 1e-8
    eye = _eye(cov.shape[0])
    lo = 0.0
    hi = 1.0
    for _ in range(32):
        mid = (lo + hi) * 0.5
        try:
            fnp.linalg.cholesky(cov - (mid * mid) * q_cov + jitter * eye)
            lo = mid
        except Exception:
            hi = mid
    return lo


def _taper_joint_k3_terms(
    cov: fnp.ndarray,
    gamma: fnp.ndarray,
    u: fnp.ndarray,
    v: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    if gamma.shape[1] == 0:
        return gamma, u, v
    eigvals, eigvecs = fnp.linalg.eigh(cov)
    budget = _HYBRID_JOINT_K3_COV_FRACTION * fnp.maximum(eigvals, _MIN_VARIANCE)
    gamma_eig = eigvecs.T @ gamma
    term_load = (gamma_eig * gamma_eig) * (
        fnp.sum(u * u, axis=0)[None, :] * fnp.sum(v * v, axis=0)[None, :]
        + fnp.sum(u * v, axis=0)[None, :] * fnp.sum(u * v, axis=0)[None, :]
    )
    weights = fnp.ones(gamma.shape[1])
    for _ in range(3):
        load = term_load @ (weights * weights)
        direction_scale = fnp.minimum(1.0, fnp.sqrt(budget / fnp.maximum(load, _MIN_VARIANCE)))
        weights = weights * fnp.min(
            fnp.where(term_load > 0.0, direction_scale[:, None], 1.0),
            axis=0,
        )
    return gamma * weights[None, :], u, v


def _joint_k3_transport(
    cov: fnp.ndarray,
    k3: _FactoredThird,
    max_terms: int | None = _HYBRID_JOINT_K3_MAX_TERMS,
    taper: str = "eigen",
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, float, fnp.ndarray]:
    width = cov.shape[0]
    eye = _eye(width)
    jitter = fnp.maximum(fnp.mean(fnp.diag(cov)), _MIN_VARIANCE) * 1e-6
    chol = fnp.linalg.cholesky(cov + jitter * eye)
    symmetric_order = taper == "hybr"
    gamma, u, v = _joint_k3_quadratic_terms(chol, k3, max_terms, symmetric_order)
    if taper == "eigen":
        gamma, u, v = _taper_joint_k3_terms(cov, gamma, u, v)
    q_cov = _joint_k3_quadratic_cov(gamma, u, v)
    pd_damping = _largest_pd_transport_scale(cov, q_cov)
    damping = min(_HYBRID_JOINT_K3_GLOBAL_DAMP, pd_damping) if taper in ("global", "hybr") else pd_damping
    gamma = gamma * damping
    q_cov = q_cov * (damping * damping)
    chol = fnp.linalg.cholesky(cov - q_cov + jitter * eye)
    return chol, gamma, u, v, damping, q_cov


def _apply_joint_k3_transport(
    signs: fnp.ndarray,
    mean: fnp.ndarray,
    cov: fnp.ndarray,
    chol: fnp.ndarray,
    gamma: fnp.ndarray,
    u: fnp.ndarray,
    v: fnp.ndarray,
) -> fnp.ndarray:
    linear = signs @ chol.T
    if gamma.shape[1] == 0:
        return linear + mean[None, :]
    ug = signs @ u
    vg = signs @ v
    uv = fnp.sum(u * v, axis=0)
    q = fnp.zeros_like(linear)
    chunk = 192
    for start in range(0, gamma.shape[1], chunk):
        stop = min(start + chunk, gamma.shape[1])
        q = q + (
            (ug[:, start:stop] * vg[:, start:stop] - uv[None, start:stop])
            @ gamma[:, start:stop].T
        )
    pre = linear + q
    sample_mean = fnp.mean(pre, axis=0)
    centered = pre - sample_mean[None, :]
    sample_cov = (centered.T @ centered) / float(centered.shape[0])
    jitter = fnp.maximum(fnp.mean(fnp.diag(cov)), _MIN_VARIANCE) * 1e-6
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * _eye(cov.shape[0]))
    target_chol = fnp.linalg.cholesky(cov + jitter * _eye(cov.shape[0]))
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    return centered @ recolor + mean[None, :]


def _hybrid_analytic_prefix_hadamard_means(
    mlp: MLP,
    n_samples: int,
    rng: fnp.random.Generator,
    prefix_layers: int,
    variance_match_strength: float,
    strassen_levels: int,
    skew_matched: bool = False,
    joint_k3_matched: bool = False,
    joint_k3_max_terms: int | None = _HYBRID_JOINT_K3_MAX_TERMS,
    joint_k3_taper: str = "eigen",
) -> fnp.ndarray:
    """Analytic Gaussian-closure prefix, then Hadamard sample the prefix preactivation."""
    prefix_layers = min(max(int(prefix_layers), 1), mlp.depth)
    mean = _zeros_vec(mlp.width)
    cov = _eye(mlp.width)
    analytic_rows = []
    pre_mean = None
    pre_cov = None
    pre_kappa3_diag = None
    relu_kappa3_diag = None
    z2_k3 = None
    if joint_k3_matched and prefix_layers == 2:
        pre_mean, pre_cov, z2_k3 = _layer2_gaussian_prefix_and_factored_k3(mlp)
        mean, cov = _gaussian_relu_mean_cov(pre_mean, pre_cov)
        analytic_rows = [_zero_mean_relu_mean_cov(mlp.weights[0].T @ mlp.weights[0])[0], mean]
    else:
        for layer_idx, w in enumerate(mlp.weights[:prefix_layers]):
            pre_mean = mean @ w
            pre_cov = w.T @ cov @ w
            if skew_matched and layer_idx == 1 and relu_kappa3_diag is not None:
                pre_kappa3_diag = relu_kappa3_diag @ (w * w * w)
            if layer_idx == 0:
                mean, cov = _zero_mean_relu_mean_cov(pre_cov)
                if skew_matched:
                    var0 = fnp.maximum(fnp.diag(pre_cov), _MIN_VARIANCE)
                    relu_kappa3_diag = _ZERO_MEAN_RELU_THIRD_CENTRAL_COEFF * var0 * fnp.sqrt(var0)
            else:
                mean, cov = _gaussian_relu_mean_cov(pre_mean, pre_cov)
                relu_kappa3_diag = None
            analytic_rows.append(mean)

    assert pre_mean is not None and pre_cov is not None
    n_blocks = max(n_samples // (2 * mlp.width), 1)
    signs = _hadamard_sign_half_blocks(mlp, n_blocks * 2 * mlp.width, rng)
    if joint_k3_matched and z2_k3 is not None:
        full_signs = fnp.concatenate((signs, -signs), axis=0)
        centered_chol, gamma, u, v, _, _ = _joint_k3_transport(
            pre_cov,
            z2_k3,
            joint_k3_max_terms,
            joint_k3_taper,
        )
        pre = _apply_joint_k3_transport(full_signs, pre_mean, pre_cov, centered_chol, gamma, u, v)
    else:
        centered_chol = fnp.linalg.cholesky(pre_cov + fnp.maximum(fnp.mean(fnp.diag(pre_cov)), _MIN_VARIANCE) * 1e-6 * _eye(mlp.width))
        pre_half = signs @ centered_chol.T
        pre = fnp.concatenate((pre_half, -pre_half), axis=0) + pre_mean[None, :]
    if skew_matched and prefix_layers == 2 and pre_kappa3_diag is not None:
        var = fnp.maximum(fnp.diag(pre_cov), _MIN_VARIANCE)
        std = fnp.sqrt(var)
        gamma = pre_kappa3_diag / (var * std)
        centered_pre = pre - pre_mean[None, :]
        pre = pre + (gamma[None, :] / 6.0) * (
            (centered_pre * centered_pre) / var[None, :] - 1.0
        ) * std[None, :]
        sample_mean = fnp.mean(pre, axis=0)
        centered_pre = pre - sample_mean[None, :]
        sample_var = fnp.maximum(fnp.mean(centered_pre * centered_pre, axis=0), _MIN_VARIANCE)
        pre = centered_pre * fnp.sqrt(var / sample_var)[None, :] + pre_mean[None, :]
    x = fnp.maximum(pre, 0.0)

    rows = list(analytic_rows[:-1])
    rows.append(fnp.mean(x, axis=0))
    for layer_idx, w in enumerate(mlp.weights[prefix_layers:], start=prefix_layers):
        pre = _strassen_matmul(x, w, strassen_levels)
        x = fnp.maximum(pre, 0.0)
        if layer_idx == prefix_layers:
            pre_mean_sample = fnp.mean(pre, axis=0)
            pre_centered = pre - pre_mean_sample[None, :]
            target_var = _gaussian_relu_variance(
                pre_mean_sample,
                fnp.mean(pre_centered * pre_centered, axis=0),
            )
            sample_mean = fnp.mean(x, axis=0)
            centered_layer = x - sample_mean[None, :]
            sample_var = fnp.maximum(fnp.mean(centered_layer * centered_layer, axis=0), _MIN_VARIANCE)
            scale = 1.0 + variance_match_strength * (fnp.sqrt(target_var / sample_var) - 1.0)
            x = centered_layer * scale[None, :] + sample_mean[None, :]
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _hadamard_first_cov_recolored_means(
    mlp: MLP,
    n_samples: int,
    rng: fnp.random.Generator,
    variance_match_layers: int = 0,
    variance_match_start_layer: int = 1,
    variance_match_strength: float = 1.0,
    exact_recolor_layers: int = 0,
    radial_chi: bool = False,
    strassen_levels: int = 0,
    split_factor: int = 1,
    mirror_layer: int | None = None,
    final_cv3: bool = False,
    antithetic_fraction: float = 1.0,
    posthoc: _HadamardPosthocConfig | None = None,
) -> fnp.ndarray:
    """Hadamard ensemble recolored to the exact first ReLU covariance."""
    initial_samples = n_samples
    if mirror_layer is not None:
        initial_samples = max((n_samples // (4 * mlp.width)) * (2 * mlp.width), 2 * mlp.width)
    w0 = mlp.weights[0]
    n_blocks = max(initial_samples // (2 * mlp.width), 1)
    antithetic_blocks = max(min(int(round(n_blocks * antithetic_fraction)), n_blocks), 0)
    y_parts = []
    if antithetic_blocks:
        x_half = _hadamard_sign_half_blocks(mlp, antithetic_blocks * 2 * mlp.width, rng, split_factor)
        pre_half = _strassen_matmul(x_half, w0, strassen_levels)
        if radial_chi:
            scales = fnp.array(_chi_stratified_radial_scales(mlp.width))
            n_scale_blocks = x_half.shape[0] // mlp.width
            pre_half = pre_half * fnp.concatenate((scales,) * n_scale_blocks)[:, None]
        y_parts.append(fnp.maximum(pre_half, 0.0))
        y_parts.append(fnp.maximum(-pre_half, 0.0))
    fresh_half_blocks = 2 * (n_blocks - antithetic_blocks)
    if fresh_half_blocks:
        x_fresh = _hadamard_sign_fresh_half_blocks(mlp, fresh_half_blocks, rng)
        pre_fresh = _strassen_matmul(x_fresh, w0, strassen_levels)
        if radial_chi:
            scales = fnp.array(_chi_stratified_radial_scales(mlp.width))
            pre_fresh = pre_fresh * fnp.concatenate((scales,) * fresh_half_blocks)[:, None]
        y_parts.append(fnp.maximum(pre_fresh, 0.0))
    y = fnp.concatenate(tuple(y_parts), axis=0)

    target_mean, target_cov = _zero_mean_relu_mean_cov(w0.T @ w0)
    sample_mean = fnp.mean(y, axis=0)
    centered = y - sample_mean[None, :]
    sample_cov = _strassen_matmul(centered.T, centered, strassen_levels) / float(centered.shape[0])
    jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
    eye = _eye(mlp.width)
    sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
    target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
    recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
    x = _strassen_matmul(centered, recolor, strassen_levels) + target_mean[None, :]

    cv3_s_blocks = None
    if final_cv3:
        block_rows = 2 * mlp.width
        n_blocks = x.shape[0] // block_rows
        if n_blocks >= 2:
            x_blocks = fnp.reshape(x[: n_blocks * block_rows], (n_blocks, block_rows, mlp.width))
            block_third = fnp.mean(x_blocks * x_blocks * x_blocks, axis=1)
            pre_std = fnp.sqrt(fnp.maximum(fnp.diag(w0.T @ w0), _MIN_VARIANCE))
            target_third = (pre_std * pre_std * pre_std) * math.sqrt(2.0 / math.pi)
            target_std = fnp.sqrt(fnp.maximum(fnp.diag(target_cov), _MIN_VARIANCE))
            normalized = (block_third - target_third[None, :]) / (
                target_std[None, :] * target_std[None, :] * target_std[None, :]
            )
            cv3_s_blocks = fnp.mean(normalized, axis=1)

    posthoc = posthoc or _HadamardPosthocConfig()
    rows = [target_mean]
    final_pre = None
    for layer_idx, w in enumerate(mlp.weights[1:], start=1):
        pre = _strassen_matmul(x, w, strassen_levels)
        final_pre = pre
        x = fnp.maximum(pre, 0.0)
        if layer_idx <= exact_recolor_layers:
            target_pre_mean = target_mean @ w
            target_pre_cov = w.T @ target_cov @ w
            target_mean, target_cov = _gaussian_relu_mean_cov(target_pre_mean, target_pre_cov)
            sample_mean = fnp.mean(x, axis=0)
            centered_layer = x - sample_mean[None, :]
            sample_cov = (centered_layer.T @ centered_layer) / float(centered_layer.shape[0])
            jitter = fnp.maximum(fnp.mean(fnp.diag(target_cov)), _MIN_VARIANCE) * 1e-6
            sample_chol = fnp.linalg.cholesky(sample_cov + jitter * eye)
            target_chol = fnp.linalg.cholesky(target_cov + jitter * eye)
            recolor = fnp.linalg.inv(sample_chol.T) @ target_chol.T
            x = centered_layer @ recolor + target_mean[None, :]
        if variance_match_start_layer <= layer_idx < variance_match_start_layer + variance_match_layers:
            pre_mean = fnp.mean(pre, axis=0)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(pre_mean, fnp.mean(pre_centered * pre_centered, axis=0))
            sample_mean = fnp.mean(x, axis=0)
            centered_layer = x - sample_mean[None, :]
            sample_var = fnp.maximum(fnp.mean(centered_layer * centered_layer, axis=0), _MIN_VARIANCE)
            strength = variance_match_strength
            if layer_idx == 1 and posthoc.kurtosis_gate is not None:
                pre_z = pre_centered / fnp.sqrt(fnp.maximum(fnp.mean(pre_centered * pre_centered, axis=0), _MIN_VARIANCE))[None, :]
                excess = fnp.maximum(fnp.mean(pre_z * pre_z * pre_z * pre_z, axis=0) - 3.0, 0.0)
                strength = strength / (1.0 + posthoc.kurtosis_gate * excess)
            scale = 1.0 + strength * (fnp.sqrt(target_var / sample_var) - 1.0)
            if layer_idx == 1 and posthoc.scale_cap is not None:
                cap = posthoc.scale_cap
                scale = fnp.maximum(fnp.minimum(scale, cap), 1.0 / cap)
            x = centered_layer * scale[None, :] + sample_mean[None, :]
        if posthoc.second_variance_strength is not None and layer_idx == 2:
            pre_mean = fnp.mean(pre, axis=0)
            pre_centered = pre - pre_mean[None, :]
            target_var = _gaussian_relu_variance(pre_mean, fnp.mean(pre_centered * pre_centered, axis=0))
            sample_mean = fnp.mean(x, axis=0)
            centered_layer = x - sample_mean[None, :]
            sample_var = fnp.maximum(fnp.mean(centered_layer * centered_layer, axis=0), _MIN_VARIANCE)
            scale = 1.0 + posthoc.second_variance_strength * (fnp.sqrt(target_var / sample_var) - 1.0)
            x = centered_layer * scale[None, :] + sample_mean[None, :]
        rows.append(fnp.mean(x, axis=0))
        if mirror_layer is not None and layer_idx == mirror_layer:
            mirror_mean = rows[-1]
            x = fnp.concatenate((x, 2.0 * mirror_mean[None, :] - x), axis=0)
    if posthoc.trimmed_final:
        block_rows = 2 * mlp.width
        n_blocks = x.shape[0] // block_rows
        if n_blocks >= 3:
            final_blocks = fnp.reshape(x[: n_blocks * block_rows], (n_blocks, block_rows, mlp.width))
            block_means = fnp.mean(final_blocks, axis=1)
            sorted_means = fnp.sort(block_means, axis=0)
            rows[-1] = fnp.mean(sorted_means[1:-1], axis=0)
    if final_pre is not None and posthoc.gaussian_pull:
        pre_mean = fnp.mean(final_pre, axis=0)
        pre_centered = final_pre - pre_mean[None, :]
        gaussian_mean = _gaussian_relu_mean(pre_mean, fnp.mean(pre_centered * pre_centered, axis=0))
        rows[-1] = (1.0 - posthoc.gaussian_pull) * rows[-1] + posthoc.gaussian_pull * gaussian_mean
    if final_pre is not None and posthoc.edgeworth_blend:
        edgeworth_mean = _edgeworth_relu_mean(final_pre)
        rows[-1] = (1.0 - posthoc.edgeworth_blend) * rows[-1] + posthoc.edgeworth_blend * edgeworth_mean
    if final_cv3 and cv3_s_blocks is not None:
        block_rows = 2 * mlp.width
        n_blocks = cv3_s_blocks.shape[0]
        if x.shape[0] >= n_blocks * block_rows:
            final_blocks = fnp.reshape(x[: n_blocks * block_rows], (n_blocks, block_rows, mlp.width))
            block_final_means = fnp.mean(final_blocks, axis=1)
            s_mean = fnp.mean(cv3_s_blocks)
            s_centered = cv3_s_blocks - s_mean
            denom = fnp.maximum(fnp.mean(s_centered * s_centered), _MIN_VARIANCE)
            final_mean = rows[-1]
            cov = fnp.mean(s_centered[:, None] * (block_final_means - final_mean[None, :]), axis=0)
            beta = 0.5 * cov / denom
            rows[-1] = final_mean - beta * s_mean
    return fnp.stack(rows, axis=0)


def _deep_hadamard_blocks_for_budget(mlp: MLP, budget: int, strassen_levels: int) -> int:
    rows_per_block = 2 * mlp.width
    measured_row_cost = _DEEP_MEASURED_ROW_FLOPS.get(strassen_levels)
    if measured_row_cost is not None and mlp.width == 256 and mlp.depth == 32:
        per_block_cost = (
            rows_per_block
            * measured_row_cost
            * _DEEP_BLOCK_COST_SAFETY
            * _DEEP_RESIDUAL_BLOCK_SAFETY.get(strassen_levels, 1.0)
        )
    else:
        plain_layer_cost = rows_per_block * mlp.width * (2 * mlp.width - 1)
        strassen_discount = (7.0 / 8.0) ** max(strassen_levels, 0)
        per_block_cost = (
            mlp.depth
            * plain_layer_cost
            * strassen_discount
            * _DEEP_BLOCK_FIXED_OVERHEAD
            * _DEEP_BLOCK_COST_SAFETY
        )
    return min(max(int((0.1 * budget) // per_block_cost), 1), _DEEP_HADAMARD_MAX_BLOCKS)


def _hadamard_sample_count_for_budget(
    mlp: MLP,
    budget: int,
    strassen_levels: int | None = None,
) -> int:
    explicit = os.environ.get("WHEST_EXPERIMENT_SAMPLES")
    if explicit:
        return max(int(explicit), 2)
    block_override = os.environ.get("WHEST_HADAMARD_BLOCKS")
    if block_override:
        blocks = max(int(block_override), 1)
    elif strassen_levels is not None:
        blocks = _deep_hadamard_blocks_for_budget(mlp, budget, strassen_levels)
    else:
        blocks = _DEEP_HADAMARD_BLOCKS
    rows = blocks * 2 * mlp.width
    rough_cost_per_sample = 2.0 * mlp.depth * mlp.width * mlp.width
    max_rows = max(2, int((0.2 * budget) // rough_cost_per_sample))
    return min(rows, max_rows)


def _parse_hadamard_tokens(mode: str) -> tuple[
    int,
    int | None,
    int,
    int | None,
    bool,
    float | None,
    int,
    int,
    float,
    int | None,
    bool,
    bool,
    int | None,
    str,
    _HadamardPosthocConfig,
]:
    strassen_levels = 1
    n_blocks = None
    split_factor = 1
    mirror_layer = None
    final_cv3 = False
    variance_strength = None
    exact_recolor_layers = 0
    variance_match_start_layer = 1
    antithetic_fraction = 1.0
    hybrid_prefix_layers = None
    hybrid_skew_matched = False
    hybrid_joint_k3_matched = False
    hybrid_joint_k3_max_terms = _HYBRID_JOINT_K3_MAX_TERMS
    hybrid_joint_k3_taper = "eigen"
    scale_cap = None
    kurtosis_gate = None
    gaussian_pull = 0.0
    edgeworth_blend = 0.0
    trimmed_final = False
    second_variance_strength = None
    suffix = mode[len("hadamard") :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        return (
            strassen_levels,
            n_blocks,
            split_factor,
            mirror_layer,
            final_cv3,
            variance_strength,
            exact_recolor_layers,
            variance_match_start_layer,
            antithetic_fraction,
            hybrid_prefix_layers,
            hybrid_skew_matched,
            hybrid_joint_k3_matched,
            hybrid_joint_k3_max_terms,
            hybrid_joint_k3_taper,
            _HadamardPosthocConfig(),
        )
    for token in suffix.split("_"):
        if token.startswith("st"):
            strassen_levels = max(int(token[len("st") :]), 0)
        elif token.startswith("b"):
            n_blocks = max(int(token[len("b") :]), 1)
        elif token.startswith("split"):
            split_factor = max(int(token[len("split") :]), 1)
        elif token.startswith("mirror"):
            mirror_layer = max(int(token[len("mirror") :]), 1)
        elif token == "cv3":
            final_cv3 = True
        elif token == "noanti":
            antithetic_fraction = 0.0
        elif token == "anti50":
            antithetic_fraction = 0.5
        elif token.startswith("hybr"):
            hybrid_prefix_layers = 2
            hybrid_joint_k3_matched = True
            hybrid_joint_k3_max_terms = max(int(token[len("hybr") :]), 1)
            hybrid_joint_k3_taper = "hybr"
        elif token.startswith("hybx"):
            hybrid_prefix_layers = max(int(token[len("hybx") :]), 1)
            hybrid_joint_k3_matched = True
        elif token == "kfull":
            hybrid_joint_k3_max_terms = None
        elif token.startswith("k") and token[1:].isdigit():
            hybrid_joint_k3_max_terms = max(int(token[1:]), 1)
        elif token == "tg":
            hybrid_joint_k3_taper = "global"
        elif token == "te":
            hybrid_joint_k3_taper = "eigen"
        elif token.startswith("hybs"):
            hybrid_prefix_layers = max(int(token[len("hybs") :]), 1)
            hybrid_skew_matched = True
        elif token.startswith("hyb"):
            hybrid_prefix_layers = max(int(token[len("hyb") :]), 1)
        elif token.startswith("cap"):
            scale_cap = max(int(token[len("cap") :]) / 100.0, 1.0)
        elif token.startswith("kg"):
            kurtosis_gate = max(int(token[len("kg") :]) / 100.0, 0.0)
        elif token.startswith("gp"):
            gaussian_pull = max(int(token[len("gp") :]) / 100.0, 0.0)
        elif token.startswith("ew"):
            edgeworth_blend = max(int(token[len("ew") :]) / 100.0, 0.0)
        elif token == "tr":
            trimmed_final = True
        elif token.startswith("w2"):
            second_variance_strength = max(int(token[len("w2") :]) / 100.0, 0.0)
        elif token == "l2x":
            exact_recolor_layers = 1
            variance_match_start_layer = 99
        elif token == "l2xv":
            exact_recolor_layers = 1
            variance_match_start_layer = 2
        elif token == "l3x":
            exact_recolor_layers = 2
            variance_match_start_layer = 99
        elif token.startswith("s"):
            variance_strength = int(token[len("s") :]) / 100.0
        else:
            raise ValueError(f"Unsupported Hadamard mode token: {token}")
    return (
        strassen_levels,
        n_blocks,
        split_factor,
        mirror_layer,
        final_cv3,
        variance_strength,
        exact_recolor_layers,
        variance_match_start_layer,
        antithetic_fraction,
        hybrid_prefix_layers,
        hybrid_skew_matched,
        hybrid_joint_k3_matched,
        hybrid_joint_k3_max_terms,
        hybrid_joint_k3_taper,
        _HadamardPosthocConfig(
            scale_cap=scale_cap,
            kurtosis_gate=kurtosis_gate,
            gaussian_pull=gaussian_pull,
            edgeworth_blend=edgeworth_blend,
            trimmed_final=trimmed_final,
            second_variance_strength=second_variance_strength,
        ),
    )


class Estimator(BaseEstimator):
    """Depth-aware WhestBench estimator."""

    def setup(self, ctx: SetupContext) -> None:
        _all_terms_iso_k3()
        _terms_iso_k3()
        _terms_iso_k3_grouped()
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
                n_samples = _hadamard_sample_count_for_budget(
                    mlp,
                    budget,
                    _DEEP_STRASSEN_LEVELS,
                )
                return _hadamard_first_cov_recolored_means(
                    mlp,
                    n_samples,
                    fnp.random.default_rng(mlp.seed),
                    variance_match_layers=1,
                    variance_match_strength=_DEEP_VARIANCE_MATCH_STRENGTH,
                    strassen_levels=_DEEP_STRASSEN_LEVELS,
                )
            return _factorized_k3_propagation(mlp)
        if mode == "r1":
            return _factorized_k3_propagation(mlp)
        if mode == "hadamard_first_cov":
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_first_cov_recolored_means(mlp, n_samples, fnp.random.default_rng(mlp.seed))
        if mode == "hadamard_var1":
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_first_cov_recolored_means(
                mlp,
                n_samples,
                fnp.random.default_rng(mlp.seed),
                variance_match_layers=1,
            )
        if mode.startswith("hadamard_var1_s"):
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_first_cov_recolored_means(
                mlp,
                n_samples,
                fnp.random.default_rng(mlp.seed),
                variance_match_layers=1,
                variance_match_strength=int(mode[len("hadamard_var1_s") :]) / 100.0,
            )
        if mode == "hadamard_chi":
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_first_cov_recolored_means(
                mlp,
                n_samples,
                fnp.random.default_rng(mlp.seed),
                variance_match_layers=1,
                variance_match_strength=_DEEP_VARIANCE_MATCH_STRENGTH,
                radial_chi=True,
            )
        if mode == "hadamard_var2":
            n_samples = _hadamard_sample_count_for_budget(mlp, budget)
            return _hadamard_first_cov_recolored_means(
                mlp,
                n_samples,
                fnp.random.default_rng(mlp.seed),
                variance_match_layers=2,
            )
        if mode.startswith("hadamard_b") and mode[len("hadamard_b") :].isdigit():
            n_blocks = max(int(mode[len("hadamard_b") :]), 1)
            n_samples = n_blocks * 2 * mlp.width
            return _hadamard_first_cov_recolored_means(
                mlp,
                n_samples,
                fnp.random.default_rng(mlp.seed),
                variance_match_layers=1,
                variance_match_strength=_DEEP_VARIANCE_MATCH_STRENGTH,
            )
        if mode.startswith("hadamard"):
            (
                strassen_levels,
                n_blocks,
                split_factor,
                mirror_layer,
                final_cv3,
                variance_strength,
                exact_recolor_layers,
                variance_match_start_layer,
                antithetic_fraction,
                hybrid_prefix_layers,
                hybrid_skew_matched,
                hybrid_joint_k3_matched,
                hybrid_joint_k3_max_terms,
                hybrid_joint_k3_taper,
                posthoc,
            ) = _parse_hadamard_tokens(mode)
            if hybrid_prefix_layers is not None:
                n_hybrid_blocks = (
                    n_blocks
                    if n_blocks is not None
                    else _hybrid_analytic_blocks_for_budget(
                        mlp,
                        budget,
                        strassen_levels,
                        hybrid_prefix_layers,
                        hybrid_joint_k3_matched,
                    )
                )
                return _hybrid_analytic_prefix_hadamard_means(
                    mlp,
                    n_hybrid_blocks * 2 * mlp.width,
                    fnp.random.default_rng(mlp.seed),
                    hybrid_prefix_layers,
                    _DEEP_VARIANCE_MATCH_STRENGTH if variance_strength is None else variance_strength,
                    strassen_levels,
                    hybrid_skew_matched,
                    hybrid_joint_k3_matched,
                    hybrid_joint_k3_max_terms,
                    hybrid_joint_k3_taper,
                )
            n_samples = (
                n_blocks * 2 * mlp.width
                if n_blocks is not None
                else _hadamard_sample_count_for_budget(mlp, budget, strassen_levels)
            )
            return _hadamard_first_cov_recolored_means(
                mlp,
                n_samples,
                fnp.random.default_rng(mlp.seed),
                variance_match_layers=1,
                variance_match_start_layer=variance_match_start_layer,
                variance_match_strength=(
                    _DEEP_VARIANCE_MATCH_STRENGTH
                    if variance_strength is None
                    else variance_strength
                ),
                exact_recolor_layers=exact_recolor_layers,
                strassen_levels=strassen_levels,
                split_factor=split_factor,
                mirror_layer=mirror_layer,
                final_cv3=final_cv3,
                antithetic_fraction=antithetic_fraction,
                posthoc=posthoc,
            )
        raise ValueError(f"Unsupported WHEST_EXPERIMENT_MODE/WHEST_K3_MODE: {mode}")
