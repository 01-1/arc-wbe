"""Print a residual-time tree for the root estimator on generated MLPs."""

from __future__ import annotations

import argparse
import functools
import html
import json
import os
import sys
import time
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import flopscope as flops
import flopscope._budget as flops_budget
from whestbench import SetupContext

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import estimator
from local_engine import build_mlp


@dataclass
class Node:
    name: str
    parent: "Node | None" = field(default=None, repr=False)
    wall_s: float = 0.0
    self_s: float = 0.0
    timed_s: float = 0.0
    timed_self_s: float = 0.0
    flops: int = 0
    ops: int = 0
    non_residual: bool = False
    completely_inside_timer: bool = False
    zeroed_by_timer: bool = False
    calls: int = 0
    children: dict[str, "Node"] = field(default_factory=dict)


def _child_node(parent: Node, name: str) -> Node:
    child = parent.children.get(name)
    if child is None:
        child = Node(name, parent=parent)
        parent.children[name] = child
    return child


def _wrap(obj, name: str, label: str, stack: list[Node]) -> None:
    if not hasattr(obj, name):
        return
    fn = getattr(obj, name)
    if not callable(fn):
        return

    namespace = label.replace(".", "_")

    @functools.wraps(fn)
    def wrapped(*args, __fn=fn, __namespace=namespace, **kwargs):
        parent = stack[-1]
        node = _child_node(parent, __namespace)
        stack.append(node)
        start = time.perf_counter()
        try:
            with flops.namespace(__namespace):
                return __fn(*args, **kwargs)
        finally:
            node.wall_s += time.perf_counter() - start
            node.calls += 1
            stack.pop()

    setattr(obj, name, wrapped)


class CallTreeProfiler:
    def __init__(self, root: Node, *, hide_flopscope: bool = False, hide_numpy: bool = False):
        self.root = root
        self.hide_flopscope = hide_flopscope
        self.hide_numpy = hide_numpy
        self.frames: dict[object, tuple[Node, float, float, bool, bool, bool]] = {}
        self.op_nodes: dict[tuple[int, int], list[Node]] = {}
        self.op_owners: dict[tuple[int, int], Node] = {}
        self.active_op_timers: list[tuple[int, int]] = []
        self.active_counted_wrappers: list[Node] = []
        self.active_c_calls: dict[object, list[tuple[object, Node]]] = defaultdict(list)

    def __call__(self, frame, event, arg):
        if event == "call":
            parent_frame = frame.f_back
            if parent_frame in self.active_c_calls and self.active_c_calls[parent_frame]:
                _, parent = self.active_c_calls[parent_frame][-1]
            elif parent_frame in self.frames:
                parent, _, _, _, _, _ = self.frames[parent_frame]
            else:
                parent = self.root

            name = self._frame_name(frame)
            skip = self._skip_frame(frame)
            if skip:
                node = parent
            else:
                node = _child_node(parent, name)
                node.non_residual = parent.non_residual or self._non_residual_frame(frame)
                node.calls += 1

            now = time.perf_counter()
            is_counted_wrapper = self._is_counted_wrapper_frame(frame)
            self.frames[frame] = (
                node,
                now,
                0.0,
                skip,
                bool(self.active_counted_wrappers),
                is_counted_wrapper,
            )
            if is_counted_wrapper and not skip:
                self.active_counted_wrappers.append(node)
            return self

        if event == "return":
            entry = self.frames.pop(frame, None)
            if entry is None:
                return self
            node, start, child_s, skip, started_inside_wrapper, is_counted_wrapper = entry
            elapsed = time.perf_counter() - start
            own_elapsed = max(0.0, elapsed - child_s)
            if not is_counted_wrapper:
                self._assert_counted_wrapper_boundary(node, started_inside_wrapper)
            if not skip:
                node.completely_inside_timer = (
                    node.completely_inside_timer or started_inside_wrapper
                )
                node.wall_s += elapsed
                node.self_s += own_elapsed
            self._record_deduct_stack(frame, arg, node, skip)
            self._update_active_op_timer(frame, arg)
            if is_counted_wrapper and not skip:
                if not self.active_counted_wrappers or self.active_counted_wrappers[-1] is not node:
                    raise AssertionError(f"counted wrapper stack mismatch at {node.name}")
                self.active_counted_wrappers.pop()
            parent_frame = frame.f_back
            if parent_frame in self.active_c_calls and self.active_c_calls[parent_frame]:
                parent_key, _ = self.active_c_calls[parent_frame][-1]
                (
                    parent_node,
                    parent_start,
                    parent_child_s,
                    parent_skip,
                    parent_started_inside,
                    parent_is_counted_wrapper,
                ) = self.frames[parent_key]
                self.frames[parent_key] = (
                    parent_node,
                    parent_start,
                    parent_child_s + elapsed,
                    parent_skip,
                    parent_started_inside,
                    parent_is_counted_wrapper,
                )
            elif parent_frame in self.frames:
                (
                    parent_node,
                    parent_start,
                    parent_child_s,
                    parent_skip,
                    parent_started_inside,
                    parent_is_counted_wrapper,
                ) = self.frames[parent_frame]
                self.frames[parent_frame] = (
                    parent_node,
                    parent_start,
                    parent_child_s + elapsed,
                    parent_skip,
                    parent_started_inside,
                    parent_is_counted_wrapper,
                )
            return self

        if event == "c_call":
            parent_frame = frame
            if parent_frame not in self.frames:
                return self
            parent, _, _, _, _, _ = self.frames[parent_frame]
            name = self._c_name(arg)
            skip = self._skip_c_call(arg)
            if skip:
                node = parent
            else:
                node = _child_node(parent, name)
                node.non_residual = parent.non_residual or self._non_residual_c_call(arg)
                node.calls += 1
            now = time.perf_counter()
            key = (frame, arg)
            self.frames[key] = (
                node,
                now,
                0.0,
                skip,
                bool(self.active_counted_wrappers),
                False,
            )
            self.active_c_calls[frame].append((key, node))
            return self

        if event == "c_return" or event == "c_exception":
            entry = self.frames.pop((frame, arg), None)
            if entry is None:
                return self
            node, start, child_s, skip, started_inside_wrapper, _ = entry
            active_c_calls = self.active_c_calls.get(frame)
            if active_c_calls:
                active_c_calls.pop()
                if not active_c_calls:
                    self.active_c_calls.pop(frame, None)
            elapsed = time.perf_counter() - start
            own_elapsed = max(0.0, elapsed - child_s)
            self._assert_counted_wrapper_boundary(node, started_inside_wrapper)
            if not skip:
                node.completely_inside_timer = (
                    node.completely_inside_timer or started_inside_wrapper
                )
                node.wall_s += elapsed
                node.self_s += own_elapsed
            if frame in self.frames:
                (
                    parent_node,
                    parent_start,
                    parent_child_s,
                    parent_skip,
                    parent_started_inside,
                    parent_is_counted_wrapper,
                ) = self.frames[frame]
                self.frames[frame] = (
                    parent_node,
                    parent_start,
                    parent_child_s + elapsed,
                    parent_skip,
                    parent_started_inside,
                    parent_is_counted_wrapper,
                )
            return self

        return self

    def _record_deduct_stack(self, frame, return_value, node: Node, skip: bool) -> None:
        if frame.f_code.co_name != "deduct":
            return
        if not frame.f_code.co_filename.endswith("/flopscope/_budget.py"):
            return
        budget = getattr(return_value, "_budget", None)
        op_index = getattr(return_value, "_op_index", None)
        if budget is None or op_index is None:
            return
        nodes = []
        seen = set()
        if not skip:
            nodes.append(node)
            seen.add(id(node))
        for active_node, _, _, active_skip, _, _ in self.frames.values():
            if active_skip or id(active_node) in seen:
                continue
            nodes.append(active_node)
            seen.add(id(active_node))
        self.op_nodes[(id(budget), int(op_index))] = nodes
        if self.active_counted_wrappers:
            self.op_owners[(id(budget), int(op_index))] = self.active_counted_wrappers[-1]

    def _update_active_op_timer(self, frame, return_value) -> None:
        if not frame.f_code.co_filename.endswith("/flopscope/_budget.py"):
            return
        if frame.f_code.co_name == "__enter__":
            budget = getattr(return_value, "_budget", None)
            op_index = getattr(return_value, "_op_index", None)
            if budget is not None and op_index is not None:
                self.active_op_timers.append((id(budget), int(op_index)))
            return
        if frame.f_code.co_name == "__exit__":
            timer = frame.f_locals.get("self")
            budget = getattr(timer, "_budget", None)
            op_index = getattr(timer, "_op_index", None)
            key = None if budget is None or op_index is None else (id(budget), int(op_index))
            if key in self.active_op_timers:
                self.active_op_timers.remove(key)
            elif self.active_op_timers:
                self.active_op_timers.pop()

    def _record_timed_self(self, node: Node, elapsed_s: float) -> None:
        if elapsed_s <= 0.0:
            return
        node.timed_self_s += elapsed_s
        node.timed_s += elapsed_s
        cur = node.parent
        while cur is not None:
            cur.timed_s += elapsed_s
            cur = cur.parent

    def _assert_counted_wrapper_boundary(self, node: Node, started_inside_wrapper: bool) -> None:
        ended_inside_wrapper = bool(self.active_counted_wrappers)
        if started_inside_wrapper != ended_inside_wrapper:
            raise AssertionError(
                f"counted-wrapper boundary crossed by {node.name}: "
                f"started_inside_wrapper={started_inside_wrapper}, "
                f"ended_inside_wrapper={ended_inside_wrapper}"
            )

    @staticmethod
    def _is_counted_wrapper_frame(frame) -> bool:
        return frame.f_code is flops_budget._WRAPPED_CO

    def _skip_frame(self, frame) -> bool:
        filename = frame.f_code.co_filename
        if self.hide_flopscope and "/site-packages/flopscope/" in filename:
            return True
        if self.hide_numpy and "/site-packages/numpy/" in filename:
            return True
        return False

    def _skip_c_call(self, func) -> bool:
        return self.hide_numpy and self._non_residual_c_call(func)

    @staticmethod
    def _non_residual_frame(frame) -> bool:
        filename = frame.f_code.co_filename
        return "/site-packages/flopscope/" in filename or "/site-packages/numpy/" in filename

    @staticmethod
    def _non_residual_c_call(func) -> bool:
        module = getattr(func, "__module__", "") or ""
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", ""))
        text = f"{module}.{qualname}"
        return (
            module.startswith("numpy")
            or "numpy" in text
            or "ndarray" in text
            or "ufunc" in text
        )

    @staticmethod
    def _frame_name(frame) -> str:
        filename = frame.f_code.co_filename
        name = frame.f_code.co_name
        namespace = frame.f_locals.get("__namespace")
        if name == "wrapped" and namespace:
            return f"wrapped:{namespace}"
        if filename.endswith("/estimator.py"):
            return f"estimator.py:{frame.f_code.co_firstlineno}:{name}"
        if filename.startswith("<"):
            return f"{filename}:{name}"
        path = Path(filename)
        parent = path.parent.name
        return f"{parent}/{path.name}:{frame.f_code.co_firstlineno}:{name}"

    @staticmethod
    def _c_name(func) -> str:
        module = getattr(func, "__module__", None)
        name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
        if module:
            return f"c:{module}.{name}"
        return f"c:{name}"


def _install_wrappers(stack: list[Node]) -> None:
    for name in [
        "_factorized_k3_propagation",
        "_factored_nonlin_k3_r1_fast",
        "_final_r1_relu_mean_from_tower",
        "_relu_wick_from_stats",
        "_hermite_prob",
        "_ds_part_sum",
        "_harmonic_dslice_general",
        "_symmetrize",
        "_expand",
        "_vec_part_coef",
        "_multigraph_coef",
    ]:
        _wrap(estimator, name, name, stack)

    for class_name, methods in {
        "_FactoredThird": ["contract_w", "get_dslice", "dslice_21", "diag", "contracted_diag"],
        "_HTensor": ["contract_w", "get_dslice"],
    }.items():
        cls = getattr(estimator, class_name)
        for method in methods:
            _wrap(cls, method, f"{class_name}_{method}", stack)


def _collect_op_costs(ctx):
    cost_by_path = defaultdict(float)
    ops_by_path = defaultdict(int)
    flops_by_path = defaultdict(int)
    for op in ctx.op_log:
        parts = tuple((op.namespace or "").split(".")) if op.namespace else ()
        op_time = op.flopscope_backend_duration_s + op.flopscope_overhead_duration_s
        for index in range(1, len(parts) + 1):
            path = parts[:index]
            cost_by_path[path] += op_time
            ops_by_path[path] += 1
            flops_by_path[path] += op.flop_cost
    return cost_by_path, ops_by_path, flops_by_path


def _subtree_backend_cost(node: Node, path: tuple[str, ...], cost_by_path) -> float:
    cost = cost_by_path[path]
    for child in node.children.values():
        cost += _subtree_backend_cost(child, (*path, child.name), cost_by_path)
    return cost


def _print_residual_namespace_tree(
    node: Node,
    path: tuple[str, ...],
    cost_by_path,
    ops_by_path,
    flops_by_path,
    total_residual_s: float,
    depth: int = 0,
    max_depth: int = 8,
    min_ms: float = 0.15,
    min_flops_pct: float = 0.0,
    total_flops: int = 0,
) -> float:
    child_residuals = []
    for child in node.children.values():
        child_path = (*path, child.name)
        child_residuals.append(
            (
                child,
                child_path,
                _subtree_residual(child, child_path, cost_by_path),
            )
        )

    inclusive = _subtree_residual(node, path, cost_by_path)
    exclusive = inclusive - sum(value for _, _, value in child_residuals)

    if node.name != "<root>":
        pct = 100.0 * inclusive / max(total_residual_s, 1e-12)
        exclusive_pct = 100.0 * exclusive / max(total_residual_s, 1e-12)
        print(
            f"{'  ' * depth}{node.name} "
            f"calls={node.calls} "
            f"incl={inclusive * 1000.0:.2f}ms ({pct:.1f}%) "
            f"excl={exclusive * 1000.0:.2f}ms ({exclusive_pct:.1f}%) "
            f"wall={node.wall_s * 1000.0:.2f}ms "
            f"ops={ops_by_path[path]} "
            f"flops={flops_by_path[path]:,}"
        )

    if depth >= max_depth:
        return inclusive

    child_depth = depth if node.name == "<root>" else depth + 1
    for child, child_path, child_residual in sorted(
        child_residuals, key=lambda item: item[2], reverse=True
    ):
        child_flops = flops_by_path[child_path]
        child_flops_pct = 100.0 * child_flops / max(total_flops, 1)
        if (
            node.name != "<root>"
            and child_residual * 1000.0 < min_ms
            and (min_flops_pct <= 0 or child_flops_pct < min_flops_pct)
        ):
            continue
        _print_residual_namespace_tree(
            child,
            child_path,
            cost_by_path,
            ops_by_path,
            flops_by_path,
            total_residual_s,
            child_depth,
            max_depth,
            min_ms,
            min_flops_pct,
            total_flops,
        )
    return inclusive


def _subtree_residual(node: Node, path: tuple[str, ...], cost_by_path) -> float:
    return node.wall_s - cost_by_path[path]


def _op_timed_s(op) -> float:
    return float(getattr(op, "flopscope_backend_duration_s", 0.0) or 0.0) + float(
        getattr(op, "flopscope_overhead_duration_s", 0.0) or 0.0
    )


def _print_call_tree(
    node: dict[str, object],
    total_wall_s: float,
    depth: int = 0,
    max_depth: int = 8,
    min_ms: float = 0.15,
    min_flops_pct: float = 0.0,
    total_flops: int = 0,
) -> float:
    children = list(node["children"])
    residual_s = float(node["inclusive_s"])
    own_residual_s = float(node["exclusive_s"])
    if node["name"] != "<root>":
        pct = 100.0 * residual_s / max(total_wall_s, 1e-12)
        self_pct = 100.0 * own_residual_s / max(total_wall_s, 1e-12)
        print(
            f"{'  ' * depth}{node['name']} "
            f"calls={node['calls']} "
            f"resid={residual_s * 1000.0:.2f}ms ({pct:.1f}%) "
            f"self={own_residual_s * 1000.0:.2f}ms ({self_pct:.1f}%) "
            f"timed={float(node['timed_s']) * 1000.0:.2f}ms "
            f"flops={int(node['flops']):,}"
        )

    if depth >= max_depth:
        return residual_s

    child_depth = depth if node["name"] == "<root>" else depth + 1
    for child in sorted(children, key=lambda item: item["inclusive_s"], reverse=True):
        child_flops_pct = 100.0 * int(child["flops"]) / max(total_flops, 1)
        if (
            node["name"] != "<root>"
            and float(child["inclusive_s"]) * 1000.0 < min_ms
            and (min_flops_pct <= 0 or child_flops_pct < min_flops_pct)
        ):
            continue
        _print_call_tree(
            child,
            total_wall_s,
            child_depth,
            max_depth,
            min_ms,
            min_flops_pct,
            total_flops,
        )
    return residual_s


def _print_function_table(
    node: dict[str, object],
    total_wall_s: float,
    *,
    max_rows: int = 80,
    min_ms: float = 0.15,
    min_flops_pct: float = 0.0,
    total_flops: int = 0,
) -> None:
    rows = _function_table_data(node)["children"]
    shown = 0
    for row in rows:
        residual_s = float(row["inclusive_s"])
        own_residual_s = float(row["exclusive_s"])
        flops_pct = 100.0 * int(row["flops"]) / max(total_flops, 1)
        if residual_s * 1000.0 < min_ms and (min_flops_pct <= 0 or flops_pct < min_flops_pct):
            continue
        pct = 100.0 * residual_s / max(total_wall_s, 1e-12)
        self_pct = 100.0 * own_residual_s / max(total_wall_s, 1e-12)
        print(
            f"{row['name']} "
            f"calls={row['calls']} "
            f"resid={residual_s * 1000.0:.2f}ms ({pct:.1f}%) "
            f"self={own_residual_s * 1000.0:.2f}ms ({self_pct:.1f}%) "
            f"timed={float(row['timed_s']) * 1000.0:.2f}ms "
            f"flops={int(row['flops']):,}"
        )
        shown += 1
        if shown >= max_rows:
            break


def _zero_call_subtree(node: Node) -> bool:
    return node.completely_inside_timer or node.zeroed_by_timer


def _call_node_residual(node: Node, residual_scale: float = 1.0, zeroed: bool = False) -> float:
    if zeroed or _zero_call_subtree(node):
        return 0.0
    return (node.wall_s - node.timed_s) * residual_scale


def _call_node_flops(node: Node) -> int:
    if node.flops:
        return node.flops
    return sum(_call_node_flops(child) for child in node.children.values())


def _annotate_call_tree_op_times(profiler: CallTreeProfiler, ctx) -> None:
    budget_id = id(ctx)
    missing_owners = []
    for op_index, op in enumerate(ctx.op_log):
        nodes = profiler.op_nodes.get((budget_id, op_index))
        if not nodes:
            continue
        owner = profiler.op_owners.get((budget_id, op_index))
        if owner is None:
            missing_owners.append((op_index, op.op_name, op.namespace))
            continue
        op_time_s = _op_timed_s(op)
        owner.timed_s += op_time_s
        flops_count = int(getattr(op, "flop_cost", 0) or 0)
        for cur in nodes:
            cur.flops += flops_count
            cur.ops += 1
    if missing_owners:
        preview = ", ".join(
            f"#{op_index} {op_name} namespace={namespace!r}"
            for op_index, op_name, namespace in missing_owners[:10]
        )
        suffix = "" if len(missing_owners) <= 10 else f", ... {len(missing_owners)} total"
        raise AssertionError(f"ops without counted-wrapper owner: {preview}{suffix}")


def _finalize_call_tree_timed(node: Node) -> float:
    child_timed_s = sum(_finalize_call_tree_timed(child) for child in node.children.values())
    if _zero_call_subtree(node):
        node.timed_s = node.wall_s
        node.timed_self_s = node.self_s
        return node.timed_s
    node.timed_s = max(node.timed_s, child_timed_s)
    node.timed_self_s = node.timed_s - child_timed_s
    return node.timed_s


def _print_flops_tree(
    node: dict[str, object],
    *,
    total_flops: int,
    depth: int = 0,
    max_depth: int = 8,
    min_flops_pct: float = 0.0,
) -> None:
    if node["name"] != "<root>":
        flops_pct = 100.0 * int(node["flops"]) / max(total_flops, 1)
        elapsed_s = float(node["inclusive_s"])
        print(
            f"{'  ' * depth}{node['name']} "
            f"ops={node['ops']:,} "
            f"flops={node['flops']:,} ({flops_pct:.1f}%) "
            f"flopscope_time={elapsed_s * 1000.0:.2f}ms"
        )

    if depth >= max_depth:
        return

    child_depth = depth if node["name"] == "<root>" else depth + 1
    for child in node["children"]:
        child_flops_pct = 100.0 * int(child["flops"]) / max(total_flops, 1)
        if node["name"] != "<root>" and child_flops_pct < min_flops_pct:
            continue
        _print_flops_tree(
            child,
            total_flops=total_flops,
            depth=child_depth,
            max_depth=max_depth,
            min_flops_pct=min_flops_pct,
        )


def _namespace_tree_data(node: Node, path: tuple[str, ...], cost_by_path, ops_by_path, flops_by_path):
    children = [
        _namespace_tree_data(child, (*path, child.name), cost_by_path, ops_by_path, flops_by_path)
        for child in node.children.values()
    ]
    inclusive = _subtree_residual(node, path, cost_by_path)
    exclusive = inclusive - sum(child["inclusive_s"] for child in children)
    return {
        "name": node.name,
        "calls": node.calls,
        "wall_s": node.wall_s,
        "self_s": node.self_s,
        "inclusive_s": inclusive,
        "exclusive_s": exclusive,
        "ops": ops_by_path[path],
        "flops": flops_by_path[path],
        "children": sorted(children, key=lambda child: child["inclusive_s"], reverse=True),
    }


def _call_tree_data(node: Node, residual_scale: float = 1.0, zeroed: bool = False):
    zeroed = zeroed or _zero_call_subtree(node)
    children = [_call_tree_data(child, residual_scale, zeroed) for child in node.children.values()]
    inclusive_s = _call_node_residual(node, residual_scale, zeroed)
    own_residual_s = 0.0 if zeroed else (node.self_s - node.timed_self_s) * residual_scale
    return {
        "name": node.name,
        "calls": node.calls,
        "wall_s": node.wall_s,
        "self_s": node.self_s,
        "timed_s": node.timed_s,
        "timed_self_s": node.timed_self_s,
        "inclusive_s": inclusive_s,
        "exclusive_s": own_residual_s,
        "ops": node.ops if node.ops else sum(child["ops"] for child in children),
        "flops": node.flops if node.flops else sum(child["flops"] for child in children),
        "non_residual": node.non_residual,
        "children": sorted(children, key=lambda child: child["inclusive_s"], reverse=True),
    }


def _function_table_data(call_tree: dict[str, object]) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}

    def visit(node: dict[str, object]) -> None:
        name = str(node["name"])
        if name != "<root>":
            row = rows.get(name)
            if row is None:
                row = {
                    "name": name,
                    "calls": 0,
                    "wall_s": 0.0,
                    "self_s": 0.0,
                    "timed_s": 0.0,
                    "timed_self_s": 0.0,
                    "inclusive_s": 0.0,
                    "exclusive_s": 0.0,
                    "ops": 0,
                    "flops": 0,
                    "children": [],
                }
                rows[name] = row
            row["calls"] = int(row["calls"]) + int(node["calls"])
            row["wall_s"] = float(row["wall_s"]) + float(node["wall_s"])
            row["self_s"] = float(row["self_s"]) + float(node["self_s"])
            row["timed_s"] = float(row["timed_s"]) + float(node["timed_s"])
            row["timed_self_s"] = float(row["timed_self_s"]) + float(node["timed_self_s"])
            row["inclusive_s"] = float(row["inclusive_s"]) + float(node["inclusive_s"])
            row["exclusive_s"] = float(row["exclusive_s"]) + float(node["exclusive_s"])
            row["ops"] = int(row["ops"]) + int(node["ops"])
            row["flops"] = int(row["flops"]) + int(node["flops"])
        for child in node["children"]:
            visit(child)

    visit(call_tree)
    return {
        "name": "<root>",
        "calls": sum(int(row["calls"]) for row in rows.values()),
        "wall_s": sum(float(row["wall_s"]) for row in rows.values()),
        "self_s": sum(float(row["self_s"]) for row in rows.values()),
        "timed_s": sum(float(row["timed_s"]) for row in rows.values()),
        "timed_self_s": sum(float(row["timed_self_s"]) for row in rows.values()),
        "inclusive_s": sum(float(row["inclusive_s"]) for row in rows.values()),
        "exclusive_s": sum(float(row["exclusive_s"]) for row in rows.values()),
        "ops": sum(int(row["ops"]) for row in rows.values()),
        "flops": sum(int(row["flops"]) for row in rows.values()),
        "children": sorted(
            rows.values(),
            key=lambda row: (float(row["exclusive_s"]), float(row["inclusive_s"])),
            reverse=True,
        ),
    }


def _assert_additive_call_tree(node: dict[str, object], tolerance_s: float = 1e-5) -> None:
    children = node["children"]
    for child in children:
        _assert_additive_call_tree(child, tolerance_s)
    expected = float(node["exclusive_s"]) + sum(float(child["inclusive_s"]) for child in children)
    actual = float(node["inclusive_s"])
    if node["name"] == "<root>":
        return
    if abs(actual - expected) > tolerance_s:
        raise AssertionError(
            f"non-additive call tree at {node['name']}: "
            f"inclusive={actual:.12f}, exclusive+children={expected:.12f}"
        )


def _empty_flops_tree_node(name: str) -> dict[str, object]:
    return {
        "name": name,
        "calls": 0,
        "wall_s": 0.0,
        "self_s": 0.0,
        "inclusive_s": 0.0,
        "exclusive_s": 0.0,
        "backend_s": 0.0,
        "overhead_s": 0.0,
        "ops": 0,
        "flops": 0,
        "children": [],
        "_children_by_name": {},
    }


def _flops_tree_child(parent: dict[str, object], name: str) -> dict[str, object]:
    children_by_name = parent["_children_by_name"]
    if name not in children_by_name:
        child = _empty_flops_tree_node(name)
        children_by_name[name] = child
        parent["children"].append(child)
    return children_by_name[name]


def _flops_op_label(op) -> str:
    details = []
    for attr in ("subscripts", "shapes"):
        value = getattr(op, attr, None)
        if value:
            details.append(str(value))
    if details:
        return f"{op.op_name} {' '.join(details)}"
    return str(op.op_name)


def _flops_tree_data(ctx) -> dict[str, object]:
    root = _empty_flops_tree_node("<root>")
    for op in ctx.op_log:
        flop_cost = int(getattr(op, "flop_cost", 0) or 0)
        backend_s = float(getattr(op, "flopscope_backend_duration_s", 0.0) or 0.0)
        overhead_s = float(getattr(op, "flopscope_overhead_duration_s", 0.0) or 0.0)
        namespace = getattr(op, "namespace", "") or "<no namespace>"
        parts = tuple(part for part in str(namespace).split(".") if part) or ("<no namespace>",)
        path_nodes = [root]
        node = root
        for part in parts:
            node = _flops_tree_child(node, part)
            path_nodes.append(node)
        leaf = _flops_tree_child(node, _flops_op_label(op))
        path_nodes.append(leaf)
        for item in path_nodes:
            item["flops"] += flop_cost
            item["ops"] += 1
            item["calls"] = item["ops"]
            item["backend_s"] += backend_s
            item["overhead_s"] += overhead_s
            item["inclusive_s"] = item["backend_s"] + item["overhead_s"]

    def finish(node: dict[str, object]) -> dict[str, object]:
        node["children"] = sorted(
            [finish(child) for child in node["children"]],
            key=lambda child: (child["flops"], child["ops"]),
            reverse=True,
        )
        node.pop("_children_by_name", None)
        return node

    return finish(root)


def _write_html_report(
    output_path: Path,
    *,
    metadata: dict[str, object],
    namespace_tree: dict[str, object],
    call_tree: dict[str, object],
    function_table: dict[str, object],
    flops_tree: dict[str, object],
) -> None:
    payload = json.dumps(
        {
            "metadata": metadata,
            "namespaceTree": namespace_tree,
            "callTree": call_tree,
            "functionTable": function_table,
            "flopsTree": flops_tree,
        }
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Residual Profile</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f7f7f4;
  --ink: #1f2528;
  --muted: #657077;
  --line: #d8ddd9;
  --accent: #2f7d68;
  --accent-2: #9a5b25;
  --panel: #ffffff;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #151716;
    --ink: #edf0ed;
    --muted: #a2aaa5;
    --line: #343a36;
    --accent: #62c6a8;
    --accent-2: #e1a463;
    --panel: #1f2321;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(10px);
}}
h1 {{ margin: 0 0 8px; font-size: 20px; letter-spacing: 0; }}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
}}
.pill {{
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 4px 7px;
  background: var(--panel);
}}
main {{ padding: 18px; max-width: 1500px; margin: 0 auto; }}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 14px;
}}
button, input, select {{
  font: inherit;
  color: var(--ink);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 9px;
}}
input {{ min-width: 260px; }}
.tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
.tab[aria-selected="true"] {{
  border-color: var(--accent);
  color: var(--accent);
}}
.tree {{
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  overflow: auto;
}}
.row {{
  display: grid;
  grid-template-columns: minmax(520px, 1fr) 82px 150px 110px 150px 90px;
  gap: 10px;
  align-items: center;
  min-width: 1120px;
  min-height: 28px;
  padding: 2px 10px;
  border-top: 1px solid color-mix(in srgb, var(--line) 55%, transparent);
  font-size: 13px;
}}
.row.header {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--panel);
  color: var(--muted);
  font-weight: 650;
  border-top: 0;
}}
.name {{
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}}
.name-text {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.twisty {{
  width: 20px;
  height: 20px;
  display: inline-grid;
  place-items: center;
  border: 0;
  padding: 0;
  background: transparent;
  color: var(--muted);
}}
.barcell {{
  display: grid;
  grid-template-columns: 78px 1fr;
  gap: 8px;
  align-items: center;
}}
.bar {{
  position: relative;
  height: 8px;
  background: color-mix(in srgb, var(--accent) 20%, transparent);
  overflow: hidden;
}}
.fill {{
  position: absolute;
  inset: 0 auto 0 0;
  display: block;
  height: 100%;
  background: var(--accent);
}}
.fill.exclusive {{
  background: var(--accent-2);
}}
.fill.flops {{
  background: var(--accent-2);
}}
.muted {{ color: var(--muted); }}
.hidden {{ display: none; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
<header>
  <h1>Residual Profile</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="controls">
    <input id="filter" placeholder="Filter function names">
    <label class="muted">Min ms <input id="minMs" type="number" min="0" step="0.1"></label>
    <label class="muted">Min FLOPs % <input id="minFlops" type="number" min="0" step="0.1"></label>
    <button id="expand">Expand All</button>
    <button id="collapse">Collapse All</button>
  </div>
  <div class="tabs">
    <button class="tab" data-view="namespace" aria-selected="true">Residual Namespace Tree</button>
    <button class="tab" data-view="call" aria-selected="false">Unified Call Tree</button>
    <button class="tab" data-view="functions" aria-selected="false">Per-Function View</button>
    <button class="tab" data-view="flops" aria-selected="false">Full FLOPs Tree</button>
  </div>
  <div class="tree" id="tree"></div>
</main>
<script>
const data = {payload};
const state = {{
  view: "namespace",
  collapsed: new Set(),
  filter: "",
  minMs: Number(data.metadata.min_ms || 0),
  minFlops: Number(data.metadata.min_flops_pct || 0)
}};

function fmtMs(seconds) {{ return `${{(seconds * 1000).toFixed(2)}}ms`; }}
function fmtInt(value) {{ return Math.round(value).toLocaleString(); }}
function pct(value, total) {{ return total > 0 ? 100 * value / total : 0; }}
function metric(node) {{ return node.inclusive_s || 0; }}
function flopMetric(node) {{ return node.flops || 0; }}
function isFlopsView() {{ return state.view === "flops"; }}
function isFunctionView() {{ return state.view === "functions"; }}
function rootForView() {{
  if (state.view === "namespace") return data.namespaceTree;
  if (state.view === "functions") return data.functionTable;
  if (state.view === "flops") return data.flopsTree;
  return data.callTree;
}}
function totalForView() {{
  return isFlopsView() ? data.metadata.flops : data.metadata.residual_wall_time_s;
}}
function rowVisible(node) {{
  const flopPct = pct(flopMetric(node), data.metadata.flops);
  const timeOk = !isFlopsView() && metric(node) * 1000 >= state.minMs;
  const flopsOk = state.minFlops > 0 && flopPct >= state.minFlops;
  const minOk = node.name === "<root>" || timeOk || flopsOk;
  const filterOk = !state.filter || node.name.toLowerCase().includes(state.filter);
  if (minOk && filterOk) return true;
  return node.children.some(rowVisible);
}}
function renderMeta() {{
  const meta = data.metadata;
  document.getElementById("meta").innerHTML = [
    `width=${{meta.width}} depth=${{meta.depth}} seed=${{meta.seed}} mode=${{meta.mode}}`,
    `FLOPs=${{fmtInt(meta.flops)}}`,
    `ops=${{fmtInt(meta.ops)}}`,
    `backend=${{fmtMs(meta.backend_time_s)}}`,
    `overhead=${{fmtMs(meta.overhead_time_s)}}`,
    `residual=${{fmtMs(meta.residual_wall_time_s)}}`
  ].map(text => `<span class="pill"><code>${{text}}</code></span>`).join("");
}}
function render() {{
  const tree = document.getElementById("tree");
  const root = rootForView();
  let maxVisible = 0;
  let maxVisibleFlops = 0;
  function scan(node) {{
    if (node.name !== "<root>" && rowVisible(node)) {{
      maxVisible = Math.max(maxVisible, metric(node));
      maxVisibleFlops = Math.max(maxVisibleFlops, flopMetric(node));
    }}
    for (const child of node.children) scan(child);
  }}
  scan(root);
  const countTitle = isFlopsView() ? "Ops" : "Calls";
  const firstMetricTitle = isFlopsView() ? "Scope Time" : "Residual";
  const secondMetricTitle = isFlopsView() ? "Self" : "Exclusive";
  const rows = [`<div class="row header">
    <div>Function</div><div>${{countTitle}}</div><div>${{firstMetricTitle}}</div><div>${{secondMetricTitle}}</div><div>FLOPs</div><div>Ops</div>
  </div>`];
  function walk(node, depth, path) {{
    if (node.name !== "<root>") {{
      if (!rowVisible(node)) return;
      const id = path.join("/");
      const hasChildren = !isFunctionView() && node.children.some(rowVisible);
      const collapsed = !isFunctionView() && state.collapsed.has(id);
      const value = metric(node);
      const selfValue = isFlopsView() ? node.self_s : node.exclusive_s;
      const timeLabel = isFlopsView() && value === 0 ? "" : fmtMs(value);
      const selfLabel = isFlopsView() && selfValue === 0 ? "" : fmtMs(selfValue);
      const residualWidth = Math.min(100, pct(value, maxVisible));
      const exclusiveWidth = isFlopsView() ? 0 : Math.min(residualWidth, pct(Math.max(0, selfValue), maxVisible));
      rows.push(`<div class="row" data-path="${{htmlEscape(id)}}">
        <div class="name" style="padding-left:${{depth * 18}}px">
          ${{hasChildren ? `<button class="twisty" data-toggle="${{htmlEscape(id)}}">${{collapsed ? "▸" : "▾"}}</button>` : `<span class="twisty"></span>`}}
          <span class="name-text" title="${{htmlEscape(node.name)}}">${{htmlEscape(node.name)}}</span>
        </div>
        <div>${{fmtInt(node.calls)}}</div>
        <div class="barcell"><span>${{timeLabel}}</span><span class="bar"><span class="fill" style="width:${{residualWidth}}%"></span>${{exclusiveWidth > 0 ? `<span class="fill exclusive" style="width:${{exclusiveWidth}}%"></span>` : ""}}</span></div>
        <div>${{selfLabel}}</div>
        <div class="barcell"><span>${{node.flops ? fmtInt(node.flops) : ""}}</span><span class="bar"><span class="fill flops" style="width:${{Math.min(100, pct(flopMetric(node), maxVisibleFlops))}}%"></span></span></div>
        <div>${{node.ops ? fmtInt(node.ops) : ""}}</div>
      </div>`);
      if (collapsed) return;
    }}
    for (const child of node.children) walk(child, isFunctionView() || node.name === "<root>" ? depth : depth + 1, [...path, child.name]);
  }}
  walk(root, 0, []);
  tree.innerHTML = rows.join("");
  for (const button of tree.querySelectorAll("[data-toggle]")) {{
    button.addEventListener("click", () => {{
      const id = button.getAttribute("data-toggle");
      if (state.collapsed.has(id)) state.collapsed.delete(id);
      else state.collapsed.add(id);
      render();
    }});
  }}
}}
function htmlEscape(value) {{
  return String(value).replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
document.getElementById("filter").addEventListener("input", event => {{
  state.filter = event.target.value.trim().toLowerCase();
  render();
}});
document.getElementById("minMs").addEventListener("input", event => {{
  state.minMs = Number(event.target.value || 0);
  render();
}});
document.getElementById("minFlops").addEventListener("input", event => {{
  state.minFlops = Number(event.target.value || 0);
  render();
}});
document.getElementById("expand").addEventListener("click", () => {{ state.collapsed.clear(); render(); }});
document.getElementById("collapse").addEventListener("click", () => {{
  state.collapsed.clear();
  function collect(node, path) {{
    if (node.name !== "<root>") state.collapsed.add(path.join("/"));
    for (const child of node.children) collect(child, [...path, child.name]);
  }}
  collect(rootForView(), []);
  render();
}});
for (const tab of document.querySelectorAll(".tab")) {{
  tab.addEventListener("click", () => {{
    state.view = tab.dataset.view;
    state.collapsed.clear();
    document.querySelectorAll(".tab").forEach(item => item.setAttribute("aria-selected", String(item === tab)));
    render();
  }});
}}
renderMeta();
document.getElementById("minMs").value = state.minMs;
document.getElementById("minFlops").value = state.minFlops;
render();
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=68_000_000_000)
    parser.add_argument("--mode", default="")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-ms", type=float, default=5.0)
    parser.add_argument(
        "--min-flops",
        type=float,
        default=1.0,
        help="Minimum FLOP percentage of total run FLOPs for a row to be shown.",
    )
    parser.add_argument("--hide-flopscope", action="store_true")
    parser.add_argument("--hide-numpy", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("profile_residual_tree.html"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.mode:
        os.environ["WHEST_K3_MODE"] = args.mode
    else:
        os.environ.pop("WHEST_K3_MODE", None)
    os.environ.pop("WHEST_EXPERIMENT_MODE", None)

    root = Node("<root>")
    stack = [root]
    _install_wrappers(stack)

    mlp = build_mlp(args.width, args.depth, args.seed)
    est = estimator.Estimator()
    est.setup(
        SetupContext(
            width=args.width,
            depth=args.depth,
            flop_budget=args.budget,
            api_version="residual-tree-profile",
            seed=args.seed,
        )
    )

    with flops.BudgetContext(flop_budget=args.budget, quiet=True):
        est.predict(mlp, args.budget)

    root.children.clear()
    call_root = Node("<root>")
    profiler = CallTreeProfiler(
        call_root,
        hide_flopscope=args.hide_flopscope,
        hide_numpy=args.hide_numpy,
    )
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as ctx:
        sys.setprofile(profiler)
        est.predict(mlp, args.budget)
        sys.setprofile(None)

    cost_by_path, ops_by_path, flops_by_path = _collect_op_costs(ctx)
    flops_tree = _flops_tree_data(ctx)
    _annotate_call_tree_op_times(profiler, ctx)
    _finalize_call_tree_timed(call_root)
    call_tree = _call_tree_data(call_root)
    _assert_additive_call_tree(call_tree)
    function_table = _function_table_data(call_tree)

    print(f"MLP width={args.width} depth={args.depth} seed={args.seed} mode={args.mode or 'default'}")
    print(f"flops={ctx.flops_used:,} ops={len(ctx.op_log):,}")
    print(f"backend_time_s={ctx.flopscope_backend_time:.6f}")
    print(f"overhead_time_s={ctx.flopscope_overhead_time:.6f}")
    print(f"residual_wall_time_s={ctx.residual_wall_time:.6f}\n")
    print("Residual namespace tree: estimator wrappers with backend+overhead subtracted")
    _print_residual_namespace_tree(
        root,
        (),
        cost_by_path,
        ops_by_path,
        flops_by_path,
        ctx.residual_wall_time,
        max_depth=args.max_depth,
        min_ms=args.min_ms,
        min_flops_pct=args.min_flops,
        total_flops=ctx.flops_used,
    )
    print()
    print("Call tree: estimator and non-estimator Python/C calls in the same tree")
    print("Note: residual time subtracts call-tree work measured inside flopscope counted wrappers.")
    _print_call_tree(
        call_tree,
        ctx.residual_wall_time,
        max_depth=args.max_depth,
        min_ms=args.min_ms,
        min_flops_pct=args.min_flops,
        total_flops=ctx.flops_used,
    )
    print()
    print("Per-function view: flat aggregation of unified call tree rows by function name")
    _print_function_table(
        call_tree,
        ctx.residual_wall_time,
        min_ms=args.min_ms,
        min_flops_pct=args.min_flops,
        total_flops=ctx.flops_used,
    )
    print()
    print("FLOPs tree: exact flopscope op log by namespace and op")
    _print_flops_tree(
        flops_tree,
        total_flops=ctx.flops_used,
        max_depth=args.max_depth,
        min_flops_pct=args.min_flops,
    )

    _write_html_report(
        args.output,
        metadata={
            "width": args.width,
            "depth": args.depth,
            "seed": args.seed,
            "mode": args.mode or "default",
            "flops": ctx.flops_used,
            "ops": len(ctx.op_log),
            "backend_time_s": ctx.flopscope_backend_time,
            "overhead_time_s": ctx.flopscope_overhead_time,
            "residual_wall_time_s": ctx.residual_wall_time,
            "min_ms": args.min_ms,
            "min_flops_pct": args.min_flops,
        },
        namespace_tree=_namespace_tree_data(root, (), cost_by_path, ops_by_path, flops_by_path),
        call_tree=call_tree,
        function_table=function_table,
        flops_tree=flops_tree,
    )
    print(f"\nWrote HTML report: {args.output.resolve()}")
    if not args.no_browser:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
