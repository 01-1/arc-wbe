"""Print a residual-time tree for the root estimator on generated MLPs."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import re
import sys
import time
import types
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import flopscope as flops
import flopscope._budget as flops_budget
import flopscope._ndarray as flops_ndarray
from whestbench import SetupContext

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import estimator
from local_engine import build_mlp

COUNTED_WRAPPER_CODES = {flops_budget._WRAPPED_CO}
_MISSING = object()


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


@dataclass
class TimerRef:
    nodes: tuple[Node, ...] = ()
    val: float = 0.0
    outer: bool = True


def _child_node(parent: Node, name: str) -> Node:
    child = parent.children.get(name)
    if child is None:
        child = Node(name, parent=parent)
        parent.children[name] = child
    return child


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
            if is_counted_wrapper:
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
            if is_counted_wrapper:
                if not self.active_counted_wrappers or self.active_counted_wrappers[-1] is not node:
                    raise AssertionError(f"counted wrapper stack mismatch at {node.name}")
                self.active_counted_wrappers.pop()
            parent_frame = frame.f_back
            parent_child_elapsed = child_s if skip else elapsed
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
                    parent_child_s + parent_child_elapsed,
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
                    parent_child_s + parent_child_elapsed,
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
            parent_child_elapsed = child_s if skip else elapsed
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
                    parent_child_s + parent_child_elapsed,
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
        return frame.f_code in COUNTED_WRAPPER_CODES

    def _skip_frame(self, frame) -> bool:
        filename = frame.f_code.co_filename
        if frame.f_code in COUNTED_WRAPPER_CODES and filename == __file__:
            return True
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
    visible_children = []
    other = _empty_call_tree_other_node()
    for child in sorted(children, key=lambda item: item["inclusive_s"], reverse=True):
        if float(child["inclusive_s"]) * 1000.0 < min_ms:
            _add_call_tree_other_node(other, child)
            continue
        visible_children.append(child)

    for child in visible_children:
        _print_call_tree(
            child,
            total_wall_s,
            child_depth,
            max_depth,
            min_ms,
            total_flops,
        )
    if other["calls"]:
        _print_call_tree_other(other, total_wall_s, child_depth)
    return residual_s


def _empty_call_tree_other_node() -> dict[str, object]:
    return {
        "name": "Other (< min ms)",
        "calls": 0,
        "inclusive_s": 0.0,
        "exclusive_s": 0.0,
        "timed_s": 0.0,
        "flops": 0,
    }


def _add_call_tree_other_node(other: dict[str, object], node: dict[str, object]) -> None:
    other["calls"] = int(other["calls"]) + int(node["calls"])
    other["inclusive_s"] = float(other["inclusive_s"]) + float(node["inclusive_s"])
    other["exclusive_s"] = float(other["exclusive_s"]) + float(node["exclusive_s"])
    other["timed_s"] = float(other["timed_s"]) + float(node["timed_s"])
    other["flops"] = int(other["flops"]) + int(node["flops"])


def _print_call_tree_other(node: dict[str, object], total_wall_s: float, depth: int) -> None:
    residual_s = float(node["inclusive_s"])
    own_residual_s = float(node["exclusive_s"])
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


def _print_function_table(
    node: dict[str, object],
    total_wall_s: float,
    *,
    max_rows: int = 80,
    min_ms: float = 0.15,
    total_flops: int = 0,
) -> None:
    rows = _function_table_data(node)["children"]
    shown = 0
    for row in rows:
        residual_s = float(row["inclusive_s"])
        own_residual_s = float(row["exclusive_s"])
        if residual_s * 1000.0 < min_ms:
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


class TargetedCallProfiler:
    def __init__(self) -> None:
        self.root = Node("<root>")
        self.stack: list[tuple[Node, float, float, bool]] = []
        self.active_timer_depth = 0
        self.timer_outer_stack: list[bool] = []
        self.timer_refs: dict[tuple[int, ...], TimerRef] = {}

    def enter_call(self, name: str) -> None:
        parent = self.stack[-1][0] if self.stack else self.root
        node = _child_node(parent, name)
        node.calls += 1
        self.stack.append((node, time.perf_counter(), 0.0, self.active_timer_depth > 0))

    def exit_call(self, name: str) -> None:
        if not self.stack:
            raise AssertionError(f"targeted call stack underflow at {name}")
        node, start, child_s, started_inside_timer = self.stack.pop()
        if node.name != name:
            raise AssertionError(f"targeted call stack mismatch: expected {node.name}, got {name}")
        ended_inside_timer = self.active_timer_depth > 0
        if started_inside_timer != ended_inside_timer:
            raise AssertionError(
                f"counted-wrapper boundary crossed by targeted call {name}: "
                f"started_inside_timer={started_inside_timer}, ended_inside_timer={ended_inside_timer}"
            )
        elapsed = time.perf_counter() - start
        node.wall_s += elapsed
        node.self_s += max(0.0, elapsed - child_s)
        node.completely_inside_timer = node.completely_inside_timer or started_inside_timer
        if self.stack:
            parent_node, parent_start, parent_child_s, parent_started_inside = self.stack[-1]
            self.stack[-1] = (
                parent_node,
                parent_start,
                parent_child_s + elapsed,
                parent_started_inside,
            )

    def enter_timer(self) -> None:
        self.timer_outer_stack.append(self.active_timer_depth == 0)
        self.active_timer_depth += 1

    def exit_timer(self) -> TimerRef:
        if self.active_timer_depth <= 0:
            raise AssertionError("targeted timer stack underflow")
        self.active_timer_depth -= 1
        is_outer = self.timer_outer_stack.pop()
        if not is_outer:
            return TimerRef(outer=False)
        nodes = tuple(item[0] for item in self.stack)
        key = tuple(id(node) for node in nodes)
        ref = self.timer_refs.get(key)
        if ref is None:
            ref = TimerRef(nodes=nodes)
            self.timer_refs[key] = ref
        return ref

    def apply_timer_refs(self) -> None:
        for ref in self.timer_refs.values():
            for node in ref.nodes:
                node.timed_s += ref.val


def _targeted_function_specs() -> list[tuple[object, str, str]]:
    specs: list[tuple[object, str, str]] = []
    seen: set[tuple[int, str]] = set()

    def maybe_add(owner: object, attr_name: str, fn: object) -> None:
        if not inspect.isfunction(fn):
            return
        code = getattr(fn, "__code__", None)
        if code is None or not code.co_filename.endswith("/estimator.py"):
            return
        key = (id(owner), attr_name)
        if key in seen:
            return
        seen.add(key)
        specs.append((owner, attr_name, f"estimator.py:{code.co_firstlineno}:{code.co_name}"))

    for attr_name, value in vars(estimator).items():
        maybe_add(estimator, attr_name, value)
        if not isinstance(value, type):
            continue
        for class_attr_name, class_value in vars(value).items():
            maybe_add(value, class_attr_name, class_value)

    return specs


def _install_targeted_function_wrappers(
    profiler: TargetedCallProfiler,
    *,
    allowed_labels: set[str] | None = None,
) -> list[tuple[object, str, object]]:
    replacements: list[tuple[object, str, object]] = []
    helper_name = "__targeted_local_wrapper"
    helper_original = getattr(estimator, helper_name, _MISSING)
    setattr(
        estimator,
        helper_name,
        functools.partial(_targeted_local_wrapper, profiler),
    )
    replacements.append((estimator, helper_name, helper_original))
    _install_targeted_local_function_wrappers(
        replacements,
        profiler,
        allowed_labels=allowed_labels,
    )
    for owner, attr_name, label in _targeted_function_specs():
        if allowed_labels is not None and label not in allowed_labels:
            continue
        original = getattr(owner, attr_name)

        @functools.wraps(original)
        def wrapped(*args, __fn=original, __label=label, **kwargs):
            profiler.enter_call(__label)
            try:
                return __fn(*args, **kwargs)
            finally:
                profiler.exit_call(__label)

        setattr(owner, attr_name, wrapped)
        replacements.append((owner, attr_name, original))
    return replacements


def _local_function_labels(fn: object) -> dict[tuple[int, str], str]:
    labels: dict[tuple[int, str], str] = {}

    def visit(code) -> None:
        for const in code.co_consts:
            if not isinstance(const, types.CodeType):
                continue
            if const.co_filename.endswith("/estimator.py"):
                labels[(const.co_firstlineno, const.co_name)] = (
                    f"estimator.py:{const.co_firstlineno}:{const.co_name}"
                )
            visit(const)

    code = getattr(fn, "__code__", None)
    if code is not None:
        visit(code)
    return labels


def _targeted_local_wrapper(profiler: TargetedCallProfiler, label: str):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            profiler.enter_call(label)
            try:
                return fn(*args, **kwargs)
            finally:
                profiler.exit_call(label)

        return wrapped

    return decorate


def _patch_function_source_for_local_wrappers(
    fn: object,
    profiler: TargetedCallProfiler,
    *,
    allowed_labels: set[str] | None = None,
):
    local_labels = _local_function_labels(fn)
    selected = {
        key: label
        for key, label in local_labels.items()
        if allowed_labels is None or label in allowed_labels
    }
    if not selected:
        return None

    try:
        source_lines, start_line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return None

    patched_lines: list[str] = []
    decorator_re = re.compile(r"^(?P<indent>\s*)@")
    def_re = re.compile(r"^(?P<indent>\s*)def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
    index = 0
    while index < len(source_lines):
        line = source_lines[index]
        line_no = start_line + index
        stripped = line.lstrip()
        is_top_def = stripped.startswith("def ") and len(line) == len(stripped)
        decorator_match = decorator_re.match(line)
        if decorator_match and not is_top_def:
            lookahead = index + 1
            while lookahead < len(source_lines) and decorator_re.match(source_lines[lookahead]):
                lookahead += 1
            if lookahead < len(source_lines):
                def_match = def_re.match(source_lines[lookahead])
                if def_match:
                    key = (start_line + lookahead, def_match.group("name"))
                    label = selected.get(key)
                    if label is not None:
                        patched_lines.append(
                            f"{decorator_match.group('indent')}@__targeted_local_wrapper({label!r})\n"
                        )
            while index < lookahead:
                patched_lines.append(source_lines[index])
                index += 1
            continue
        def_match = def_re.match(line)
        if def_match and not is_top_def:
            key = (line_no, def_match.group("name"))
            label = selected.get(key)
            if label is not None:
                patched_lines.append(
                    f"{def_match.group('indent')}@__targeted_local_wrapper({label!r})\n"
                )
        patched_lines.append(line)
        index += 1

    patched_source = ("\n" * (start_line - 1)) + "".join(patched_lines)
    namespace: dict[str, object] = {}
    exec(compile(patched_source, fn.__code__.co_filename, "exec"), fn.__globals__, namespace)
    patched = namespace.get(fn.__name__)
    if not inspect.isfunction(patched):
        return None
    return patched


def _install_targeted_local_function_wrappers(
    replacements: list[tuple[object, str, object]],
    profiler: TargetedCallProfiler,
    *,
    allowed_labels: set[str] | None = None,
) -> None:
    for owner, attr_name, label in _targeted_function_specs():
        if owner is not estimator:
            continue
        original = getattr(owner, attr_name)
        patched = _patch_function_source_for_local_wrappers(
            original,
            profiler,
            allowed_labels=allowed_labels,
        )
        if patched is None:
            continue
        functools.update_wrapper(patched, original)
        setattr(owner, attr_name, patched)
        replacements.append((owner, attr_name, original))


def _targeted_labels_over_min_call_ms(
    function_table: dict[str, object],
    min_call_ms: float,
) -> set[str]:
    labels = set()
    for row in function_table["children"]:
        calls = int(row["calls"])
        if calls <= 0:
            continue
        per_call_ms = 1000.0 * float(row["inclusive_s"]) / calls
        if per_call_ms >= min_call_ms:
            labels.add(str(row["name"]))
    return labels


def _install_counted_wrapper_probe(
    targeted_profiler: TargetedCallProfiler | None = None,
) -> tuple[list[tuple[object, str, object]], object]:
    """Replace existing flopscope counted wrappers with an equivalent probe."""

    replacements: list[tuple[object, str, object]] = []
    seen: set[tuple[int, str]] = set()
    original_wrapped_co = flops_budget._WRAPPED_CO

    def make_profiled(original):
        fn = original.__wrapped__

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            from flopscope._validation import require_budget

            budget = require_budget()
            fs_t0 = time.perf_counter()
            backend_baseline = budget._total_flopscope_backend_time
            overhead_baseline = budget._total_flopscope_overhead_time
            ops_before = len(budget._op_log)
            try:
                if targeted_profiler is not None:
                    targeted_profiler.enter_timer()
                return fn(*args, **kwargs)
            finally:
                timer_ref = None
                if targeted_profiler is not None:
                    timer_ref = targeted_profiler.exit_timer()
                wall = time.perf_counter() - fs_t0
                if timer_ref is not None:
                    timer_ref.val += wall
                backend_delta = budget._total_flopscope_backend_time - backend_baseline
                overhead_delta = budget._total_flopscope_overhead_time - overhead_baseline
                wrapper_own_overhead = max(wall - backend_delta - overhead_delta, 0.0)
                budget._total_flopscope_overhead_time += wrapper_own_overhead
                ops_added = range(ops_before, len(budget._op_log))
                ops_count = len(budget._op_log) - ops_before
                if ops_count and wrapper_own_overhead > 0:
                    per_op = wrapper_own_overhead / ops_count
                    for idx in ops_added:
                        op = budget._op_log[idx]
                        budget._op_log[idx] = op._replace(
                            flopscope_overhead_duration_s=(
                                op.flopscope_overhead_duration_s or 0.0
                            )
                            + per_op
                        )

        COUNTED_WRAPPER_CODES.add(wrapped.__code__)
        return wrapped

    def make_export_profiled(original):
        inner = make_profiled(original.__wrapped__)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = inner(*args, **kwargs)
            if isinstance(result, flops_ndarray._np.ndarray):
                return flops_ndarray._asflopscope(result)
            if isinstance(result, tuple):
                wrapped_elems = [
                    flops_ndarray._asflopscope(item)
                    if isinstance(item, flops_ndarray._np.ndarray)
                    else item
                    for item in result
                ]
                if type(result) is not tuple and hasattr(type(result), "_fields"):
                    return type(result)(*wrapped_elems)
                return tuple(wrapped_elems)
            if isinstance(result, list):
                return [
                    flops_ndarray._asflopscope(item)
                    if isinstance(item, flops_ndarray._np.ndarray)
                    else item
                    for item in result
                ]
            return result

        return wrapped

    def maybe_replace(owner: object, attr_name: str) -> None:
        key = (id(owner), attr_name)
        if key in seen:
            return
        seen.add(key)
        try:
            value = getattr(owner, attr_name)
        except Exception:
            return
        code = getattr(value, "__code__", None)
        wrapped_value = getattr(value, "__wrapped__", None)
        if not hasattr(value, "__wrapped__"):
            return
        if code is flops_budget._WRAPPED_CO:
            replacement = make_profiled(value)
        elif getattr(wrapped_value, "__code__", None) is flops_budget._WRAPPED_CO:
            replacement = make_export_profiled(value)
        else:
            return
        setattr(owner, attr_name, replacement)
        replacements.append((owner, attr_name, value))

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("flopscope."):
            continue
        for attr_name, value in list(vars(module).items()):
            maybe_replace(module, attr_name)
            if not isinstance(value, type):
                continue
            for class_attr_name in list(vars(value)):
                maybe_replace(value, class_attr_name)

    if replacements:
        flops_budget._WRAPPED_CO = replacements[0][2].__code__
        replacement_code = getattr(getattr(replacements[0][0], replacements[0][1]), "__code__", None)
        if replacement_code is not None:
            flops_budget._WRAPPED_CO = replacement_code

    return replacements, original_wrapped_co


def _restore_counted_wrapper_probe(
    state: tuple[list[tuple[object, str, object]], object],
) -> None:
    replacements, original_wrapped_co = state
    for module, attr_name, original in reversed(replacements):
        setattr(module, attr_name, original)
    flops_budget._WRAPPED_CO = original_wrapped_co


def _restore_targeted_function_wrappers(replacements: list[tuple[object, str, object]]) -> None:
    for owner, attr_name, original in reversed(replacements):
        if original is _MISSING:
            delattr(owner, attr_name)
        else:
            setattr(owner, attr_name, original)


def _targeted_call_tree_data(node: Node) -> dict[str, object]:
    _finalize_call_tree_timed(node)
    data = _call_tree_data(node)
    _assert_additive_call_tree(data)
    return data


def _tree_display_residual_s(tree: dict[str, object]) -> float:
    return sum(float(child["inclusive_s"]) for child in tree["children"])


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
    other = _empty_flops_other_node()
    for child in node["children"]:
        child_flops_pct = 100.0 * int(child["flops"]) / max(total_flops, 1)
        if child_flops_pct < min_flops_pct:
            _add_flops_other_node(other, child)
            continue
        _print_flops_tree(
            child,
            total_flops=total_flops,
            depth=child_depth,
            max_depth=max_depth,
            min_flops_pct=min_flops_pct,
        )
    if other["ops"]:
        _print_flops_other(other, total_flops, child_depth)


def _empty_flops_other_node() -> dict[str, object]:
    return {
        "name": "Other (< min flops)",
        "calls": 0,
        "ops": 0,
        "flops": 0,
        "inclusive_s": 0.0,
    }


def _add_flops_other_node(other: dict[str, object], node: dict[str, object]) -> None:
    other["calls"] = int(other["calls"]) + int(node["calls"])
    other["ops"] = int(other["ops"]) + int(node["ops"])
    other["flops"] = int(other["flops"]) + int(node["flops"])
    other["inclusive_s"] = float(other["inclusive_s"]) + float(node["inclusive_s"])


def _print_flops_other(node: dict[str, object], total_flops: int, depth: int) -> None:
    flops_pct = 100.0 * int(node["flops"]) / max(total_flops, 1)
    elapsed_s = float(node["inclusive_s"])
    print(
        f"{'  ' * depth}{node['name']} "
        f"ops={node['ops']:,} "
        f"flops={node['flops']:,} ({flops_pct:.1f}%) "
        f"flopscope_time={elapsed_s * 1000.0:.2f}ms"
    )


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


def _node_path(node: Node) -> tuple[str, ...]:
    parts = []
    cur: Node | None = node
    while cur is not None and cur.name != "<root>":
        parts.append(cur.name)
        cur = cur.parent
    return tuple(reversed(parts))


def _flops_tree_data(profiler: CallTreeProfiler, ctx) -> dict[str, object]:
    root = _empty_flops_tree_node("<root>")
    budget_id = id(ctx)
    for op_index, op in enumerate(ctx.op_log):
        flop_cost = int(getattr(op, "flop_cost", 0) or 0)
        backend_s = float(getattr(op, "flopscope_backend_duration_s", 0.0) or 0.0)
        overhead_s = float(getattr(op, "flopscope_overhead_duration_s", 0.0) or 0.0)
        owner = profiler.op_owners.get((budget_id, op_index))
        parts = _node_path(owner) if owner is not None else ("<unowned>",)
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


def _flops_tree_data_from_op_namespaces(ctx) -> dict[str, object]:
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


def _empty_tree_data() -> dict[str, object]:
    return {
        "name": "<root>",
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


def _run_targeted_profile(
    est: estimator.Estimator,
    mlp,
    budget: int,
    *,
    allowed_labels: set[str] | None = None,
) -> tuple[dict[str, object], dict[str, object], object, int, float]:
    targeted_profiler = TargetedCallProfiler()
    targeted_function_replacements = _install_targeted_function_wrappers(
        targeted_profiler,
        allowed_labels=allowed_labels,
    )
    targeted_counted_replacements = _install_counted_wrapper_probe(targeted_profiler)
    try:
        with flops.BudgetContext(flop_budget=budget, quiet=True) as targeted_ctx:
            est.predict(mlp, budget)
    finally:
        _restore_counted_wrapper_probe(targeted_counted_replacements)
        _restore_targeted_function_wrappers(targeted_function_replacements)
    targeted_profiler.apply_timer_refs()
    targeted_call_tree = _targeted_call_tree_data(targeted_profiler.root)
    targeted_function_table = _function_table_data(targeted_call_tree)
    targeted_tree_residual_s = _tree_display_residual_s(targeted_call_tree)
    wrapped_count = sum(
        1
        for _, attr_name, original in targeted_function_replacements
        if attr_name != "__targeted_local_wrapper" and original is not _MISSING
    )
    return (
        targeted_call_tree,
        targeted_function_table,
        targeted_ctx,
        wrapped_count,
        targeted_tree_residual_s,
    )


def _write_html_report(
    output_path: Path,
    *,
    metadata: dict[str, object],
    call_tree: dict[str, object],
    function_table: dict[str, object],
    targeted_call_tree: dict[str, object],
    targeted_function_table: dict[str, object],
    flops_tree: dict[str, object],
) -> None:
    payload = json.dumps(
        {
            "metadata": metadata,
            "callTree": call_tree,
            "functionTable": function_table,
            "targetedCallTree": targeted_call_tree,
            "targetedFunctionTable": targeted_function_table,
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
input[type="checkbox"] {{ min-width: auto; }}
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
    <label class="muted"><input id="hideLowOther" type="checkbox"> Hide low Other</label>
    <button id="expand">Expand All</button>
    <button id="collapse">Collapse All</button>
  </div>
  <div class="tabs">
    <button class="tab" data-view="targeted" aria-selected="true">Targeted Calls</button>
    <button class="tab" data-view="targeted-functions" aria-selected="false">Targeted Functions</button>
    <button class="tab" data-view="call" aria-selected="false">Unified Call Tree</button>
    <button class="tab" data-view="functions" aria-selected="false">Profiled Functions</button>
    <button class="tab" data-view="flops" aria-selected="false">Full FLOPs Tree</button>
  </div>
  <div class="tree" id="tree"></div>
</main>
<script>
const data = {payload};
const state = {{
  view: "targeted",
  collapsed: new Set(),
  filter: "",
  minMs: Number(data.metadata.min_ms || 0),
  minFlops: Number(data.metadata.min_flops_pct || 0),
  hideLowOther: false
}};

function fmtMs(seconds) {{ return `${{(seconds * 1000).toFixed(2)}}ms`; }}
function fmtInt(value) {{ return Math.round(value).toLocaleString(); }}
function pct(value, total) {{ return total > 0 ? 100 * value / total : 0; }}
function metric(node) {{ return node.inclusive_s || 0; }}
function flopMetric(node) {{ return node.flops || 0; }}
function isFlopsView() {{ return state.view === "flops"; }}
function isFunctionView() {{ return state.view === "functions" || state.view === "targeted-functions"; }}
function isTargetedView() {{ return state.view === "targeted" || state.view === "targeted-functions"; }}
function rootForView() {{
  if (state.view === "targeted-functions") return data.targetedFunctionTable;
  if (state.view === "functions") return data.functionTable;
  if (state.view === "targeted") return data.targetedCallTree;
  if (state.view === "flops") return data.flopsTree;
  return data.callTree;
}}
function totalForView() {{
  if (isTargetedView()) return data.metadata.targeted_residual_wall_time_s;
  return isFlopsView() ? data.metadata.flops : data.metadata.residual_wall_time_s;
}}
function nodePassesMin(node) {{
  if (node.name === "<root>") return true;
  return isFlopsView()
    ? pct(flopMetric(node), data.metadata.flops) >= state.minFlops
    : metric(node) * 1000 >= state.minMs;
}}
function rowVisible(node) {{
  const minOk = nodePassesMin(node);
  const filterOk = !state.filter || node.name.toLowerCase().includes(state.filter);
  if (minOk && filterOk) return true;
  return node.children.some(rowVisible);
}}
function otherNode(children, name) {{
  const other = {{
    name,
    calls: 0,
    inclusive_s: 0,
    exclusive_s: 0,
    timed_s: 0,
    self_s: 0,
    flops: 0,
    ops: 0,
    children: [],
    synthetic: true
  }};
  for (const child of children) {{
    other.calls += child.calls || 0;
    other.inclusive_s += child.inclusive_s || 0;
    other.exclusive_s += child.exclusive_s || 0;
    other.timed_s += child.timed_s || 0;
    other.self_s += child.self_s || 0;
    other.flops += child.flops || 0;
    other.ops += child.ops || 0;
  }}
  return other.calls ? other : null;
}}
function otherPassesActiveMin(node) {{
  return isFlopsView()
    ? pct(flopMetric(node), data.metadata.flops) >= state.minFlops
    : metric(node) * 1000 >= state.minMs;
}}
function maybeOtherNode(children, name) {{
  const other = otherNode(children, name);
  if (!other) return null;
  if (state.hideLowOther && !otherPassesActiveMin(other)) return null;
  return other;
}}
function renderedChildren(node) {{
  const children = node.children || [];
  const otherName = isFlopsView() ? "Other (< min flops)" : "Other (< min ms)";
  if (isFlopsView()) {{
    const shown = children.filter(child => nodePassesMin(child) && rowVisible(child));
    const hidden = children.filter(child => !nodePassesMin(child));
    const other = (!state.filter || otherName.toLowerCase().includes(state.filter)) ? maybeOtherNode(hidden, otherName) : null;
    return other ? [...shown, other] : shown;
  }}
  if (isFunctionView()) {{
    return children.filter(child => nodePassesMin(child) && rowVisible(child));
  }}
  const shown = children.filter(child => nodePassesMin(child) && rowVisible(child));
  const hidden = children.filter(child => !nodePassesMin(child));
  const other = (!state.filter || otherName.toLowerCase().includes(state.filter)) ? maybeOtherNode(hidden, otherName) : null;
  return other ? [...shown, other] : shown;
}}
function renderMeta() {{
  const meta = data.metadata;
  const items = [
    `width=${{meta.width}} depth=${{meta.depth}} seed=${{meta.seed}} mode=${{meta.mode}}`,
    `FLOPs=${{fmtInt(meta.flops)}}`,
    `ops=${{fmtInt(meta.ops)}}`,
    `backend=${{fmtMs(meta.backend_time_s)}}`,
    `overhead=${{fmtMs(meta.overhead_time_s)}}`,
    `residual=${{fmtMs(meta.residual_wall_time_s)}}`
  ];
  if (isTargetedView()) {{
    items.push(`targeted ctx residual=${{fmtMs(meta.targeted_residual_wall_time_s)}}`);
    items.push(`targeted tree residual=${{fmtMs(meta.targeted_tree_residual_s)}}`);
    items.push(`targeted wrapped=${{fmtInt(meta.targeted_wrapped_functions)}}`);
  }}
  document.getElementById("meta").innerHTML = items.map(text => `<span class="pill"><code>${{text}}</code></span>`).join("");
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
    for (const child of renderedChildren(node)) scan(child);
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
      if (!node.synthetic && !rowVisible(node)) return;
      const id = path.join("/");
      const children = node.synthetic ? [] : renderedChildren(node);
      const hasChildren = !isFunctionView() && children.length > 0;
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
    const children = node.synthetic ? [] : renderedChildren(node);
    for (const child of children) walk(child, isFunctionView() || node.name === "<root>" ? depth : depth + 1, [...path, child.name]);
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
document.getElementById("hideLowOther").addEventListener("change", event => {{
  state.hideLowOther = event.target.checked;
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
    renderMeta();
    render();
  }});
}}
renderMeta();
document.getElementById("minMs").value = state.minMs;
document.getElementById("minFlops").value = state.minFlops;
document.getElementById("hideLowOther").checked = state.hideLowOther;
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
    parser.add_argument(
        "--targeted-min-call-ms",
        type=float,
        default=0.0,
        help="Only wrap targeted functions whose profiled inclusive time per call is at least this many ms.",
    )
    parser.add_argument(
        "--unified-call-tree",
        action="store_true",
        help="Also run the high-overhead sys.setprofile unified call tree.",
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

    targeted_allowed_labels = None
    if args.targeted_min_call_ms > 0:
        bootstrap_call_tree, bootstrap_function_table, _, _, _ = _run_targeted_profile(
            est,
            mlp,
            args.budget,
            allowed_labels=None,
        )
        targeted_allowed_labels = _targeted_labels_over_min_call_ms(
            bootstrap_function_table,
            args.targeted_min_call_ms,
        )
    (
        targeted_call_tree,
        targeted_function_table,
        targeted_ctx,
        targeted_wrapped_functions,
        targeted_tree_residual_s,
    ) = _run_targeted_profile(
        est,
        mlp,
        args.budget,
        allowed_labels=targeted_allowed_labels,
    )

    call_tree = _empty_tree_data()
    function_table = _function_table_data(call_tree)
    ctx = targeted_ctx
    flops_tree = _flops_tree_data_from_op_namespaces(targeted_ctx)
    if args.unified_call_tree:
        call_root = Node("<root>")
        profiler = CallTreeProfiler(
            call_root,
            hide_flopscope=args.hide_flopscope,
            hide_numpy=args.hide_numpy,
        )
        try:
            with flops.BudgetContext(flop_budget=args.budget, quiet=True) as unified_ctx:
                sys.setprofile(profiler)
                est.predict(mlp, args.budget)
        finally:
            sys.setprofile(None)

        _annotate_call_tree_op_times(profiler, unified_ctx)
        flops_tree = _flops_tree_data(profiler, unified_ctx)
        _finalize_call_tree_timed(call_root)
        call_tree = _call_tree_data(call_root)
        _assert_additive_call_tree(call_tree)
        function_table = _function_table_data(call_tree)
        ctx = unified_ctx

    print(f"MLP width={args.width} depth={args.depth} seed={args.seed} mode={args.mode or 'default'}")
    print(f"flops={targeted_ctx.flops_used:,} ops={len(targeted_ctx.op_log):,}")
    print(f"backend_time_s={targeted_ctx.flopscope_backend_time:.6f}")
    print(f"overhead_time_s={targeted_ctx.flopscope_overhead_time:.6f}")
    print(f"residual_wall_time_s={targeted_ctx.residual_wall_time:.6f}\n")
    print("Targeted calls: low-overhead wrappers around selected high per-call functions")
    _print_call_tree(
        targeted_call_tree,
        targeted_ctx.residual_wall_time,
        max_depth=args.max_depth,
        min_ms=args.min_ms,
        total_flops=targeted_ctx.flops_used,
    )
    print()
    print("Targeted functions: flat aggregation of targeted call rows by function name")
    _print_function_table(
        targeted_call_tree,
        targeted_ctx.residual_wall_time,
        min_ms=args.min_ms,
        total_flops=targeted_ctx.flops_used,
    )
    print()
    if args.unified_call_tree:
        print("Call tree: estimator and non-estimator Python/C calls in the same tree")
        print("Note: residual time subtracts call-tree work measured inside flopscope counted wrappers.")
        _print_call_tree(
            call_tree,
            ctx.residual_wall_time,
            max_depth=args.max_depth,
            min_ms=args.min_ms,
            total_flops=ctx.flops_used,
        )
        print()
        print("Profiled functions: flat aggregation of unified call tree rows by function name")
        _print_function_table(
            call_tree,
            ctx.residual_wall_time,
            min_ms=args.min_ms,
            total_flops=ctx.flops_used,
        )
        print()
    print(
        "FLOPs tree: exact flopscope op log by counted-wrapper call path and op"
        if args.unified_call_tree
        else "FLOPs tree: exact flopscope op log by namespace and op"
    )
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
            "flops": targeted_ctx.flops_used,
            "ops": len(targeted_ctx.op_log),
            "backend_time_s": targeted_ctx.flopscope_backend_time,
            "overhead_time_s": targeted_ctx.flopscope_overhead_time,
            "residual_wall_time_s": targeted_ctx.residual_wall_time,
            "targeted_residual_wall_time_s": targeted_ctx.residual_wall_time,
            "targeted_tree_residual_s": targeted_tree_residual_s,
            "targeted_min_call_ms": args.targeted_min_call_ms,
            "targeted_wrapped_functions": targeted_wrapped_functions,
            "unified_call_tree": args.unified_call_tree,
            "min_ms": args.min_ms,
            "min_flops_pct": args.min_flops,
        },
        call_tree=call_tree,
        function_table=function_table,
        targeted_call_tree=targeted_call_tree,
        targeted_function_table=targeted_function_table,
        flops_tree=flops_tree,
    )
    print(f"\nWrote HTML report: {args.output.resolve()}")
    if not args.no_browser:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
