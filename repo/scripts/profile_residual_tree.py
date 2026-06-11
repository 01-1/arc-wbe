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
from whestbench import SetupContext

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import estimator
from local_engine import build_mlp


@dataclass
class Node:
    name: str
    wall_s: float = 0.0
    self_s: float = 0.0
    flops: int = 0
    ops: int = 0
    non_residual: bool = False
    calls: int = 0
    children: dict[str, "Node"] = field(default_factory=dict)


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
        node = parent.children.setdefault(__namespace, Node(__namespace))
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
        self.frames: dict[object, tuple[Node, float, float, bool]] = {}

    def __call__(self, frame, event, arg):
        if event == "call":
            parent_frame = frame.f_back
            if parent_frame in self.frames:
                parent, _, _, _ = self.frames[parent_frame]
            else:
                parent = self.root

            name = self._frame_name(frame)
            skip = self._skip_frame(frame)
            if skip:
                node = parent
            else:
                node = parent.children.setdefault(name, Node(name))
                node.non_residual = parent.non_residual or self._non_residual_frame(frame)
                node.calls += 1

            now = time.perf_counter()
            self.frames[frame] = (node, now, 0.0, skip)
            return self

        if event == "return":
            entry = self.frames.pop(frame, None)
            if entry is None:
                return self
            node, start, child_s, skip = entry
            elapsed = time.perf_counter() - start
            if not skip:
                node.wall_s += elapsed
                node.self_s += max(0.0, elapsed - child_s)
            parent_frame = frame.f_back
            if parent_frame in self.frames:
                parent_node, parent_start, parent_child_s, parent_skip = self.frames[parent_frame]
                self.frames[parent_frame] = (
                    parent_node,
                    parent_start,
                    parent_child_s + elapsed,
                    parent_skip,
                )
            return self

        if event == "c_call":
            parent_frame = frame
            if parent_frame not in self.frames:
                return self
            parent, _, _, _ = self.frames[parent_frame]
            name = self._c_name(arg)
            skip = self._skip_c_call(arg)
            if skip:
                node = parent
            else:
                node = parent.children.setdefault(name, Node(name))
                node.non_residual = parent.non_residual or self._non_residual_c_call(arg)
                node.calls += 1
            now = time.perf_counter()
            self.frames[(frame, arg)] = (node, now, 0.0, skip)
            return self

        if event == "c_return" or event == "c_exception":
            entry = self.frames.pop((frame, arg), None)
            if entry is None:
                return self
            node, start, child_s, skip = entry
            elapsed = time.perf_counter() - start
            if not skip:
                node.wall_s += elapsed
                node.self_s += max(0.0, elapsed - child_s)
            if frame in self.frames:
                parent_node, parent_start, parent_child_s, parent_skip = self.frames[frame]
                self.frames[frame] = (
                    parent_node,
                    parent_start,
                    parent_child_s + elapsed,
                    parent_skip,
                )
            return self

        return self

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
            and child_flops_pct < min_flops_pct
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


def _print_call_tree(
    node: Node,
    total_wall_s: float,
    residual_scale: float = 1.0,
    depth: int = 0,
    max_depth: int = 8,
    min_ms: float = 0.15,
    min_flops_pct: float = 0.0,
    total_flops: int = 0,
) -> float:
    children = list(node.children.values())
    child_residual_s = sum(_call_node_residual(child, residual_scale) for child in children)
    own_residual_s = 0.0 if node.non_residual else max(0.0, node.self_s) * residual_scale
    residual_s = own_residual_s + child_residual_s
    if node.name != "<root>":
        pct = 100.0 * residual_s / max(total_wall_s, 1e-12)
        self_pct = 100.0 * own_residual_s / max(total_wall_s, 1e-12)
        print(
            f"{'  ' * depth}{node.name} "
            f"calls={node.calls} "
            f"resid={residual_s * 1000.0:.2f}ms ({pct:.1f}%) "
            f"self={own_residual_s * 1000.0:.2f}ms ({self_pct:.1f}%) "
            f"flops={_call_node_flops(node):,}"
        )

    if depth >= max_depth:
        return residual_s

    child_depth = depth if node.name == "<root>" else depth + 1
    for child in sorted(children, key=lambda item: _call_node_residual(item, residual_scale), reverse=True):
        child_flops_pct = 100.0 * _call_node_flops(child) / max(total_flops, 1)
        if (
            node.name != "<root>"
            and _call_node_residual(child, residual_scale) * 1000.0 < min_ms
            and child_flops_pct < min_flops_pct
        ):
            continue
        _print_call_tree(
            child,
            total_wall_s,
            residual_scale,
            child_depth,
            max_depth,
            min_ms,
            min_flops_pct,
            total_flops,
        )
    return residual_s


def _call_node_residual(node: Node, residual_scale: float = 1.0) -> float:
    if node.non_residual:
        return 0.0
    return (0.0 if node.non_residual else max(0.0, node.self_s) * residual_scale) + sum(
        _call_node_residual(child, residual_scale) for child in node.children.values()
    )


def _call_node_flops(node: Node) -> int:
    return node.flops + sum(_call_node_flops(child) for child in node.children.values())


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


def _call_tree_data(node: Node, residual_scale: float = 1.0):
    children = [_call_tree_data(child, residual_scale) for child in node.children.values()]
    child_residual_s = 0.0 if node.non_residual else sum(child["inclusive_s"] for child in children)
    own_residual_s = 0.0 if node.non_residual else max(0.0, node.self_s) * residual_scale
    inclusive_s = 0.0 if node.non_residual else own_residual_s + child_residual_s
    return {
        "name": node.name,
        "calls": node.calls,
        "wall_s": node.wall_s,
        "self_s": node.self_s,
        "inclusive_s": inclusive_s,
        "exclusive_s": own_residual_s,
        "ops": 0,
        "flops": 0,
        "non_residual": node.non_residual,
        "children": sorted(children, key=lambda child: child["inclusive_s"], reverse=True),
    }


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
    flops_tree: dict[str, object],
) -> None:
    payload = json.dumps(
        {
            "metadata": metadata,
            "namespaceTree": namespace_tree,
            "callTree": call_tree,
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
  height: 8px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 20%, transparent);
  overflow: hidden;
}}
.fill {{
  display: block;
  height: 100%;
  background: var(--accent);
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
    <label class="muted">Min ms <input id="minMs" type="number" min="0" step="0.1" value="0.5"></label>
    <label class="muted">Min FLOPs % <input id="minFlops" type="number" min="0" step="0.1" value="0"></label>
    <button id="expand">Expand All</button>
    <button id="collapse">Collapse All</button>
  </div>
  <div class="tabs">
    <button class="tab" data-view="namespace" aria-selected="true">Residual Namespace Tree</button>
    <button class="tab" data-view="call" aria-selected="false">Unified Call Tree</button>
    <button class="tab" data-view="flops" aria-selected="false">Full FLOPs Tree</button>
  </div>
  <div class="tree" id="tree"></div>
</main>
<script>
const data = {payload};
const state = {{ view: "namespace", collapsed: new Set(), filter: "", minMs: 0.5, minFlops: 0 }};

function fmtMs(seconds) {{ return `${{(seconds * 1000).toFixed(2)}}ms`; }}
function fmtInt(value) {{ return Math.round(value).toLocaleString(); }}
function pct(value, total) {{ return total > 0 ? 100 * value / total : 0; }}
function metric(node) {{ return node.inclusive_s || 0; }}
function flopMetric(node) {{ return node.flops || 0; }}
function isFlopsView() {{ return state.view === "flops"; }}
function rootForView() {{
  if (state.view === "namespace") return data.namespaceTree;
  if (state.view === "flops") return data.flopsTree;
  return data.callTree;
}}
function totalForView() {{
  return isFlopsView() ? data.metadata.flops : data.metadata.residual_wall_time_s;
}}
function rowVisible(node) {{
  const flopPct = pct(flopMetric(node), data.metadata.flops);
  const timeOk = !isFlopsView() && metric(node) * 1000 >= state.minMs;
  const flopsOk = flopPct >= state.minFlops;
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
  const secondMetricTitle = state.view === "namespace" ? "Exclusive" : "Self";
  const rows = [`<div class="row header">
    <div>Function</div><div>${{countTitle}}</div><div>${{firstMetricTitle}}</div><div>${{secondMetricTitle}}</div><div>FLOPs</div><div>Ops</div>
  </div>`];
  function walk(node, depth, path) {{
    if (node.name !== "<root>") {{
      if (!rowVisible(node)) return;
      const id = path.join("/");
      const hasChildren = node.children.some(rowVisible);
      const collapsed = state.collapsed.has(id);
      const value = metric(node);
      const selfValue = state.view === "namespace" ? node.exclusive_s : node.self_s;
      const timeLabel = isFlopsView() && value === 0 ? "" : fmtMs(value);
      const selfLabel = isFlopsView() && selfValue === 0 ? "" : fmtMs(selfValue);
      rows.push(`<div class="row" data-path="${{htmlEscape(id)}}">
        <div class="name" style="padding-left:${{depth * 18}}px">
          ${{hasChildren ? `<button class="twisty" data-toggle="${{htmlEscape(id)}}">${{collapsed ? "▸" : "▾"}}</button>` : `<span class="twisty"></span>`}}
          <span class="name-text" title="${{htmlEscape(node.name)}}">${{htmlEscape(node.name)}}</span>
        </div>
        <div>${{fmtInt(node.calls)}}</div>
        <div class="barcell"><span>${{timeLabel}}</span><span class="bar"><span class="fill" style="width:${{Math.min(100, pct(value, maxVisible))}}%"></span></span></div>
        <div>${{selfLabel}}</div>
        <div class="barcell"><span>${{node.flops ? fmtInt(node.flops) : ""}}</span><span class="bar"><span class="fill flops" style="width:${{Math.min(100, pct(flopMetric(node), maxVisibleFlops))}}%"></span></span></div>
        <div>${{node.ops ? fmtInt(node.ops) : ""}}</div>
      </div>`);
      if (collapsed) return;
    }}
    for (const child of node.children) walk(child, node.name === "<root>" ? depth : depth + 1, [...path, child.name]);
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
    parser.add_argument("--min-ms", type=float, default=0.15)
    parser.add_argument(
        "--min-flops",
        type=float,
        default=0.0,
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
    print("Note: this is residual time; flopscope and NumPy subtrees are visible but have zero residual.")
    _print_call_tree(
        call_root,
        ctx.residual_wall_time,
        max_depth=args.max_depth,
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
        },
        namespace_tree=_namespace_tree_data(root, (), cost_by_path, ops_by_path, flops_by_path),
        call_tree=_call_tree_data(call_root),
        flops_tree=flops_tree,
    )
    print(f"\nWrote HTML report: {args.output.resolve()}")
    if not args.no_browser:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
