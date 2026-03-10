from __future__ import annotations

"""
Shared validation checks for detecting non-concrete solution structures.
"""

import ast
import inspect
import logging
import numbers
import textwrap
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

try:
    from AlgoTuner.security.code_validator import TamperingDetector
except Exception:
    _PROTECTED_MODULE_ROOTS: frozenset[str] = frozenset()
else:
    _PROTECTED_MODULE_ROOTS = frozenset(
        module_name.split(".", 1)[0]
        for module_name in TamperingDetector.PROTECTED_MODULES
        if module_name
    )


_MISSING = object()


@dataclass(frozen=True)
class ValidationGlobalReference:
    name: str
    expected_object: Any


@dataclass(frozen=True)
class ValidationAttributeReference:
    display_name: str
    root_name: str
    attr_path: tuple[str, ...]
    expected_object: Any


@dataclass(frozen=True)
class ValidationDependencySnapshot:
    global_refs: tuple[ValidationGlobalReference, ...] = ()
    attribute_refs: tuple[ValidationAttributeReference, ...] = ()


def _as_function(method: Any) -> Any:
    return getattr(method, "__func__", method)


def _is_snapshot_candidate(value: Any) -> bool:
    return isinstance(value, ModuleType) or inspect.isfunction(value) or isinstance(value, type)


def _is_protected_module(value: Any) -> bool:
    module_name = getattr(value, "__name__", "")
    if not module_name:
        return False
    return module_name.split(".", 1)[0] in _PROTECTED_MODULE_ROOTS


def _extract_attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return None


def _extract_module_attribute_chains(func: Any) -> set[tuple[str, ...]]:
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    chains: set[tuple[str, ...]] = set()

    class _AttributeCollector(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            chain = _extract_attribute_chain(node)
            if chain and len(chain) > 1:
                chains.add(chain)
            self.generic_visit(node)

    _AttributeCollector().visit(tree)
    return chains


def _resolve_attr_chain(root_obj: Any, attr_path: tuple[str, ...]) -> Any:
    current = root_obj
    for attr in attr_path:
        if not hasattr(current, attr):
            return _MISSING
        current = getattr(current, attr)
    return current


def capture_validation_dependency_snapshot(
    task_instance: Any,
    expected_is_solution_method: Any | None = None,
) -> ValidationDependencySnapshot | None:
    method = expected_is_solution_method or getattr(task_instance, "is_solution", None)
    if not callable(method):
        return None

    func = _as_function(method)
    globals_map = getattr(func, "__globals__", None)
    if not isinstance(globals_map, dict):
        return None

    try:
        closure_vars = inspect.getclosurevars(func)
        referenced_globals = dict(closure_vars.globals)
    except TypeError:
        referenced_globals = {
            name: globals_map[name]
            for name in getattr(getattr(func, "__code__", None), "co_names", ())
            if name in globals_map
        }

    global_refs = tuple(
        ValidationGlobalReference(name=name, expected_object=value)
        for name, value in sorted(referenced_globals.items())
        if _is_snapshot_candidate(value)
    )

    module_roots = {
        name: value
        for name, value in referenced_globals.items()
        if isinstance(value, ModuleType) and _is_protected_module(value)
    }

    attribute_refs: list[ValidationAttributeReference] = []
    for chain in sorted(_extract_module_attribute_chains(func)):
        root_name = chain[0]
        root_obj = module_roots.get(root_name)
        if root_obj is None:
            continue

        resolved = _resolve_attr_chain(root_obj, chain[1:])
        if resolved is _MISSING:
            continue

        attribute_refs.append(
            ValidationAttributeReference(
                display_name=".".join(chain),
                root_name=root_name,
                attr_path=chain[1:],
                expected_object=resolved,
            )
        )

    return ValidationDependencySnapshot(
        global_refs=global_refs,
        attribute_refs=tuple(attribute_refs),
    )


def detect_validation_dependency_tampering(
    task_instance: Any,
    expected_snapshot: ValidationDependencySnapshot | None,
) -> str | None:
    if expected_snapshot is None:
        return None

    method = getattr(task_instance, "is_solution", None)
    if not callable(method):
        return "SECURITY_VIOLATION: task is_solution method is missing or non-callable"

    func = _as_function(method)
    globals_map = getattr(func, "__globals__", None)
    if not isinstance(globals_map, dict):
        return None

    for global_ref in expected_snapshot.global_refs:
        current = globals_map.get(global_ref.name, _MISSING)
        if current is _MISSING:
            return (
                f"SECURITY_VIOLATION: validator global '{global_ref.name}' "
                "is missing at runtime"
            )
        if current is not global_ref.expected_object:
            return (
                f"SECURITY_VIOLATION: validator global '{global_ref.name}' "
                "was modified at runtime"
            )

    for attr_ref in expected_snapshot.attribute_refs:
        root_obj = globals_map.get(attr_ref.root_name, _MISSING)
        if root_obj is _MISSING:
            return (
                f"SECURITY_VIOLATION: validator dependency '{attr_ref.display_name}' "
                "is missing at runtime"
            )

        current = _resolve_attr_chain(root_obj, attr_ref.attr_path)
        if current is _MISSING:
            return (
                f"SECURITY_VIOLATION: validator dependency '{attr_ref.display_name}' "
                "is missing at runtime"
            )
        if current is not attr_ref.expected_object:
            return (
                f"SECURITY_VIOLATION: validator dependency '{attr_ref.display_name}' "
                "was modified at runtime"
            )

    return None


def _has_container_proxy_overrides(
    value: Any,
    base_type: type,
    method_names: tuple[str, ...],
) -> bool:
    """Detect builtin container subclasses that override access/materialization hooks."""
    cls = type(value)
    if cls is base_type:
        return False

    for method_name in method_names:
        base_attr = getattr(base_type, method_name, None)
        cls_attr = getattr(cls, method_name, None)
        if cls_attr is not base_attr:
            return True

    return False


def prepare_isolated_solver_result_for_validation(
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    """Use the timed isolated result for validation and refuse unsafe parent replays."""
    if not benchmark_result.get("success"):
        return benchmark_result

    if benchmark_result.get("validation_result") is not None:
        return benchmark_result

    if benchmark_result.get("result") is not None:
        logging.info(
            "[ISOLATED_VALIDATION] Using materialized timed solver result from isolated benchmark"
        )
        return benchmark_result

    error_message = (
        "Timed solver result unavailable for validation; refusing unsafe untimed solver replay."
    )
    logging.error(f"[ISOLATED_VALIDATION] {error_message}")

    stripped_result = {
        "stripped_after_validation": True,
        "validation_completed": True,
        "validation_failed": True,
        "reason": "timed_result_unavailable",
    }
    benchmark_result["result"] = stripped_result
    benchmark_result["result_summary"] = stripped_result
    benchmark_result["validation_result"] = {
        "success": False,
        "error_type": "invalid_solution",
        "error": error_message,
    }
    return benchmark_result


def find_nonconcrete_solution(solution: Any) -> str | None:
    stack = [("solution", solution)]
    seen: set[int] = set()
    nodes_seen = 0
    max_nodes = 1_000_000

    while stack:
        path, value = stack.pop()
        nodes_seen += 1
        if nodes_seen > max_nodes:
            return "Solution too large to validate safely"

        if value is None:
            continue
        if isinstance(value, (str, bytes, bool, numbers.Number, np.generic)):
            continue
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                obj_id = id(value)
                if obj_id in seen:
                    continue
                seen.add(obj_id)
                for idx, elem in enumerate(value.flat):
                    stack.append((f"{path}[{idx}]", elem))
            continue
        if isinstance(value, (list, tuple)):
            if isinstance(value, list) and _has_container_proxy_overrides(
                value,
                list,
                ("__iter__", "__getitem__", "__len__"),
            ):
                return f"Solution contains list-like proxy at {path}: {type(value).__name__}"
            if isinstance(value, tuple) and _has_container_proxy_overrides(
                value,
                tuple,
                ("__iter__", "__getitem__", "__len__"),
            ):
                return f"Solution contains tuple-like proxy at {path}: {type(value).__name__}"
            obj_id = id(value)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for idx, elem in enumerate(value):
                stack.append((f"{path}[{idx}]", elem))
            continue
        if isinstance(value, dict):
            if _has_container_proxy_overrides(
                value,
                dict,
                ("__iter__", "__getitem__", "__len__", "items", "keys", "values", "get"),
            ):
                return f"Solution contains dict-like proxy at {path}: {type(value).__name__}"
            obj_id = id(value)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for key, elem in value.items():
                if not isinstance(key, (str, bytes, bool, numbers.Number, np.generic)):
                    return (
                        f"Solution has non-primitive dict key at {path}: {type(key).__name__}"
                    )
                stack.append((f"{path}[{key!r}]", elem))
            continue
        if hasattr(value, "__array__"):
            return f"Solution contains array-like proxy at {path}: {type(value).__name__}"
        if hasattr(value, "__iter__"):
            return f"Solution contains non-list iterable at {path}: {type(value).__name__}"
        return f"Solution contains unsupported type at {path}: {type(value).__name__}"

    return None
