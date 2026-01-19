"""
Shared validation checks for detecting non-concrete solution structures.
"""

import numbers
from typing import Any

import numpy as np


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
            obj_id = id(value)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for idx, elem in enumerate(value):
                stack.append((f"{path}[{idx}]", elem))
            continue
        if isinstance(value, dict):
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
