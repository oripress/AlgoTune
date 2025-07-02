"""
Robust casting utilities for AlgoTune.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import types
import typing
from dataclasses import is_dataclass, fields
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import scipy.sparse as sparse
from numpy.typing import NDArray


def _truncate_repr(val: Any, max_list_items: int = 5, max_str_len: int = 250) -> str:
    """Provides a truncated string representation for objects, especially lists/tuples."""
    try:
        r = repr(val)
    except Exception:
        # Fallback if repr() itself raises an error on the object
        try:
            r = object.__repr__(val)
        except Exception:
            r = "<unrepresentable object>"

    if len(r) <= max_str_len:
        return r

    if isinstance(val, (list, tuple)):
        num_items = len(val)
        if num_items == 0: # Empty list/tuple but long repr (e.g. a custom class)
             return r[:max_str_len -3] + "..."

        # Try to show head and tail for lists/tuples
        # Only truncate if the list is substantially larger than what we'd show
        if num_items > max_list_items * 2 + 1: 
            head_reprs = []
            current_len = 0
            # Max length for combined head and tail strings, plus ellipsis and count
            # Reserve space for "[..., ...] (total N items)"
            reserved_space = len("[, ..., ] ()") + len(str(num_items)) + len("total  items") + 20 # Generous buffer
            
            for i in range(max_list_items):
                item_r = repr(val[i])
                if current_len + len(item_r) + len(", ") > (max_str_len - reserved_space) / 2:
                    break
                head_reprs.append(item_r)
                current_len += len(item_r) + (len(", ") if i < max_list_items -1 else 0)

            tail_reprs = []
            current_len = 0
            for i in range(max_list_items):
                item_r = repr(val[num_items - max_list_items + i])
                if current_len + len(item_r) + len(", ") > (max_str_len - reserved_space) / 2:
                    break
                tail_reprs.append(item_r)
                current_len += len(item_r) + (len(", ") if i < max_list_items -1 else 0)
            
            if head_reprs and tail_reprs:
                return f"[{', '.join(head_reprs)}, ..., {', '.join(tail_reprs)}] (total {num_items} items)"
            elif head_reprs: # Only head fits within reasonable limits
                 return f"[{', '.join(head_reprs)}, ...] (total {num_items} items)"
            # Fallback if smart truncation still too complex or doesn't save much
            return r[:max_str_len - 3] + "..."
        else: # List is not long enough for "head...tail" display but its repr is too long
             return r[:max_str_len - 3] + "..."

    # For non-list/tuple types if their repr is too long, or list truncation fallback
    return r[:max_str_len - 3] + "..."



def parse_string(text: str) -> Any:
    txt = text.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(txt)
    except (SyntaxError, ValueError):
        return text


def _is_union(tp: Any) -> bool:
    origin = get_origin(tp)
    return (
        origin is Union
        or origin is types.UnionType
        or isinstance(tp, types.UnionType)
    )


def _safe_numpy(values: Any, dtype: Any | None = None) -> np.ndarray:
    """
    Wrap list/tuple/array‑like into np.ndarray.
    NOTE: We intentionally do *not* auto‑convert dicts any more
    because that broke legitimate dict parameters.
    """
    if isinstance(values, np.ndarray):
        return values.astype(dtype) if dtype and values.dtype != dtype else values

    # Dicts are no longer accepted silently.
    if isinstance(values, dict):
        raise TypeError("Dict→ndarray conversion disabled (keys lost).")

    try:
        return np.array(values, dtype=dtype)
    except Exception as e:
        raise TypeError(f"Cannot convert {_truncate_repr(values)} to ndarray ({dtype})") from e


def _dtype_from_args(args: Tuple[Any, ...]) -> Any | None:
    for arg in args:
        try:
            return np.dtype(arg)
        except TypeError:
            continue
    return None


def _shape_list_to_tuple(val: Any) -> Tuple[int, int] | Any:
    if (
        isinstance(val, list)
        and len(val) == 2
        and all(isinstance(i, int) for i in val)
    ):
        return tuple(val)
    return val


###############################################################################
# Primitive casting
###############################################################################

def _cast_primitive(val: Any, target: Type) -> Any:
    if target is typing.Any:
        return val

    if val is None:
        if target in (int, float, bool, complex, str):
            raise TypeError(f"Cannot convert None to {target.__name__}")
        return None

    if isinstance(val, (np.number, np.bool_)):
        val = val.item()

    if isinstance(val, target):
        return val

    if target is int:
        try:
            return int(float(val))
        except Exception as e:
            raise TypeError(f"Cannot cast {_truncate_repr(val)} to int") from e

    if target is float:
        try:
            return float(val)
        except Exception as e:
            raise TypeError(f"Cannot cast {_truncate_repr(val)} to float") from e

    if target is bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("true", "false"):
                return s == "true"
        raise TypeError(
            f"Cannot cast {_truncate_repr(val)} to bool (expect bool or 'true'/'false' str)"
        )

    if target is complex:
        try:
            return complex(val)
        except Exception as e:
            raise TypeError(f"Cannot cast {_truncate_repr(val)} to complex") from e

    if target is str:
        return str(val)

    if inspect.isclass(target) and issubclass(target, Enum):
        try:
            return target(val)
        except Exception:
            for member in target:
                if str(val).lower() == member.name.lower():
                    return member
        raise TypeError(f"{_truncate_repr(val)} not valid for enum {target}")

    try:
        return target(val)
    except Exception as e:
        raise TypeError(f"Cannot cast {_truncate_repr(val)} to {target}") from e


###############################################################################
# Core recursive dispatcher
###############################################################################

def cast_nested_value(value: Any, hint: Any) -> Any:  # noqa: C901
    logging.debug(
        "cast_nested_value: val=%s (%s)  hint=%s",
        _truncate_repr(str(value), max_str_len=80),
        type(value),
        hint,
    )

    # 1. Union / Optional
    if _is_union(hint):
        last = None
        for sub in get_args(hint):
            try:
                return cast_nested_value(value, sub)
            except (TypeError, ValueError) as e:
                last = e
        raise TypeError(
            f"{_truncate_repr(value)} cannot be cast to any option in {hint}"
        ) from last

    # 2. Any
    if hint is typing.Any:
        return value

    origin = get_origin(hint)
    args = get_args(hint)

    # 3. NumPy arrays
    if hint is np.ndarray or origin in (np.ndarray, NDArray):
        dtype = _dtype_from_args(args)
        if not isinstance(value, (list, tuple, np.ndarray)):
            try:
                return _safe_numpy([value], dtype=dtype).squeeze()
            except TypeError:
                pass
        return _safe_numpy(value, dtype=dtype)

    # 4. SciPy sparse
    if inspect.isclass(hint) and issubclass(hint, sparse.spmatrix):
        if isinstance(value, hint):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"Expect dict for {hint}, got {type(value)}")

        shape = _shape_list_to_tuple(value.get("shape"))
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise TypeError("Sparse dict missing valid 'shape'")

        data = value.get("data")
        if data is None:
            raise TypeError("Sparse dict missing 'data'")

        try:
            if issubclass(hint, (sparse.csr_matrix, sparse.csc_matrix)):
                return hint((data, value["indices"], value["indptr"]), shape=shape)
            if issubclass(hint, sparse.coo_matrix):
                return hint((data, (value["row"], value["col"])), shape=shape)
            raise TypeError(f"Unsupported sparse type {hint}")
        except KeyError as e:
            raise TypeError(f"Sparse dict missing key {e}") from e
        except Exception as e:
            raise TypeError(f"Failed constructing {hint}: {e}") from e

    # 5. Primitive & dataclass / enum
    if origin is None:
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs = {}
            for f in fields(hint):
                if f.name in value:
                    kwargs[f.name] = cast_nested_value(value[f.name], f.type)
                elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    raise TypeError(f"Missing field '{f.name}' for {hint}")
            return hint(**kwargs)

        return _cast_primitive(value, hint)

    # 6. Sequence / List
    if origin in (list, List, Sequence):
        elem_type = args[0] if args else Any
        if not isinstance(value, (list, tuple, np.ndarray)):
            # If target is a sequence but value is not, raise error immediately
            raise TypeError(f"Cannot cast non-sequence type {type(value).__name__} to sequence type {hint}")
        # Value is a sequence, proceed with casting elements
        return [cast_nested_value(v, elem_type) for v in value]

    # 7. Set / FrozenSet
    if origin in (set, Set, frozenset, typing.FrozenSet):
        elem_type = args[0] if args else Any
        iterable = (
            value
            if isinstance(value, (list, tuple, set, frozenset, np.ndarray))
            else list(value)
        )
        casted = {cast_nested_value(v, elem_type) for v in iterable}
        return casted if origin in (set, Set) else frozenset(casted)

    # 8. Tuple
    if origin in (tuple, Tuple):
        if not isinstance(value, tuple):
            value = tuple(value)
        if not args:
            return value
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(cast_nested_value(v, args[0]) for v in value)
        if len(args) != len(value):
            raise TypeError(
                f"Tuple length mismatch for {hint}: expected {len(args)}, got {len(value)}"
            )
        return tuple(cast_nested_value(v, t) for v, t in zip(value, args))

    # 9. Dict / Mapping
    if origin in (dict, Dict, typing.Mapping):
        if not isinstance(value, dict):
            value = dict(value)
        key_t, val_t = args if len(args) == 2 else (Any, Any)
        out = {}
        for k, v in value.items():
            ck = cast_nested_value(k, key_t)
            vv = v
            if ck == "shape":
                vv = _shape_list_to_tuple(vv)

            # ---- special‑case: "params" stored as [A, B] -------------
            if (
                ck == "params"
                and isinstance(vv, (list, tuple))
                and len(vv) == 2
                and all(isinstance(x, (int, float, np.number)) for x in vv)
            ):
                vv = {"A": vv[0], "B": vv[1]}
            # ----------------------------------------------------------

    # 10. Fallback
    raise TypeError(f"Unsupported type hint: {hint}")


###############################################################################
# Public API
###############################################################################

def cast_input(raw: Any, task, ctx: Dict[str, Any] | None = None) -> Any:
    if isinstance(raw, str):
        raw = parse_string(raw)

    try:
        sig = inspect.signature(task.solve)
        prm = sig.parameters.get("problem")
        if prm and prm.annotation is not inspect.Parameter.empty:
            return cast_nested_value(raw, prm.annotation)
    except (ValueError, TypeError, AttributeError):
        pass
    return raw


def cast_result(res: Any, task) -> Any:
    if res is None:
        return None

    hint = getattr(task, "output_type", None)
    if not hint and hasattr(task, "is_solution"):
        try:
            for p in inspect.signature(task.is_solution).parameters.values():
                if (
                    p.name == "solution"
                    and p.annotation is not inspect.Parameter.empty
                ):
                    hint = p.annotation
                    break
        except (ValueError, TypeError):
            pass

    return cast_nested_value(res, hint) if hint else res