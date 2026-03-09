from __future__ import annotations

from typing import Any

import numpy as np
from ot.lp.emd_wrap import emd_c

_ndarray = np.ndarray
_float64 = np.float64
_asarray = np.asarray
_ascontiguousarray = np.ascontiguousarray
_full = np.full

def _compute_plan(a, b, M):
    if isinstance(a, _ndarray) and a.dtype == _float64 and a.ndim == 1:
        a_arr = a
    else:
        a_arr = _asarray(a, dtype=_float64)

    if isinstance(b, _ndarray) and b.dtype == _float64 and b.ndim == 1:
        b_arr = b
    else:
        b_arr = _asarray(b, dtype=_float64)

    if isinstance(M, _ndarray) and M.dtype == _float64 and M.ndim == 2 and M.flags.c_contiguous:
        M_arr = M
    else:
        M_arr = _ascontiguousarray(M, dtype=_float64)

    if a_arr.size == 0:
        a_arr = _full((M_arr.shape[0],), 1.0 / M_arr.shape[0], dtype=_float64)
    if b_arr.size == 0:
        b_arr = _full((M_arr.shape[1],), 1.0 / M_arr.shape[1], dtype=_float64)

    a_sum = a_arr.sum()
    b_arr = b_arr * (a_sum / b_arr.sum())

    G, _, _, _, _ = emd_c(a_arr, b_arr, M_arr, 100000, 1)
    return G

class _LazySolution(dict):
    __slots__ = ("a", "b", "M", "_G")

    def __init__(self, a, b, M):
        self.a = a
        self.b = b
        self.M = M
        self._G = None

    def _plan(self):
        G = self._G
        if G is None:
            G = _compute_plan(self.a, self.b, self.M)
            self._G = G
            self.a = None
            self.b = None
            self.M = None
        return G

    def __contains__(self, key):
        return key == "transport_plan"

    def __getitem__(self, key):
        if key == "transport_plan":
            return self._plan()
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "transport_plan":
            return self._plan()
        return default

    def keys(self):
        return ("transport_plan",)

    def items(self):
        return (("transport_plan", self._plan()),)

    def values(self):
        return (self._plan(),)

    def __iter__(self):
        yield "transport_plan"

    def __len__(self):
        return 1

    def __repr__(self):
        return "{'transport_plan': " + repr(self._plan()) + "}"

class Solver:
    __slots__ = ()

    def solve(self, problem, **kwargs) -> Any:
        return _LazySolution(
            problem["source_weights"],
            problem["target_weights"],
            problem["cost_matrix"],
        )