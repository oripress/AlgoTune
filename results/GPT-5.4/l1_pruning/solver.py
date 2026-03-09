from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

import fastproj

@njit(cache=True)
def _scan_and_project(v: np.ndarray, u_sorted_asc: np.ndarray, k: float) -> np.ndarray:
    n = v.size
    theta = 0.0
    cumsum = 0.0
    count = 0

    for pos in range(n - 1, -1, -1):
        count += 1
        val = u_sorted_asc[pos]
        cumsum += val
        t = (cumsum - k) / count
        if val < t:
            theta = t
            break

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = v[i]
        if x > theta:
            out[i] = x - theta
        elif x < -theta:
            out[i] = x + theta
        else:
            out[i] = 0.0
    return out

class Solver:
    __slots__ = ()

    def __init__(self) -> None:
        _scan_and_project(np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64), 0.0)

    def solve(self, problem, **kwargs) -> Any:
        v_in = problem["v"]
        k = float(problem["k"])

        if type(v_in) is list:
            return {"solution": fastproj.project_list(v_in, k)}

        v = np.asarray(v_in, dtype=np.float64).reshape(-1)
        n = v.size
        if n == 0:
            return {"solution": np.empty(0, dtype=np.float64)}

        u = np.abs(v)
        u.sort()
        return {"solution": _scan_and_project(v, u, k)}