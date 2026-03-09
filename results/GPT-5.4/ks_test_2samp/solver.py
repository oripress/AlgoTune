from __future__ import annotations

import ast
from functools import lru_cache
import math
from typing import Any

import numpy as np
from scipy.stats import distributions

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

try:
    from scipy.stats._stats_py import _attempt_exact_2kssamp, _compute_prob_outside_square
except Exception:  # pragma: no cover
    _attempt_exact_2kssamp = None
    _compute_prob_outside_square = None

if njit is not None:
    @njit(cache=True)
    def _ks_h_equal_sorted_numba(x: np.ndarray, y: np.ndarray) -> int:
        n = x.size
        i = 0
        j = 0
        max_d = 0
        min_d = 0

        while i < n and j < n:
            if x[i] <= y[j]:
                v = x[i]
            else:
                v = y[j]

            while i < n and x[i] == v:
                i += 1
            while j < n and y[j] == v:
                j += 1

            delta = i - j
            if delta > max_d:
                max_d = delta
            if delta < min_d:
                min_d = delta

        while i < n:
            v = x[i]
            while i < n and x[i] == v:
                i += 1
            delta = i - j
            if delta > max_d:
                max_d = delta

        while j < n:
            v = y[j]
            while j < n and y[j] == v:
                j += 1
            delta = i - j
            if delta < min_d:
                min_d = delta

        if max_d >= -min_d:
            return max_d
        return -min_d

    @njit(cache=True)
    def _ks_stat_sorted_numba(x: np.ndarray, y: np.ndarray) -> float:
        n1 = x.size
        n2 = y.size
        inv1 = 1.0 / n1
        inv2 = 1.0 / n2
        i = 0
        j = 0
        max_s = 0.0
        min_s = 0.0

        while i < n1 and j < n2:
            if x[i] <= y[j]:
                v = x[i]
            else:
                v = y[j]

            while i < n1 and x[i] == v:
                i += 1
            while j < n2 and y[j] == v:
                j += 1

            delta = i * inv1 - j * inv2
            if delta > max_s:
                max_s = delta
            if delta < min_s:
                min_s = delta

        while i < n1:
            v = x[i]
            while i < n1 and x[i] == v:
                i += 1
            delta = i * inv1 - j * inv2
            if delta > max_s:
                max_s = delta

        while j < n2:
            v = y[j]
            while j < n2 and y[j] == v:
                j += 1
            delta = i * inv1 - j * inv2
            if delta < min_s:
                min_s = delta

        if max_s >= -min_s:
            return max_s
        return -min_s
else:
    _ks_h_equal_sorted_numba = None
    _ks_stat_sorted_numba = None

_equal_tables: dict[int, np.ndarray] = {}
_equal_hits: dict[int, int] = {}
_TABLE_BUILD_THRESHOLD = 2
_TABLE_MAX_N = 1024

def _build_equal_table(n: int) -> np.ndarray | None:
    if _compute_prob_outside_square is None:
        return None
    arr = np.empty(n + 1, dtype=np.float64)
    arr[0] = 1.0
    for h in range(1, n + 1):
        arr[h] = float(_compute_prob_outside_square(n, h))
    _equal_tables[n] = arr
    return arr

def _equal_exact_lookup(n: int, h: int) -> float:
    table = _equal_tables.get(n)
    if table is not None:
        return float(table[h])

    if _compute_prob_outside_square is None:
        return -1.0

    hits = _equal_hits.get(n, 0) + 1
    _equal_hits[n] = hits

    if hits >= _TABLE_BUILD_THRESHOLD and n <= _TABLE_MAX_N:
        table = _build_equal_table(n)
        if table is not None:
            return float(table[h])

    return float(_compute_prob_outside_square(n, h))

@lru_cache(maxsize=None)
def _general_exact_pvalue(n1: int, n2: int, g: int, h: int) -> float:
    if _attempt_exact_2kssamp is None:
        return -1.0
    lcm = (n1 // g) * n2
    d = h / lcm
    success, _, p = _attempt_exact_2kssamp(n1, n2, g, d, "two-sided")
    if success:
        return float(p)
    return -1.0

class Solver:
    __slots__ = ()

    _MAX_AUTO_N = 10_000

    def __init__(self) -> None:
        if _ks_h_equal_sorted_numba is not None:
            a = np.array([0.0], dtype=np.float64)
            _ks_h_equal_sorted_numba(a, a)
        if _ks_stat_sorted_numba is not None:
            a = np.array([0.0], dtype=np.float64)
            _ks_stat_sorted_numba(a, a)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        if isinstance(problem, str):
            problem = ast.literal_eval(problem)

        x = np.asarray(problem["sample1"], dtype=np.float64)
        y = np.asarray(problem["sample2"], dtype=np.float64)
        x.sort()
        y.sort()

        n1 = int(x.size)
        n2 = int(y.size)

        if (
            max(n1, n2) <= self._MAX_AUTO_N
            and _compute_prob_outside_square is None
            and _attempt_exact_2kssamp is None
        ):
            from scipy.stats import ks_2samp

            r = ks_2samp(x, y, method="auto")
            return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}

        if n1 == n2 and _ks_h_equal_sorted_numba is not None:
            h = int(_ks_h_equal_sorted_numba(x, y))
            d = h / n1
            if h == 0:
                p = 1.0
            elif n1 <= self._MAX_AUTO_N:
                p = _equal_exact_lookup(n1, h)
                if p < 0.0:
                    p = self._pvalue_asymp(n1, n2, d)
            else:
                p = self._pvalue_asymp(n1, n2, d)
        else:
            d = self._statistic(x, y)
            if max(n1, n2) <= self._MAX_AUTO_N:
                p = self._pvalue_exact(n1, n2, d)
            else:
                p = self._pvalue_asymp(n1, n2, d)

        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0

        return {"statistic": float(d), "pvalue": float(p)}

    @staticmethod
    def _statistic(x: np.ndarray, y: np.ndarray) -> float:
        if _ks_stat_sorted_numba is not None:
            return float(_ks_stat_sorted_numba(x, y))

        n1 = x.size
        n2 = y.size
        inv1 = 1.0 / n1
        inv2 = 1.0 / n2
        i = 0
        j = 0
        max_s = 0.0
        min_s = 0.0

        while i < n1 and j < n2:
            v = x[i] if x[i] <= y[j] else y[j]
            while i < n1 and x[i] == v:
                i += 1
            while j < n2 and y[j] == v:
                j += 1
            delta = i * inv1 - j * inv2
            if delta > max_s:
                max_s = delta
            if delta < min_s:
                min_s = delta

        while i < n1:
            v = x[i]
            while i < n1 and x[i] == v:
                i += 1
            delta = i * inv1 - j * inv2
            if delta > max_s:
                max_s = delta

        while j < n2:
            v = y[j]
            while j < n2 and y[j] == v:
                j += 1
            delta = i * inv1 - j * inv2
            if delta < min_s:
                min_s = delta

        return float(max(max_s, -min_s))

    @staticmethod
    def _pvalue_asymp(n1: int, n2: int, d: float) -> float:
        m = float(max(n1, n2))
        n = float(min(n1, n2))
        en = m * n / (m + n)
        return float(distributions.kstwo.sf(d, round(en)))

    @staticmethod
    def _pvalue_exact(n1: int, n2: int, d: float) -> float:
        if d <= 0.0:
            return 1.0

        g = math.gcd(n1, n2)
        lcm = (n1 // g) * n2
        h = int(np.round(d * lcm))

        if h == 0:
            return 1.0

        if n1 == n2:
            p = _equal_exact_lookup(n1, h)
            if p >= 0.0:
                return p
        else:
            p = _general_exact_pvalue(n1, n2, g, h)
            if p >= 0.0:
                return p

        return Solver._pvalue_asymp(n1, n2, h / lcm)