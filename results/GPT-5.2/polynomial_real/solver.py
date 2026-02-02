from __future__ import annotations

from typing import Any

import math

import numpy as np

_np_roots = np.roots
_np_sort = np.sort
_np_empty = np.empty
_sqrt = math.sqrt

class Solver:
    __slots__ = ()

    def solve(self, problem, **kwargs) -> Any:
        p = problem
        n = len(p)
        if n <= 1:
            return []

        cn = p[-1]

        # Trim leading zeros only if needed.
        if p[0] == 0.0:
            i = 0
            while i < n - 1 and p[i] == 0.0:
                i += 1
            p = p[i:]
            n = len(p)
            if n <= 1:
                return []
            cn = p[-1]

        # Strip exact trailing zeros => roots at 0.
        tz = 0
        if cn == 0.0:
            j = n - 1
            while j > 0 and p[j] == 0.0:
                tz += 1
                j -= 1
            p_nz = p[: n - tz]
        else:
            p_nz = p

        deg = len(p_nz) - 1
        if deg <= 0:
            return ([0.0] * tz) if tz else []

        if deg == 1:
            a, b = p_nz[0], p_nz[1]
            r = -b / a
            if not tz:
                return [r]
            # Descending with zeros inserted.
            return ([r] + [0.0] * tz) if r >= 0.0 else ([0.0] * tz + [r])

        if deg == 2:
            a, b, d = p_nz[0], p_nz[1], p_nz[2]
            disc = b * b - 4.0 * a * d
            if disc < 0.0:
                disc = 0.0
            s = _sqrt(disc)
            inv2a = 0.5 / a
            r1 = (-b + s) * inv2a
            r2 = (-b - s) * inv2a
            if r1 >= r2:
                r_hi, r_lo = r1, r2
            else:
                r_hi, r_lo = r2, r1

            if not tz:
                return [r_hi, r_lo]

            if r_lo >= 0.0:
                return [r_hi, r_lo] + [0.0] * tz
            if r_hi <= 0.0:
                return [0.0] * tz + [r_hi, r_lo]
            # Mixed signs: positives first, then zeros, then negatives.
            return [r_hi] + [0.0] * tz + [r_lo]

        rr = _np_roots(p_nz)
        roots = _np_sort(rr.real)  # ascending float64 (contiguous)

        if tz:
            # Insert tz zeros into the ascending array without re-sorting.
            k = roots.searchsorted(0.0)
            out = _np_empty(roots.size + tz, dtype=np.float64)
            out[:k] = roots[:k]
            out[k : k + tz] = 0.0
            out[k + tz :] = roots[k:]
            return out[::-1].tolist()

        return roots[::-1].tolist()