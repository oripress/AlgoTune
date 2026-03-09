from typing import Any

import numpy as np

class Solver:
    __slots__ = ()

    def solve(self, problem, **kwargs) -> Any:
        try:
            u = problem["u"]
            v = problem["v"]
            n = len(u)
            if n != len(v):
                raise ValueError("u and v must have equal length")
            if n <= 1:
                return 0.0

            if type(u) is np.ndarray and type(v) is np.ndarray:
                su = float(u.sum())
                sv = float(v.sum())
                if su == sv:
                    c = np.cumsum(u - v, dtype=np.float64)
                    np.abs(c, out=c)
                    return float(c.sum() / su)
                c = np.cumsum(u / su - v / sv, dtype=np.float64)
                np.abs(c, out=c)
                return float(c.sum())

            c = 0.0
            dist = 0.0
            su = 0.0
            for a, b in zip(u, v):
                su += a
                c += a - b
                dist += c if c >= 0.0 else -c

            if c == 0.0:
                return dist / su

            sv = su - c
            cu = 0.0
            cv = 0.0
            dist = 0.0
            inv_su = 1.0 / su
            inv_sv = 1.0 / sv
            for a, b in zip(u, v):
                cu += a
                cv += b
                diff = cu * inv_su - cv * inv_sv
                dist += diff if diff >= 0.0 else -diff
            return dist
        except Exception:
            try:
                return float(len(problem["u"]))
            except Exception:
                return 0.0