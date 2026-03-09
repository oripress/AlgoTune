from typing import Any

import numpy as np

try:
    from simplex_cy import project as _project_cy
except Exception:
    _project_cy = None

def _project_small_seq(seq, n: int):
    svals = sorted(seq, reverse=True)
    csum = 0.0
    theta = 0.0
    i = 0
    for v in svals:
        i += 1
        csum += v
        t = (csum - 1.0) / i
        if v > t:
            theta = t

    out = [0.0] * n
    for i in range(n):
        v = seq[i] - theta
        out[i] = v if v > 0.0 else 0.0
    return out

def _project_numpy(y) -> np.ndarray:
    y = np.array(y, dtype=np.float64, copy=True).reshape(-1)
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    cssv -= 1.0
    rho = np.count_nonzero(u * np.arange(1, y.size + 1) > cssv)
    theta = cssv[rho - 1] / rho
    y -= theta
    np.maximum(y, 0.0, out=y)
    return y

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y = problem["y"]

        if isinstance(y, np.ndarray):
            if y.ndim != 1:
                y = y.reshape(-1)
            n = y.size

            if n == 0:
                return {"solution": np.empty(0, dtype=np.float64)}
            if n == 1:
                return {"solution": np.array([1.0], dtype=np.float64)}
            if n == 2:
                y0 = float(y[0])
                y1 = float(y[1])
                a = 0.5 * (y0 - y1 + 1.0)
                if a <= 0.0:
                    return {"solution": np.array([0.0, 1.0], dtype=np.float64)}
                if a >= 1.0:
                    return {"solution": np.array([1.0, 0.0], dtype=np.float64)}
                return {"solution": np.array([a, 1.0 - a], dtype=np.float64)}

            if _project_cy is not None:
                return {"solution": _project_cy(y)}
            return {"solution": _project_numpy(y)}

        n = len(y)

        if n == 0:
            return {"solution": []}
        if n == 1:
            return {"solution": [1.0]}
        if n == 2:
            y0 = y[0]
            y1 = y[1]
            a = 0.5 * (y0 - y1 + 1.0)
            if a <= 0.0:
                return {"solution": [0.0, 1.0]}
            if a >= 1.0:
                return {"solution": [1.0, 0.0]}
            return {"solution": [a, 1.0 - a]}

        if n <= 8:
            return {"solution": _project_small_seq(y, n)}

        if _project_cy is not None:
            return {"solution": _project_cy(y)}
        return {"solution": _project_numpy(y)}