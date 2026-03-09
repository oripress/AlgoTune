from typing import Any

import numpy as np
import scipy.odr as odr

def _fallback_line(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> list[float]:
    n = x.size
    if n == 0:
        return [0.0, 1.0]
    if n == 1:
        return [0.0, float(y[0])]
    w = 1.0 / np.maximum(sy, np.finfo(np.float64).tiny) ** 2
    sw = float(np.sum(w))
    xbar = float(np.dot(w, x) / sw)
    ybar = float(np.dot(w, y) / sw)
    u = x - xbar
    denom = float(np.dot(w, u * u))
    if not np.isfinite(denom) or denom <= 0.0:
        return [0.0, ybar]
    m = float(np.dot(w, u * (y - ybar)) / denom)
    return [m, ybar - m * xbar]

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        x = np.asarray(problem.get("x", ()), dtype=np.float64)
        y = np.asarray(problem.get("y", ()), dtype=np.float64)
        sx = np.asarray(problem.get("sx", ()), dtype=np.float64)
        sy = np.asarray(problem.get("sy", ()), dtype=np.float64)

        if x.size != y.size or sx.size != x.size or sy.size != x.size:
            return {"beta": _fallback_line(x, y, sy if sy.size == y.size else np.ones_like(y))}
        if x.size <= 1:
            return {"beta": _fallback_line(x, y, sy)}

        tiny = np.finfo(np.float64).tiny
        sx = np.where(np.isfinite(sx) & (sx > 0.0), sx, tiny)
        sy = np.where(np.isfinite(sy) & (sy > 0.0), sy, tiny)

        try:
            data = odr.RealData(x, y=y, sx=sx, sy=sy)
            model = odr.Model(lambda beta, x0: beta[0] * x0 + beta[1])
            beta = np.asarray(odr.ODR(data, model, beta0=[0.0, 1.0]).run().beta, dtype=np.float64)
            if beta.shape == (2,) and np.all(np.isfinite(beta)):
                return {"beta": beta.tolist()}
        except Exception:
            pass

        return {"beta": _fallback_line(x, y, sy)}