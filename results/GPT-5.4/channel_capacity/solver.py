import math
from typing import Any

import numpy as np

_LN2 = math.log(2.0)
_TINY = 1e-300

def _capacity_bits(P: np.ndarray, c_nats: np.ndarray, x: np.ndarray) -> float:
    y = P @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        logy = np.log(np.maximum(y, _TINY))
    return float(np.dot(x, c_nats - P.T @ logy) / _LN2)

def _solve_blahut_arimoto(
    P: np.ndarray, c_nats: np.ndarray, tol: float = 1e-10, max_iter: int = 50000
):
    n = P.shape[1]
    x = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iter):
        y = P @ x
        with np.errstate(divide="ignore", invalid="ignore"):
            logy = np.log(np.maximum(y, _TINY))

        d = c_nats - P.T @ logy
        lower = float(np.dot(x, d))

        dmax = float(np.max(d))
        weights = x * np.exp(d - dmax)
        z = float(np.sum(weights))
        if not np.isfinite(z) or z <= 0.0:
            return None

        upper = dmax + math.log(z)
        gap = upper - lower
        x = weights / z
        if gap <= tol:
            return x, lower / _LN2, gap

    y = P @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        logy = np.log(np.maximum(y, _TINY))
    d = c_nats - P.T @ logy
    lower = float(np.dot(x, d))
    dmax = float(np.max(d))
    upper = dmax + math.log(float(np.sum(x * np.exp(d - dmax))))
    return x, lower / _LN2, upper - lower

def _solve_cvxpy(P: np.ndarray, c_nats: np.ndarray):
    try:
        import cvxpy as cp
    except Exception:
        return None

    n = P.shape[1]
    x = cp.Variable(n)
    y = P @ x
    objective = cp.Maximize(cp.sum(cp.entr(y)) / _LN2 + (c_nats @ x) / _LN2)
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)

    for solver, kwargs in (
        (cp.ECOS, {"abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9, "warm_start": True}),
        (None, {"warm_start": True}),
    ):
        try:
            if solver is None:
                prob.solve(**kwargs)
            else:
                prob.solve(solver=solver, **kwargs)
        except Exception:
            continue
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and prob.value is not None:
            x_val = np.asarray(x.value, dtype=np.float64).reshape(-1)
            if x_val.shape != (n,) or not np.all(np.isfinite(x_val)):
                return None
            x_val = np.maximum(x_val, 0.0)
            s = float(np.sum(x_val))
            if not np.isfinite(s) or s <= 0.0:
                return None
            return x_val / s

    return None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        if problem is None or not isinstance(problem, dict) or "P" not in problem:
            return None

        try:
            P = np.asarray(problem["P"], dtype=np.float64)
        except Exception:
            return None

        if P.ndim != 2:
            return None

        m, n = P.shape
        if m <= 0 or n <= 0:
            return None
        if not np.all(np.isfinite(P)):
            return None
        if np.any(P < 0.0):
            return None
        if not np.allclose(P.sum(axis=0), 1.0, atol=1e-6):
            return None

        if n == 1:
            return {"x": [1.0], "C": 0.0}

        row_mask = np.any(P > 0.0, axis=1)
        if not np.all(row_mask):
            P = P[row_mask]

        P = np.ascontiguousarray(P, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            logP = np.where(P > 0.0, np.log(P), 0.0)
        c_nats = np.sum(P * logP, axis=0)

        x = _solve_cvxpy(P, c_nats)
        if x is None:
            ba = _solve_blahut_arimoto(P, c_nats)
            if ba is None:
                return None
            x, _, _ = ba

        x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
        s = float(np.sum(x))
        if not np.isfinite(s) or s <= 0.0:
            return None
        x = x / s

        C = _capacity_bits(P, c_nats, x)
        if not math.isfinite(C):
            return None

        return {"x": x.tolist(), "C": float(C)}