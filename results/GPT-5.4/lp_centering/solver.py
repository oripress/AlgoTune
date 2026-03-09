from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
from scipy.linalg import qr, solve
from scipy.optimize import root
class Solver:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], tuple[cp.Variable, cp.Parameter, cp.Parameter, cp.Parameter, cp.Problem]] = {}

    def _cvxpy_solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        m, n = A.shape
        key = (m, n)
        cached = self._cache.get(key)
        if cached is None:
            x = cp.Variable(n)
            c_param = cp.Parameter(n)
            A_param = cp.Parameter((m, n))
            b_param = cp.Parameter(m)
            problem = cp.Problem(
                cp.Minimize(c_param @ x - cp.sum(cp.log(x))),
                [A_param @ x == b_param],
            )
            cached = (x, c_param, A_param, b_param, problem)
            self._cache[key] = cached

        x, c_param, A_param, b_param, problem = cached
        c_param.value = c
        A_param.value = A
        b_param.value = b
        problem.solve(solver="CLARABEL", warm_start=True)
        return np.asarray(x.value, dtype=float)

    @staticmethod
    def _reduce_rows(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = A.shape[0]
        if m <= 1:
            return A, b

        _, r, piv = qr(A.T, mode="economic", pivoting=True, check_finite=False)
        if r.size == 0:
            return A[:0], b[:0]

        diag = np.abs(np.diag(r))
        if diag.size == 0:
            return A[:0], b[:0]

        tol = diag.max() * max(A.shape) * np.finfo(A.dtype).eps
        rank = int(np.sum(diag > tol))
        if rank >= m:
            return A, b

        idx = np.sort(piv[:rank])
        return A[idx], b[idx]

    @staticmethod
    def _solve_spd(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return solve(
                M,
                rhs,
                assume_a="pos",
                check_finite=False,
                overwrite_a=False,
                overwrite_b=False,
            )
        except Exception:
            return np.linalg.lstsq(M, rhs, rcond=None)[0]

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        if A.ndim == 1:
            A = A.reshape(1, -1)
        if b.ndim == 0:
            b = b.reshape(1)

        n = c.size
        m = b.size

        if m == 0 or A.size == 0:
            return {"solution": (1.0 / c).tolist()}

        try:
            x_ls, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            if rank == n:
                if np.all(np.isfinite(x_ls)) and np.min(x_ls) > 0.0:
                    if np.max(np.abs(A @ x_ls - b)) <= 1e-10 * (1.0 + np.max(np.abs(b))):
                        return {"solution": x_ls.tolist()}
        except Exception:
            pass

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c @ x - cp.sum(cp.log(x))), [A @ x == b])
        prob.solve(solver="CLARABEL")
        return {"solution": np.asarray(x.value, dtype=float).tolist()}