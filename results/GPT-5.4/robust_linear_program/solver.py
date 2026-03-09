from typing import Any

import ecos
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

class Solver:
    def __init__(self) -> None:
        self._cache: dict[
            tuple[int, int],
            tuple[sparse.csc_matrix, np.ndarray, np.ndarray, dict[str, Any], np.ndarray],
        ] = {}

    def _workspace(
        self, m: int, n: int
    ) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
        key = (m, n)
        ws = self._cache.get(key)
        if ws is None:
            rows_per_col = m * (n + 1)
            indices = np.tile(np.arange(rows_per_col, dtype=np.int32), n)
            indptr = np.arange(0, (n + 1) * rows_per_col, rows_per_col, dtype=np.int32)
            data = np.empty(n * rows_per_col, dtype=float)
            g = sparse.csc_matrix((data, indices, indptr), shape=(rows_per_col, n))
            h = np.zeros(rows_per_col, dtype=float)
            top_rows = np.arange(0, rows_per_col, n + 1, dtype=np.int32)
            dims = {"l": 0, "q": [n + 1] * m, "e": 0}
            data_view = g.data.reshape(n, m, n + 1)
            ws = (g, h, top_rows, dims, data_view)
            self._cache[key] = ws
        return ws

    def _fail(self, n: int) -> dict[str, Any]:
        return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

    def _solve_scalar(
        self, c: float, b: np.ndarray, q: np.ndarray, p: np.ndarray
    ) -> dict[str, Any] | None:
        m = b.shape[0]
        a = np.abs(p.reshape(m))
        qv = q.reshape(m)
        best_x = None
        best_obj = None
        eps = 1e-12

        for positive in (True, False):
            if positive:
                coeffs = qv + a
                lower = 0.0
                upper = np.inf
            else:
                coeffs = qv - a
                lower = -np.inf
                upper = 0.0

            feasible = True
            for di, bi in zip(coeffs, b):
                if di > eps:
                    val = bi / di
                    if val < upper:
                        upper = val
                elif di < -eps:
                    val = bi / di
                    if val > lower:
                        lower = val
                elif bi < -1e-12:
                    feasible = False
                    break

            if (not feasible) or lower > upper + 1e-12:
                continue

            if c > 0.0:
                if not np.isfinite(lower):
                    return self._fail(1)
                x = lower
            elif c < 0.0:
                if not np.isfinite(upper):
                    return self._fail(1)
                x = upper
            else:
                if lower <= 0.0 <= upper:
                    x = 0.0
                elif np.isfinite(lower):
                    x = lower
                elif np.isfinite(upper):
                    x = upper
                else:
                    x = 0.0

            obj = c * x
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_x = x

        if best_x is None:
            return self._fail(1)
        return {"objective_value": float(best_obj), "x": np.array([best_x], dtype=float)}

    def solve(self, problem, **kwargs) -> Any:
        c = np.asarray(problem["c"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)
        p = np.asarray(problem["P"], dtype=float)

        n = c.shape[0]
        m = b.shape[0]

        if m == 0:
            return self._fail(n)

        if n == 1:
            sol = self._solve_scalar(float(c[0]), b, q, p)
            if sol is not None:
                return sol

        if not np.any(p):
            try:
                res = linprog(c, A_ub=q, b_ub=b, bounds=[(None, None)] * n, method="highs")
                if res.success and res.x is not None:
                    x = np.asarray(res.x, dtype=float)
                    obj = float(c @ x)
                    if np.isfinite(obj) and np.all(np.isfinite(x)):
                        return {"objective_value": obj, "x": x}
                return self._fail(n)
            except Exception:
                pass

        try:
            g, h, top_rows, dims, data_view = self._workspace(m, n)
            h[top_rows] = b
            data_view[:, :, 0] = q.T
            data_view[:, :, 1:] = -p.transpose(1, 0, 2)

            sol = ecos.solve(c, g, h, dims, verbose=False)
            x = sol.get("x")
            exit_flag = sol.get("info", {}).get("exitFlag")

            if x is None or exit_flag not in (0, 10):
                return self._fail(n)

            obj = float(c @ x)
            if not np.isfinite(obj) or not np.all(np.isfinite(x)):
                return self._fail(n)

            return {"objective_value": obj, "x": x}
        except Exception:
            return self._fail(n)