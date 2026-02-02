from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---- Optional HiGHS backend (fast LP) ----
try:
    import highspy  # type: ignore

    _HIGHSPY = highspy
    _HIGHS = highspy.Highs()
    try:
        _HIGHS.setOptionValue("output_flag", False)
    except Exception:
        pass
except Exception:  # pragma: no cover
    _HIGHSPY = None
    _HIGHS = None

class _ShapeCache:
    __slots__ = (
        "m",
        "n",
        "mn",
        "num_col",
        "num_row",
        "nnz",
        "starts",
        "index",
        "i_idx",
        "col_cost",
        "col_lower",
        "row_lower",
    )

    def __init__(self, m: int, n: int, inf: float):
        self.m = m
        self.n = n
        self.mn = m * n
        self.num_col = self.mn + m
        self.num_row = n + 2 * m
        self.nnz = 3 * self.mn + m

        # Column starts for CSC: D columns have 3 entries each; y columns have 1.
        starts = np.empty(self.num_col + 1, dtype=np.int64)
        starts[: self.mn + 1] = np.arange(0, 3 * self.mn + 1, 3, dtype=np.int64)
        if m:
            starts[self.mn + 1 :] = 3 * self.mn + np.arange(1, m + 1, dtype=np.int64)
        else:
            starts[self.mn + 1 :] = 3 * self.mn
        self.starts = starts

        # Precompute row indices (structure only depends on m,n).
        index = np.empty(self.nnz, dtype=np.int32)
        if self.mn:
            cols = np.arange(self.mn, dtype=np.int64)
            i = (cols // n).astype(np.int64)
            t = (cols - i * n).astype(np.int64)
            base = 3 * self.mn
            index[0:base:3] = t.astype(np.int64)
            index[1:base:3] = (n + i).astype(np.int64)
            index[2:base:3] = (n + m + i).astype(np.int64)
            if m:
                index[base:] = (n + m + np.arange(m, dtype=np.int64)).astype(np.int64)
            self.i_idx = i.astype(np.int64)
        else:
            if m:
                index[:] = (n + m + np.arange(m, dtype=np.int64)).astype(np.int64)
            self.i_idx = np.empty(0, dtype=np.int64)

        self.index = index

        # Constant vectors
        col_cost = np.zeros(self.num_col, dtype=np.float64)
        col_cost[self.mn :] = -1.0  # minimize -sum y
        self.col_cost = col_cost

        self.col_lower = np.zeros(self.num_col, dtype=np.float64)
        self.row_lower = np.full(self.num_row, -inf, dtype=np.float64)

_SHAPE_CACHE: Dict[Tuple[int, int], _ShapeCache] = {}

def _get_shape_cache(m: int, n: int, inf: float) -> _ShapeCache:
    key = (m, n)
    c = _SHAPE_CACHE.get(key)
    if c is None:
        c = _ShapeCache(m, n, inf)
        _SHAPE_CACHE[key] = c
    return c

class Solver:
    def solve(self, problem: dict, **kwargs) -> Dict[str, Any]:
        P = np.asarray(problem["P"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        c = np.asarray(problem["c"], dtype=np.float64)
        T = np.asarray(problem["T"], dtype=np.float64)

        m, n = P.shape

        if _HIGHS is None or _HIGHSPY is None:
            return self._solve_scipy(P, R, B, c, T)

        highs = _HIGHS
        highspy_mod = _HIGHSPY
        inf = float(highs.getInfinity())

        cache = _get_shape_cache(m, n, inf)
        mn = cache.mn

        # Variable bounds
        col_upper = np.full(cache.num_col, inf, dtype=np.float64)
        if m:
            col_upper[mn:] = B  # 0 <= y_i <= B_i

        # Row upper bounds
        row_upper = np.empty(cache.num_row, dtype=np.float64)
        row_upper[:n] = T
        if m:
            row_upper[n : n + m] = -c
            row_upper[n + m :] = 0.0

        # Sparse values
        value = np.empty(cache.nnz, dtype=np.float64)
        if mn:
            base = 3 * mn
            value[0:base:3] = 1.0
            value[1:base:3] = -1.0
            # -(R_i * P_it) for each (i,t) in row-major order
            P_flat = P.reshape(-1)
            value[2:base:3] = -(R[cache.i_idx] * P_flat)
            if m:
                value[base:] = 1.0
        else:
            if m:
                value[:] = 1.0

        # Build model objects (cheap; major savings are from cached structure)
        a_matrix = highspy_mod.HighsSparseMatrix()
        a_matrix.num_col_ = int(cache.num_col)
        a_matrix.num_row_ = int(cache.num_row)
        a_matrix.start_ = cache.starts
        a_matrix.index_ = cache.index
        a_matrix.value_ = value

        lp = highspy_mod.HighsLp()
        lp.num_col_ = int(cache.num_col)
        lp.num_row_ = int(cache.num_row)
        lp.col_cost_ = cache.col_cost
        lp.col_lower_ = cache.col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = cache.row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_ = a_matrix
        lp.sense_ = highspy_mod.ObjSense.kMinimize
        lp.offset_ = 0.0

        highs.clear()
        highs.passModel(lp)
        highs.run()

        status = highs.getModelStatus()
        if status != highspy_mod.HighsModelStatus.kOptimal:
            # In case HiGHS reports other statuses rarely (numerical), fall back.
            return self._solve_scipy(P, R, B, c, T)

        sol = highs.getSolution()
        x = np.asarray(sol.col_value, dtype=np.float64)

        D = x[:mn].reshape(m, n)
        clicks = np.sum(P * D, axis=1)
        revenue_per_ad = np.minimum(R * clicks, B)
        total_revenue = float(np.sum(revenue_per_ad))

        return {
            "status": "optimal",
            "optimal": True,
            "displays": D.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": total_revenue,
            "objective_value": total_revenue,
        }

    @staticmethod
    def _solve_scipy(
        P: np.ndarray, R: np.ndarray, B: np.ndarray, c: np.ndarray, T: np.ndarray
    ) -> Dict[str, Any]:
        from scipy.optimize import linprog
        from scipy.sparse import csc_matrix

        m, n = P.shape
        mn = m * n
        num_col = mn + m
        num_row = n + 2 * m

        # A_ub in CSC
        nnz = 3 * mn + m
        starts = np.empty(num_col + 1, dtype=np.int64)
        starts[: mn + 1] = np.arange(0, 3 * mn + 1, 3, dtype=np.int64)
        starts[mn + 1 :] = 3 * mn + np.arange(1, m + 1, dtype=np.int64)

        index = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float64)

        cols = np.arange(mn, dtype=np.int64)
        i = cols // n
        t = cols - i * n
        base = 3 * mn
        index[0:base:3] = t
        index[1:base:3] = n + i
        index[2:base:3] = n + m + i
        data[0:base:3] = 1.0
        data[1:base:3] = -1.0
        P_flat = P.reshape(-1)
        data[2:base:3] = -(R[i] * P_flat)

        index[base:] = n + m + np.arange(m, dtype=np.int64)
        data[base:] = 1.0

        A = csc_matrix((data, index, starts), shape=(num_row, num_col))
        b = np.concatenate([T, -c, np.zeros(m, dtype=np.float64)])

        cobj = np.zeros(num_col, dtype=np.float64)
        cobj[mn:] = -1.0

        bounds = [(0.0, None)] * mn + [(0.0, float(B[i])) for i in range(m)]

        res = linprog(cobj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not res.success:
            return {"status": str(res.status), "optimal": False}

        x = res.x
        D = x[:mn].reshape(m, n)
        clicks = np.sum(P * D, axis=1)
        revenue_per_ad = np.minimum(R * clicks, B)
        total_revenue = float(np.sum(revenue_per_ad))

        return {
            "status": "optimal",
            "optimal": True,
            "displays": D.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": total_revenue,
            "objective_value": total_revenue,
        }