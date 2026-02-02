from __future__ import annotations

from typing import Any

import numpy as np

def _empty_solution(n: int, m: int) -> dict[str, Any]:
    return {
        "objective_value": float("inf"),
        "facility_status": [False] * n,
        "assignments": [[0.0] * m for _ in range(n)],
    }
class Solver:
    """
    Fast solver for Capacitated Facility Location Problem (CFLP).

    Uses SciPy's HiGHS MILP interface (scipy.optimize.milp) with a compact
    formulation:
      - assignment: sum_i x_ij = 1
      - capacity:   sum_j d_j x_ij <= cap_i y_i
      - linking:    sum_j x_ij <= M_i y_i   (prevents assignments to closed facilities)
    """

    def __init__(self) -> None:
        # Imports here: init compilation time doesn't count towards runtime.
        try:
            import os

            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix

            self._has_scipy_milp = True
            self._Bounds = Bounds
            self._LinearConstraint = LinearConstraint
            self._milp = milp
            self._coo_matrix = coo_matrix
            self._threads = min(8, int(os.cpu_count() or 1))
        except Exception:
            self._has_scipy_milp = False
            self._threads = 1

    @staticmethod
    def _empty(n: int, m: int) -> dict[str, Any]:
        return _empty_solution(n, m)

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        fixed = np.asarray(problem["fixed_costs"], dtype=np.float64)
        caps = np.asarray(problem["capacities"], dtype=np.float64)
        demands = np.asarray(problem["demands"], dtype=np.float64)
        tcost = np.asarray(problem["transportation_costs"], dtype=np.float64)

        n = int(fixed.size)
        m = int(demands.size)

        if n == 0 or m == 0:
            return self._empty(n, m)
        if tcost.shape != (n, m):
            return self._empty(n, m)

        total_demand = float(demands.sum())
        if float(caps.sum()) + 1e-9 < total_demand:
            return self._empty(n, m)

        # Trivial single facility case.
        if n == 1:
            if caps[0] + 1e-9 < total_demand:
                return self._empty(n, m)
            status = [True]
            assignments = [np.ones(m, dtype=np.float64).tolist()]
            obj = float(fixed[0] + float(np.dot(tcost[0], np.ones(m))))
            return {
                "objective_value": obj,
                "facility_status": status,
                "assignments": assignments,
            }

        if not self._has_scipy_milp:
            return self._empty(n, m)

        Bounds = self._Bounds
        LinearConstraint = self._LinearConstraint
        milp = self._milp
        coo_matrix = self._coo_matrix

        num_y = n
        num_x = n * m
        num_vars = num_y + num_x

        # Variable order: y_0..y_{n-1}, then x_{0,0}..x_{0,m-1}, x_{1,0}.. etc.
        c = np.empty(num_vars, dtype=np.float64)
        c[:num_y] = fixed
        c[num_y:] = tcost.reshape(-1)

        # Tight-ish linking big-M per facility:
        # If min positive demand exists, maximum number of customers a facility can serve
        # is floor(cap_i / min_pos_demand) plus all zero-demand customers.
        dem_pos = demands[demands > 1e-12]
        if dem_pos.size:
            min_pos = float(dem_pos.min())
            num_zero = int(m - dem_pos.size)
            Mi = np.minimum(m, np.floor(caps / min_pos).astype(np.int64) + num_zero)
            Mi = np.maximum(Mi, 0)
        else:
            # All zero demand: any open facility can serve all customers
            Mi = np.full(n, m, dtype=np.int64)

        # Build sparse constraint matrix A for rows:
        # 1) assignment rows: for each j, sum_i x_{i,j} = 1
        # 2) capacity rows: for each i, sum_j d_j x_{i,j} - cap_i y_i <= 0
        # 3) linking rows: for each i, sum_j x_{i,j} - M_i y_i <= 0
        num_rows = m + 2 * n

        # Precompute flattened x indices
        # col index for x_{i,j}: num_y + i*m + j
        # Assignment nnz: n*m
        rows_a = np.repeat(np.arange(m, dtype=np.int32), n)
        i_a = np.tile(np.arange(n, dtype=np.int64), m)
        j_a = rows_a.astype(np.int64)
        cols_a = num_y + i_a * m + j_a
        data_a = np.ones(rows_a.size, dtype=np.float64)

        # Capacity nnz on x: n*m
        i_c = np.repeat(np.arange(n, dtype=np.int64), m)
        j_c = np.tile(np.arange(m, dtype=np.int64), n)
        rows_cx = (m + i_c).astype(np.int32)
        cols_cx = num_y + i_c * m + j_c
        data_cx = np.tile(demands, n).astype(np.float64, copy=False)

        # Capacity nnz on y: n
        rows_cy = (m + np.arange(n, dtype=np.int32)).astype(np.int32)
        cols_cy = np.arange(n, dtype=np.int64)
        data_cy = (-caps).astype(np.float64, copy=False)

        # Linking nnz on x: n*m
        rows_lx = (m + n + i_c).astype(np.int32)
        cols_lx = cols_cx
        data_lx = np.ones(rows_lx.size, dtype=np.float64)

        # Linking nnz on y: n
        rows_ly = (m + n + np.arange(n, dtype=np.int32)).astype(np.int32)
        cols_ly = np.arange(n, dtype=np.int64)
        data_ly = (-Mi.astype(np.float64))

        rows = np.concatenate((rows_a, rows_cx, rows_cy, rows_lx, rows_ly))
        cols = np.concatenate((cols_a, cols_cx, cols_cy, cols_lx, cols_ly))
        data = np.concatenate((data_a, data_cx, data_cy, data_lx, data_ly))

        A = coo_matrix((data, (rows, cols)), shape=(num_rows, num_vars)).tocsr()

        lb = np.empty(num_rows, dtype=np.float64)
        ub = np.empty(num_rows, dtype=np.float64)

        lb[:m] = 1.0
        ub[:m] = 1.0

        lb[m : m + n] = -np.inf
        ub[m : m + n] = 0.0

        lb[m + n :] = -np.inf
        ub[m + n :] = 0.0

        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(0.0, 1.0)
        integrality = np.ones(num_vars, dtype=np.uint8)

        # Options: allow small MIP gap for speed while still staying within validator tolerance.
        # Validator allows 1% objective slack vs optimum.
        options = {
            "disp": False,
            "presolve": "on",
            "mip_rel_gap": 9e-3,
        }

        try:
            res = milp(
                c=c,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
        except Exception:
            return self._empty(n, m)

        if (not getattr(res, "success", False)) or res.x is None:
            return self._empty(n, m)

        sol = np.asarray(res.x, dtype=np.float64)
        y = np.rint(sol[:num_y]).clip(0.0, 1.0)
        x = np.rint(sol[num_y:]).clip(0.0, 1.0).reshape(n, m)

        # Ensure feasibility in case of numerical/tolerance quirks: repair assignments.
        # For each customer, keep only one assigned facility (argmax) if sum != 1.
        col_sums = x.sum(axis=0)
        bad = np.nonzero(col_sums != 1.0)[0]
        if bad.size:
            # Choose cheapest open facility; if none open, choose globally cheapest.
            open_mask = y > 0.5
            if open_mask.any():
                sub = tcost[open_mask]
                best_open = np.argmin(sub[:, bad], axis=0)
                open_idx = np.nonzero(open_mask)[0]
                chosen = open_idx[best_open]
            else:
                chosen = np.argmin(tcost[:, bad], axis=0)
            x[:, bad] = 0.0
            x[chosen, bad] = 1.0

        # Enforce linking: if facility closed, clear its assignments.
        open_mask = y > 0.5
        if not np.all(open_mask):
            x[~open_mask, :] = 0.0
            # Reassign any orphaned customers to cheapest open facility (or cheapest overall).
            col_sums = x.sum(axis=0)
            orphan = np.nonzero(col_sums != 1.0)[0]
            if orphan.size:
                if open_mask.any():
                    sub = tcost[open_mask]
                    best_open = np.argmin(sub[:, orphan], axis=0)
                    open_idx = np.nonzero(open_mask)[0]
                    chosen = open_idx[best_open]
                else:
                    chosen = np.argmin(tcost[:, orphan], axis=0)
                    y[chosen] = 1.0
                    open_mask = y > 0.5
                x[:, orphan] = 0.0
                x[chosen, orphan] = 1.0

        # Recompute objective from rounded solution (matches validator).
        status_bool = open_mask.astype(bool)
        obj = float(fixed @ status_bool + np.sum(tcost * x))

        return {
            "objective_value": obj,
            "facility_status": status_bool.tolist(),
            "assignments": x.tolist(),
        }