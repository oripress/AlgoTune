import numpy as np
from typing import Any, Dict
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Solver:
    def solve(self, problem: dict, **kwargs) -> Dict[str, Any]:
        """
        Solve the optimal advertising allocation using an LP reformulation.

        We introduce y_i = revenue for ad i and D_it = displays.
        Maximize sum_i y_i subject to:
          - D_it >= 0
          - sum_i D_it <= T_t  (traffic per time)
          - sum_t D_it >= c_i  (minimum displays per ad)
          - y_i <= B_i
          - y_i <= R_i * sum_t P_it * D_it  (linearized via y_i - R_i * sum_t P_it * D_it <= 0)

        This is assembled into a single sparse A_ub matrix and solved with HiGHS (scipy.linprog).
        """
        # Extract inputs
        P = np.array(problem.get("P", []), dtype=float)
        R = np.array(problem.get("R", []), dtype=float)
        B = np.array(problem.get("B", []), dtype=float)
        c = np.array(problem.get("c", []), dtype=float)
        T = np.array(problem.get("T", []), dtype=float)

        # Handle empty problem
        if P.size == 0:
            return {
                "displays": [],
                "clicks": [],
                "revenue_per_ad": [],
                "total_revenue": 0.0,
                "optimal": True,
                "status": "empty_problem",
            }

        m, n = P.shape

        # Ensure vectors have expected lengths (resize if necessary)
        if R.size != m:
            R = np.resize(R, m)
        if B.size != m:
            B = np.resize(B, m)
        if c.size != m:
            c = np.resize(c, m)
        if T.size != n:
            T = np.resize(T, n)

        # Variable ordering: first m*n D variables (i * n + t), then m y variables
        num_D = m * n
        num_y = m
        N = num_D + num_y

        # Objective: maximize sum(y) -> minimize -sum(y)
        obj = np.zeros(N, dtype=float)
        obj[num_D:] = -1.0

        # Number of inequality rows: n (traffic) + m (min displays) + m (y<=B) + m (y - R*P*D <= 0)
        total_rows = n + 3 * m

        # 1) Traffic constraints: for each time t: sum_i D_it <= T_t
        # For row t, columns are i*n + t for i=0..m-1
        # Build row and col indices for all entries in this block (length m*n)
        t_idx = np.arange(n, dtype=int)
        i_base = (np.arange(m, dtype=int) * n)  # base offsets for each i
        traffic_rows = np.repeat(t_idx, m)  # [0 repeated m times, 1 repeated m times, ...]
        traffic_cols = (t_idx[:, None] + i_base[None, :]).ravel(order='C')
        traffic_data = np.ones(m * n, dtype=float)
        traffic_b = T.astype(float)

        # 2) Minimum display constraints: for each ad i: -sum_t D_it <= -c_i
        # For row (n + i), columns are i*n + t for t=0..n-1
        min_rows = np.repeat(np.arange(m, dtype=int), n) + n
        min_cols = (np.arange(m, dtype=int)[:, None] * n + np.arange(n, dtype=int)[None, :]).ravel(order='C')
        min_data = -np.ones(m * n, dtype=float)
        min_b = (-c).astype(float)

        # 3) y_i <= B_i
        ycap_rows = n + m + np.arange(m, dtype=int)
        ycap_cols = num_D + np.arange(m, dtype=int)
        ycap_data = np.ones(m, dtype=float)
        ycap_b = B.astype(float)

        # 4) y_i - R_i * sum_t P_it * D_it <= 0
        # D coefficients: for each i,t at column i*n + t: value = -R[i] * P[i,t]
        ycons_row_D = n + 2 * m + np.repeat(np.arange(m, dtype=int), n)
        ycons_col_D = (np.arange(m, dtype=int)[:, None] * n + np.arange(n, dtype=int)[None, :]).ravel(order='C')
        ycons_data_D = (-(R[:, None] * P)).ravel(order='C').astype(float)

        # y coefficient entries (one per ad)
        ycons_row_y = n + 2 * m + np.arange(m, dtype=int)
        ycons_col_y = num_D + np.arange(m, dtype=int)
        ycons_data_y = np.ones(m, dtype=float)
        ycons_b = np.zeros(m, dtype=float)

        # Concatenate indices and data
        row_idx = np.concatenate([traffic_rows, min_rows, ycap_rows, ycons_row_D, ycons_row_y])
        col_idx = np.concatenate([traffic_cols, min_cols, ycap_cols, ycons_col_D, ycons_col_y])
        data = np.concatenate([traffic_data, min_data, ycap_data, ycons_data_D, ycons_data_y])
        b_ub_arr = np.concatenate([traffic_b, min_b, ycap_b, ycons_b])

        # Drop explicit zeros in data to reduce sparse size (keeps rows/b vector intact)
        nz_mask = data != 0.0
        if not np.all(nz_mask):
            row_idx = row_idx[nz_mask]
            col_idx = col_idx[nz_mask]
            data = data[nz_mask]

        # Build sparse A_ub
        try:
            A_ub = coo_matrix((data, (row_idx, col_idx)), shape=(total_rows, N)).tocsr()
        except Exception as e:
            return {"status": "build_matrix_error", "optimal": False, "error": str(e)}

        # Bounds: all variables >= 0
        bounds = [(0, None)] * N

        # Solve LP using HiGHS (fast)
        try:
            res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub_arr, bounds=bounds, method="highs")
        except Exception as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}

        if not getattr(res, "success", False):
            return {
                "status": res.message if hasattr(res, "message") else "failed",
                "optimal": False,
                "displays": None,
                "clicks": None,
                "revenue_per_ad": None,
                "total_revenue": 0.0,
            }

        x = res.x
        # Extract displays and clip small negatives
        D_val = x[:num_D].reshape((m, n))
        D_val = np.maximum(D_val, 0.0)

        # Compute clicks and revenues
        clicks = np.sum(P * D_val, axis=1)
        revenue_per_ad = np.minimum(R * clicks, B)
        total_revenue = float(np.sum(revenue_per_ad))

        return {
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": total_revenue,
            "optimal": True,
            "status": "optimal",
            "objective_value": float(-res.fun) if hasattr(res, "fun") else total_revenue,
        }