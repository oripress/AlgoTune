from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fast LP-based solver for the optimal advertising problem.

        Reformulation:
        Let Y_i = sum_t P_it * D_it (clicks for ad i)
        Let Z_i = min{R_i * Y_i, B_i} (revenue for ad i)

        Maximize sum_i Z_i
        subject to:
            - D_it >= 0
            - sum_i D_it <= T_t                (traffic capacity per time slot)
            - sum_t D_it >= c_i                (minimum display per ad)
            - Y_i - sum_t P_it * D_it = 0
            - Z_i <= R_i * Y_i
            - Z_i <= B_i

        This is a linear program and is solved with HiGHS.
        """
        # Extract parameters
        P = np.array(problem["P"], dtype=float)
        R = np.array(problem["R"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        c = np.array(problem["c"], dtype=float)
        T = np.array(problem["T"], dtype=float)

        # Dimensions
        m, n = P.shape

        # Variable layout:
        # D_it: size m*n, index d_idx(i, t) = i*n + t
        # Y_i: size m, indices num_D + i
        # Z_i: size m, indices num_D + m + i
        num_D = m * n
        num_Y = m
        num_Z = m
        num_vars = num_D + num_Y + num_Z

        # Objective: maximize sum Z_i -> minimize -sum Z_i
        c_obj = np.zeros(num_vars, dtype=float)
        c_obj[num_D + num_Y :] = -1.0

        # Bounds
        # D_it >= 0; Y_i free; Z_i free
        bounds = [(0.0, None)] * num_D + [(None, None)] * num_Y + [(None, None)] * num_Z

        # Helper for D indices
        # idx_D[i, t] = i*n + t
        idx_D = (np.arange(m)[:, None] * n + np.arange(n)[None, :])

        # Build A_ub x <= b_ub
        rows_ub = []
        cols_ub = []
        data_ub = []
        b_ub_list = []

        # 1) Traffic per time slot: sum_i D_it <= T_t  (n rows)
        rows_time = np.repeat(np.arange(n), m)
        cols_time = idx_D.ravel(order="F")  # groups by t across i
        data_time = np.ones(m * n, dtype=float)

        rows_ub.extend(rows_time.tolist())
        cols_ub.extend(cols_time.tolist())
        data_ub.extend(data_time.tolist())
        b_ub_list.append(T.astype(float))

        # 2) Minimum displays per ad: -sum_t D_it <= -c_i  (m rows)
        rows_min = n + np.repeat(np.arange(m), n)
        cols_min = idx_D.ravel(order="C")  # groups by i across t
        data_min = -np.ones(m * n, dtype=float)

        rows_ub.extend(rows_min.tolist())
        cols_ub.extend(cols_min.tolist())
        data_ub.extend(data_min.tolist())
        b_ub_list.append(-c.astype(float))

        # 3) Z_i <= R_i * Y_i  ->  -R_i * Y_i + Z_i <= 0  (m rows)
        rows_zry = n + m + np.arange(m)
        # Y coefficients
        cols_ub.extend((num_D + np.arange(m)).tolist())
        rows_ub.extend(rows_zry.tolist())
        data_ub.extend((-R).tolist())
        # Z coefficients
        cols_ub.extend((num_D + m + np.arange(m)).tolist())
        rows_ub.extend(rows_zry.tolist())
        data_ub.extend(np.ones(m, dtype=float).tolist())
        b_ub_list.append(np.zeros(m, dtype=float))

        # 4) Z_i <= B_i  (m rows)
        rows_zb = n + m + m + np.arange(m)
        cols_ub.extend((num_D + m + np.arange(m)).tolist())
        rows_ub.extend(rows_zb.tolist())
        data_ub.extend(np.ones(m, dtype=float).tolist())
        b_ub_list.append(B.astype(float))

        A_ub = coo_matrix(
            (data_ub, (rows_ub, cols_ub)),
            shape=(n + m + m + m, num_vars),
        )
        b_ub = np.concatenate(b_ub_list, axis=0)

        # Build A_eq x = b_eq for Y_i - sum_t P_it * D_it = 0  (m rows)
        rows_eq = []
        cols_eq = []
        data_eq = []

        # D part: -P_it on D_it for each i
        rows_eq.extend(np.repeat(np.arange(m), n).tolist())
        cols_eq.extend(idx_D.ravel(order="C").tolist())
        data_eq.extend((-P.ravel(order="C")).tolist())

        # Y part: +1 on Y_i
        rows_eq.extend(np.arange(m).tolist())
        cols_eq.extend((num_D + np.arange(m)).tolist())
        data_eq.extend(np.ones(m, dtype=float).tolist())

        A_eq = coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(m, num_vars))
        b_eq = np.zeros(m, dtype=float)

        # Solve LP
        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if not res.success:
            return {"status": res.message, "optimal": False}

        x = res.x
        D_flat = x[:num_D]
        D = D_flat.reshape((m, n))
        # Clean small negatives due to numerical tolerance
        D[D < 0] = 0.0

        # Compute clicks and revenue
        clicks = (P * D).sum(axis=1)
        revenue_per_ad = np.minimum(R * clicks, B)
        total_revenue = float(revenue_per_ad.sum())

        return {
            "status": "optimal",
            "optimal": True,
            "displays": D.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": total_revenue,
            "objective_value": float(-res.fun),
        }