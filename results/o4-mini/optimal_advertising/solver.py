import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Extract problem data
        P     = np.array(problem["P"], dtype=float)   # m×n click-through rates
        R     = np.array(problem["R"], dtype=float)   # revenue per click
        B     = np.array(problem["B"], dtype=float)   # budget caps
        c_req = np.array(problem["c"], dtype=float)   # minimum displays per ad
        T     = np.array(problem["T"], dtype=float)   # traffic capacity per slot
        m, n = P.shape

        # Variables: m*n displays D_ij plus m revenues rev_i
        num_D = m * n
        N     = num_D + m
        idD   = np.arange(num_D).reshape((m, n))

        # Objective: maximize sum rev_i -> minimize -sum rev_i
        c_vec = np.zeros(N, dtype=float)
        c_vec[num_D:] = -1.0

        # Offsets for constraint blocks
        off_tc  = 0           # traffic capacity constraints (n rows)
        off_rev = n           # linearization of revenue (m rows)
        off_bud = n + m       # budget cap (m rows)
        off_min = n + 2*m     # minimum display (m*n rows, one per D)

        # 1) Traffic: for each time slot t, sum_i D_it <= T_t
        row_tc  = np.repeat(np.arange(n), m) + off_tc
        col_tc  = idD.T.ravel()
        data_tc = np.ones(num_D)

        # 2) Revenue linearization: rev_i - R_i * sum_j P_ij D_ij <= 0
        #   a) -R_i*P_ij * D_ij
        row_rev1  = np.repeat(np.arange(m), n) + off_rev
        col_rev1  = idD.ravel()
        data_rev1 = -(np.repeat(R, n) * P.ravel())
        #   b) +1 * rev_i
        row_rev2  = np.arange(m) + off_rev
        col_rev2  = num_D + np.arange(m)
        data_rev2 = np.ones(m)

        # 3) Budget cap: rev_i <= B_i
        row_bud  = np.arange(m) + off_bud
        col_bud  = num_D + np.arange(m)
        data_bud = np.ones(m)

        # 4) Minimum displays: -sum_j D_ij <= -c_i
        row_min  = np.repeat(np.arange(m), n) + off_min
        col_min  = idD.ravel()
        data_min = -np.ones(num_D)

        # Assemble all COO entries
        rows = np.concatenate([row_tc, row_rev1, row_rev2, row_bud, row_min])
        cols = np.concatenate([col_tc, col_rev1, col_rev2, col_bud, col_min])
        data = np.concatenate([data_tc, data_rev1, data_rev2, data_bud, data_min])

        A_ub = sp.coo_matrix((data, (rows, cols)), shape=(n + 3*m, N)).tocsr()
        b_ub = np.concatenate([
            T,                        # traffic
            np.zeros(m),             # rev linearization RHS
            B,                        # budget caps
            -c_req                    # minimum displays RHS
        ])

        # Variable bounds: all >= 0
        bounds = [(0, None)] * N

        # Solve LP
        res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            return {"status": res.message, "optimal": False}

        x = res.x
        D = x[:num_D].reshape((m, n))
        rev = x[num_D:]
        clicks = (P * D).sum(axis=1)

        return {
            "status":       "optimal",
            "optimal":      True,
            "displays":     D.tolist(),
            "clicks":       clicks.tolist(),
            "revenue_per_ad": rev.tolist(),
            "total_revenue": float(rev.sum())
        }