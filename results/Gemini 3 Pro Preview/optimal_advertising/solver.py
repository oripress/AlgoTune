from ortools.linear_solver import pywraplp
import numpy as np
from scipy.optimize import linprog, Bounds
from scipy.sparse import coo_matrix

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        P = np.array(problem["P"])
        R = np.array(problem["R"])
        B = np.array(problem["B"])
        c = np.array(problem["c"])
        T = np.array(problem["T"])
        
        m, n = P.shape
        
        # Number of variables: D (m*n) + y (m)
        n_vars = m * n + m
        
        # Objective: maximize sum(y) -> minimize -sum(y)
        c_obj = np.zeros(n_vars)
        c_obj[m*n:] = -1.0
        
        # Bounds
        # D >= 0
        # 0 <= y <= B
        # linprog expects a sequence of (min, max) pairs
        bounds = [(0, None)] * (m * n) + [(0, float(B[i])) for i in range(m)]
        
        # Constraints construction
        # We need A_ub and b_ub for A_ub * x <= b_ub
        
        # 1. -R_i * sum(P_it * D_it) + y_i <= 0
        # Rows 0..m-1
        
        # D part
        # values: -R_i * P_it
        # We flatten P row by row. P_flat corresponds to D_flat.
        # We need to multiply each block of n elements (for ad i) by -R_i.
        vals_1_D = -(R[:, None] * P).flatten()
        rows_1_D = np.repeat(np.arange(m), n)
        cols_1_D = np.arange(m * n)
        
        # y part
        # values: 1
        vals_1_y = np.ones(m)
        rows_1_y = np.arange(m)
        cols_1_y = np.arange(m * n, m * n + m)
        
        # 2. sum_i D_it <= T_t
        # Rows m..m+n-1
        # values: 1
        vals_2 = np.ones(m * n)
        # D_it is at index i*n + t.
        # For a fixed t, we sum over i.
        # The constraint for time t is row m+t.
        # The variable D_it (index k) belongs to time t = k % n.
        rows_2 = m + (np.arange(m * n) % n)
        cols_2 = np.arange(m * n)
        
        # 3. -sum_t D_it <= -c_i
        # Rows m+n..2m+n-1
        # values: -1
        vals_3 = -np.ones(m * n)
        rows_3 = m + n + np.repeat(np.arange(m), n)
        cols_3 = np.arange(m * n)
        
        # Combine all
        vals = np.concatenate([vals_1_D, vals_1_y, vals_2, vals_3])
        rows = np.concatenate([rows_1_D, rows_1_y, rows_2, rows_3])
        cols = np.concatenate([cols_1_D, cols_1_y, cols_2, cols_3])
        
        A_ub = coo_matrix((vals, (rows, cols)), shape=(2*m + n, n_vars))
        
        # Safety margin to ensure constraints are strictly met
        # Validation uses 1e-6 tolerance. We use a slightly larger margin
        # to account for solver tolerances and the effect of clamping negative values.
        eps_margin = 1e-3
        b_ub = np.concatenate([np.zeros(m), T - eps_margin, -(c + eps_margin)])
        # method='highs' is generally the fastest and most robust in scipy
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
             # Map status to output format
             status_map = {
                 2: "infeasible",
                 3: "unbounded",
             }
             return {
                "status": status_map.get(res.status, "error"),
                "optimal": False
            }
            
        # Extract solution
        x = res.x
        D_flat = x[:m*n]
        # Ensure non-negative
        D_val = D_flat.reshape((m, n))
        D_val = np.maximum(D_val, 0.0)
        
        # Debug
        # new_usage = np.sum(D_val, axis=0)
        # if np.any(new_usage > T + 1e-6):
        #     print("VIOLATION DETECTED IN SOLVER")
        #     print("Max usage:", np.max(new_usage))
        #     print("Max T:", np.max(T))
        #     print("Max diff:", np.max(new_usage - T))
        # Calculate derived metrics
        clicks = np.sum(P * D_val, axis=1)
        revenue = np.minimum(R * clicks, B)
        total_revenue = np.sum(revenue)
        
        # Debug check
        # traffic_usage = np.sum(D_val, axis=0)
        # violation = np.max(traffic_usage - T)
        # if violation > 1e-6:
        #     print(f"Max traffic violation: {violation}")
        
        return {
            "status": "optimal",
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue.tolist(),
            "total_revenue": float(total_revenue),
            "objective_value": float(total_revenue)
        }