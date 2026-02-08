import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, vstack

class Solver:
    def solve(self, problem, **kwargs):
        P = np.array(problem["P"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        c_min = np.array(problem["c"], dtype=np.float64)
        T = np.array(problem["T"], dtype=np.float64)

        m, n = P.shape
        num_d = m * n
        num_vars = num_d + m  # D variables + s variables

        # Objective: minimize -sum(s)
        c_obj = np.zeros(num_vars)
        c_obj[num_d:] = -1.0

        # === Constraint 1: s_i - R_i * sum_t P_it * D_it <= 0 (m rows) ===
        i_idx1 = np.repeat(np.arange(m), n)
        t_idx1 = np.tile(np.arange(n), m)
        cols_d1 = i_idx1 * n + t_idx1
        data_d1 = -R[i_idx1] * P[i_idx1, t_idx1]

        rows1 = np.concatenate([i_idx1, np.arange(m)])
        cols1 = np.concatenate([cols_d1, num_d + np.arange(m)])
        data1 = np.concatenate([data_d1, np.ones(m)])
        A1 = csc_matrix((data1, (rows1, cols1)), shape=(m, num_vars))
        b1 = np.zeros(m)

        # === Constraint 2: sum_i D_it <= T_t for all t (n rows) ===
        rows2 = np.tile(np.arange(n), m)
        cols2 = (np.repeat(np.arange(m), n) * n + np.tile(np.arange(n), m))
        data2 = np.ones(num_d)
        A2 = csc_matrix((data2, (rows2, cols2)), shape=(n, num_vars))
        b2 = T

        # === Constraint 3: -sum_t D_it <= -c_i for all i (m rows) ===
        rows3 = np.repeat(np.arange(m), n)
        cols3 = rows3 * n + np.tile(np.arange(n), m)
        data3 = -np.ones(num_d)
        A3 = csc_matrix((data3, (rows3, cols3)), shape=(m, num_vars))
        b3 = -c_min

        A_ub = vstack([A1, A2, A3], format='csc')
        b_ub = np.concatenate([b1, b2, b3])

        # Bounds: D_it >= 0, s_i <= B_i (no explicit lower bound on s)
        bounds_lower = np.zeros(num_vars)
        bounds_lower[num_d:] = -np.inf  # s can be negative (won't be at optimum)
        bounds_upper = np.full(num_vars, np.inf)
        bounds_upper[num_d:] = B  # s_i <= B_i

        result = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=list(zip(bounds_lower, bounds_upper)),
            method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8}
        )

        if result.success:
            x = result.x
            D_val = x[:num_d].reshape(m, n)

            clicks = np.sum(P * D_val, axis=1)
            revenue_per_ad = np.minimum(R * clicks, B)
            total_revenue = float(np.sum(revenue_per_ad))

            return {
                "status": "optimal",
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue_per_ad.tolist(),
                "total_revenue": total_revenue,
                "objective_value": total_revenue,
            }
        else:
            return {"status": result.message, "optimal": False}