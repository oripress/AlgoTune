import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using a linear programming formulation.
        Returns a dictionary with solution details.
        """
        # Extract problem parameters
        P = np.array(problem["P"], dtype=float)  # shape (m, n)
        R = np.array(problem["R"], dtype=float)  # shape (m,)
        B = np.array(problem["B"], dtype=float)  # shape (m,)
        c = np.array(problem["c"], dtype=float)  # shape (m,)
        T = np.array(problem["T"], dtype=float)  # shape (n,)

        m, n = P.shape

        # Decision variables: D (m*n) flattened, then z (m) revenue per ad
        num_D = m * n
        num_vars = num_D + m

        # Objective: maximize sum(z) -> minimize -sum(z)
        c_obj = np.zeros(num_vars)
        c_obj[num_D:] = -1.0  # coefficients for z variables

        # Bounds: D >= 0, z >= 0
        bounds = [(0, None)] * num_vars

        # Constraints list
        A_ub = []
        b_ub = []

        # 1) Traffic capacity per time slot: sum_i D_{i,t} <= T[t]
        for t in range(n):
            row = np.zeros(num_vars)
            for i in range(m):
                idx = i * n + t
                row[idx] = 1.0
            A_ub.append(row)
            b_ub.append(T[t])

        # 2) Minimum display requirements per ad: -sum_t D_{i,t} <= -c[i]
        for i in range(m):
            row = np.zeros(num_vars)
            for t in range(n):
                idx = i * n + t
                row[idx] = -1.0
            A_ub.append(row)
            b_ub.append(-c[i])

        # 3) Revenue upper bound per ad: z_i <= B_i
        for i in range(m):
            row = np.zeros(num_vars)
            row[num_D + i] = 1.0
            A_ub.append(row)
            b_ub.append(B[i])

        # 4) Revenue limited by clicks: z_i - R_i * sum_t P_{i,t} D_{i,t} <= 0
        for i in range(m):
            row = np.zeros(num_vars)
            row[num_D + i] = 1.0  # z_i coefficient
            for t in range(n):
                idx = i * n + t
                row[idx] = -R[i] * P[i, t]
            A_ub.append(row)
            b_ub.append(0.0)

        # Convert to numpy arrays
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Solve LP
        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not res.success:
            return {"status": "solver_error", "optimal": False, "message": res.message}

        # Extract solution
        x = res.x
        D_val = x[:num_D].reshape((m, n))
        z_val = x[num_D:]

        # Compute clicks per ad
        clicks = (P * D_val).sum(axis=1)

        return {
            "status": "optimal",
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": z_val.tolist(),
            "total_revenue": float(z_val.sum()),
            "objective_value": float(-res.fun)  # since we minimized negative revenue
        }